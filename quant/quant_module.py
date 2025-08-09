from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



gaussian_steps = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.336, 5: 0.190, 6: 0.106, 7: 0.059, 8: 0.032}
hwgq_steps = {1: 0.799, 2: 0.538, 3: 0.3217, 4: 0.185, 5: 0.104, 6: 0.058, 7: 0.033, 8: 0.019}
factors = {
    1: 28.4,
    2: 16,
    3: 9.1,
    4: 5.3,
    5: 3.2,
    6: 2,
    7: 1.3,
    8: 1
}
class ImageInputQ(nn.Module):
    '''
    Assume image input are discrete value [0/256, 1/256, 2/256, ..., 255/256]
    '''
    def __init__(self, bit = 8):
        super(ImageInputQ, self).__init__()
        self.bit = bit
        self.step = 1/2**bit

    def forward(self, x):
        if self.step==32:
            return out
        out = torch.floor(x/self.step) * self.step 
        return out
class _gauss_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        alpha = x.std().item()
        step *= alpha
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _gauss_quantize_resclaed_step(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _hwgq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step):
        y = torch.round(x / step) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HWGQ(nn.Module):
    def __init__(self, bit=2):
        super(HWGQ, self).__init__()
        self.bit = bit
        if bit < 32:
            self.step = hwgq_steps[bit]
        else:
            self.step = None

    def forward(self, x):
        if self.bit >= 32:
            return x.clamp(min=0.0)
        lvls = float(2 ** self.bit - 1)
        clip_thr = self.step * lvls
        y = x.clamp(min=0.0, max=clip_thr)
        return _hwgq.apply(y, self.step)

class MNISTImageInputQ(nn.Module):
    def __init__(self, bit=8):
        super(MNISTImageInputQ, self).__init__()
        self.bit = bit
        self.step = 1 / 2**bit

    def forward(self, x):
        out = torch.floor(x / self.step) * self.step
        return out


class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.bit = 8
        super(QuantConv2d, self).__init__(*args, **kwargs)
        self.step = gaussian_steps[self.bit]

    def forward(self, input):
        assert self.bias is None 
        quant_weight = _gauss_quantize.apply(self.weight, self.step, self.bit)
        out = F.conv2d(
            input, quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class QuantLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.bit = 8
        super(QuantLinear, self).__init__(*args, **kwargs)
        self.step = gaussian_steps[self.bit]

    def forward(self, input):
        quant_weight = _gauss_quantize.apply(self.weight, self.step, self.bit)
        out = F.linear(input, quant_weight, self.bias)
        return out



class QuantActivConv2d(nn.Module):
    def __init__(self, inplane, outplane, abit=2,ActQ = HWGQ, **kwargs):
        super(QuantActivConv2d, self).__init__()
        self.abit = abit
        self.activ = HWGQ(abit)
        self.conv = QuantConv2d(inplane, outplane,**kwargs)
        stride = kwargs.get('stride', 1)
        kernel_size = kwargs.get('kernel_size', (1, 1))
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0] * kernel_size[1]
        else:
            kernel_size = kernel_size * kernel_size
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        out = self.activ(input)
        out = self.conv(out)
        return out


class QuantActivLinear(nn.Module):
    def __init__(self, inplane, outplane, abit, **kwargs):
        super(QuantActivLinear, self).__init__()
        self.abit = abit
        self.activ = HWGQ(abit)
        self.linear = QuantLinear(inplane, outplane, **kwargs)
        self.param_size = inplane * outplane * 1e-6
        self.register_buffer('size_product', torch.tensor(self.param_size, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        tmp = torch.tensor(input.numel() * input.element_size() * 1e-6, dtype=torch.float)
        self.memory_size.copy_(tmp)
        out = self.activ(input)
        out = self.linear(out)
        return out



class MixQuantActiv(nn.Module):

    def __init__(self, bits, ActQ = HWGQ):
        super(MixQuantActiv, self).__init__()
        self.bits = bits
        self.alpha_activ = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_activ.data.fill_(0.01)
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(HWGQ(bit=bit))

    def forward(self, input):
        outs = []
        sw = F.softmax(self.alpha_activ, dim=0)
        for i, branch in enumerate(self.mix_activ):
            outs.append(branch(input) * sw[i])
        activ = sum(outs)
        return activ


class MixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(MixQuantConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.conv_list = nn.ModuleList()
        self.steps = []
        for bit in self.bits:
            assert 0 < bit < 32
            self.conv_list.append(nn.Conv2d(inplane, outplane, **kwargs))
            self.steps.append(gaussian_steps[bit])

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        for i, bit in enumerate(self.bits):
            weight = self.conv_list[i].weight
            weight_std = weight.std().item()
            step = self.steps[i] * weight_std
            quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        conv = self.conv_list[0]
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


class MixActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, abits=None, ActQ = HWGQ, **kwargs):
        super(MixActivConv2d, self).__init__()
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        # build mix-precision branches for activations
        self.mix_activ = MixQuantActiv(self.abits, ActQ)
        
        # Since weights are fixed at 8 bits, no need for mix_weight
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        out = self.mix_activ(input)
        out = self.conv(out)  # Use the fixed 8-bit weight convolution
        return out

    def complexity_loss(self):
        sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += sw[i] * abits[i] / factors[abits[i]]
        complexity = self.size_product.item() * mix_abit * 8  # Fixed 8-bit weights
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        
        weight_shape = list(self.conv.weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * 8, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, memory_size, mix_abit))
        print('idx {} with shape {}, weight fixed at 8 bits, comp: {:.3f}M * {:.3f} * 8, '
              'param: {:.3f}M * 8'.format(layer_idx, weight_shape, size_product,
                                          mix_abit, self.param_size))
        best_arch = {'best_activ': [best_activ], 'best_weight': [8]}  # Fixed at 8-bit weights
        bitops = size_product * abits[best_activ] * 8
        bita = memory_size * abits[best_activ]
        bitw = self.param_size * 8
        mixbitops = size_product * mix_abit * 8
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * 8
        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw


class MixActivLinear(nn.Module):

    def __init__(self, inplane, outplane, abits=None, **kwargs):
        super(MixActivLinear, self).__init__()
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        # build mix-precision branches for activations
        self.mix_activ = MixQuantActiv(self.abits)
        self.linear = nn.Linear(inplane, outplane, **kwargs)
        # complexities
        self.param_size = inplane * outplane * 1e-6
        self.register_buffer('size_product', torch.tensor(self.param_size, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        tmp = torch.tensor(input.shape[1] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        out = self.mix_activ(input)
        out = self.linear(out)  # Use the fixed 8-bit weight linear layer
        return out

    def complexity_loss(self):
        sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += sw[i] * abits[i] / factors[abits[i]]
        complexity = self.size_product.item() * mix_abit * 8  # Fixed 8-bit weights
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        
        weight_shape = list(self.linear.weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * 8, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, memory_size, mix_abit))
        print('idx {} with shape {}, weight fixed at 8 bits, comp: {:.3f}M * {:.3f} * 8, '
              'param: {:.3f}M * 8'.format(layer_idx, weight_shape, size_product,
                                          mix_abit, self.param_size))
        best_arch = {'best_activ': [best_activ], 'best_weight': [8]}  # Fixed at 8-bit weights
        bitops = size_product * abits[best_activ] * 8
        bita = memory_size * abits[best_activ]
        bitw = self.param_size * 8
        mixbitops = size_product * mix_abit * 8
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * 8
        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw
