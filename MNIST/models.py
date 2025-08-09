import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from quant import quant_module as qm


class LeNet_MixQ(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet_MixQ, self).__init__()
        self.conv_func = qm.MixActivConv2d
        self.linear_func = qm.MixActivLinear
        conv_func = self.conv_func
        linear_func=self.linear_func
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_kwargs1 = {'kernel_size': 5, 'stride': 1, 'padding': 2, 'bias': False}
        conv_kwargs = {'kernel_size': 5, 'stride': 1, 'padding': 0, 'bias': False}
        qspace = {'abits': [1,2, 3, 4, 5, 6, 7, 8]}
        self.layers = nn.Sequential(
            self.conv_func(1, 6, ActQ=qm.MNISTImageInputQ, **conv_kwargs1, **qspace),  
            self.pooling,
            self.conv_func(6, 16, **conv_kwargs, **qspace), 
            self.pooling,
            nn.Flatten(),
            self.linear_func(16*5*5, 120, bias=True,**qspace),
            self.linear_func(120, 84, bias=True,**qspace),
            self.linear_func(84, num_classes, bias=True,**qspace)  
        )

    def forward(self, x):
        return self.layers(x)

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        best_arch = None
        layer_idx = 0

        for m in self.modules():
            if isinstance(m, (self.conv_func, self.linear_func)):
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw= m.fetch_best_arch(layer_idx)
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch

    def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, (self.conv_func,self.linear_func)):
                loss += m.complexity_loss()
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss




class LeNet_FixQ(nn.Module):
    def __init__(self, num_classes=10, bitw=8, bita='88888'):
        super(LeNet_FixQ, self).__init__()
        self.conv_func = qm.QuantActivConv2d
        self.linear_func = qm.QuantActivLinear
        bitw = int(bitw) 
        bita = list(map(int, bita))  

        self.bitw = bitw
        self.bita = bita
        self.model_params = {'bitw': bitw, 'bita': bita}

        print(f'Initialized LeNet_FixQ with bitw: {bitw} and bita: {bita}')
        
        conv_kwargs1 = {'kernel_size': 5, 'stride': 1, 'padding': 2, 'bias': False}
        conv_kwargs = {'kernel_size': 5, 'stride': 1, 'padding': 0, 'bias': False}
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layers = nn.Sequential(
            self.conv_func(1, 6, ActQ=qm.MNISTImageInputQ, **conv_kwargs1,abit=bita[0]),
            self.pooling,
            self.conv_func(6, 16, **conv_kwargs, abit=bita[1]),  
            self.pooling,
            nn.Flatten(),
            self.linear_func(16*5*5, 120, bias=True,abit=bita[2]),  
            self.linear_func(120, 84,bias=True,abit=bita[3]),  
            self.linear_func(84, num_classes, bias=True,abit=bita[4])  
        )

    def forward(self, x):
        return self.layers(x)

    def fetch_arch_info(self):
        sum_bitops, sum_bita = 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, (self.conv_func, self.linear_func)):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                bitops = size_product * m.abit * 8
                bita = memory_size * m.abit
                
                if isinstance(m, self.conv_func):
                    weight_shape = list(m.conv.weight.shape)
                elif isinstance(m, self.linear_func):
                    weight_shape = list(m.linear.weight.shape)
                
                print(f'idx {layer_idx} with shape {weight_shape}, bitops: {size_product:.3f}M * {m.abit} * 8, '
                    f'memory: {memory_size:.3f}K * {m.abit}')
                sum_bitops += bitops
                layer_idx += 1
        return sum_bitops, sum_bita
