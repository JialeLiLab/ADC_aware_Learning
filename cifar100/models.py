import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from quant import quant_module as qm

class InputFactor:
    def __call__(self, pic):
        return pic * 255.0 / 256.0



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, 
                 conv_func=None, qspace=None):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if conv_func is None:
            conv_func = qm.MixActivConv2d
        if qspace is None:
            qspace = {'abits': [1, 2, 3, 4, 5, 6, 7, 8]}

        conv_kwargs1 = {
            'kernel_size': 3, 'stride': stride, 'padding': dilation,
            'bias': False, 'dilation': dilation
        }
        conv_kwargs2 = {
            'kernel_size': 3, 'stride': 1, 'padding': dilation,
            'bias': False, 'dilation': dilation
        }

        self.conv1 = conv_func(inplanes, planes, **conv_kwargs1, **qspace)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_func(planes, planes, **conv_kwargs2, **qspace)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Resnet_MixQ(nn.Module):
    def __init__(self, num_classes=100, block=BasicBlock):
        super(Resnet_MixQ, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv_func = qm.MixActivConv2d
        self.linear_func = qm.MixActivLinear

        conv_kwargs1 = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        qspace = {'abits': [1, 2, 3, 4, 5, 6, 7, 8]}

        self.conv1 = self.conv_func(3, 64, ActQ=qm.ImageInputQ, **conv_kwargs1, **qspace)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2, dilate=False)
        self.layer3 = self._make_layer(block, 256, 6, stride=2, dilate=False)
        self.layer4 = self._make_layer(block, 512, 3, stride=2, dilate=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = self.linear_func(512, num_classes, bias=True, **qspace)

        self.layers = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            self.flatten,
            self.fc
        )
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        conv_func = self.conv_func
        qspace = {'abits': [1, 2, 3, 4, 5, 6, 7, 8]}  

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
            self.conv_func(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, **qspace),
            norm_layer(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, conv_func, qspace))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, conv_func=conv_func, qspace=qspace))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        best_arch = None
        layer_idx = 0

        for m in self.modules():
            if isinstance(m, (self.conv_func,self.linear_func)):
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

    
class Resnet_FixQ(nn.Module):
    def __init__(self, num_classes=100, bitw=8, bita='62212223212231121411', block=None):
        super(Resnet_FixQ, self).__init__()
        if block is None:
            block = BasicBlockFixQ  
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv_func = qm.QuantActivConv2d  
        self.linear_func = qm.QuantActivLinear
        if isinstance(bita, str):
            bita = list(map(int, bita))
        elif isinstance(bita, int):
            bita = [bita]  

        self.bitw = bitw
        self.bita = bita
        self.model_params = {'bitw': bitw, 'bita': bita}

        print(f'Initialized Resnet_FixQ with bitw: {bitw} and bita: {bita}')

        conv_kwargs1 = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        abit_iter = iter(bita)
        
        self.conv1 = self.conv_func(3, 64, ActQ=qm.ImageInputQ, **conv_kwargs1,
                                    abit=next(abit_iter))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, 3, abit_iter=abit_iter)
        self.layer2 = self._make_layer(block, 128, 4, stride=2, abit_iter=abit_iter)
        self.layer3 = self._make_layer(block, 256, 6, stride=2, abit_iter=abit_iter)
        self.layer4 = self._make_layer(block, 512, 3, stride=2, abit_iter=abit_iter)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = self.linear_func (512, num_classes, bias=True, abit=next(abit_iter))

        self.layers = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            self.flatten,
            self.fc
        )
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, abit_iter=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        conv_func = self.conv_func

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_func(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                          bias=False, abit=next(abit_iter)),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, conv_func,
                            abit_iter))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, conv_func=conv_func,
                                abit_iter=abit_iter))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def fetch_arch_info(self):
        sum_bitops, sum_bita = 0, 0
        layer_idx = 0
      
        for m in self.modules():
            if isinstance(m, (self.conv_func, self.linear_func)):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                bitops = size_product * m.abit * self.bitw
                bita = memory_size * m.abit
                bitw = m.param_size * self.bitw
                if isinstance(m, self.conv_func):
                    layer_type = 'Conv'
                    weight_shape = list(m.conv.weight.shape)
                elif isinstance(m, self.linear_func):
                    layer_type = 'Linear'
                    if hasattr(m, 'linear'):
                        weight_shape = list(m.linear.weight.shape)
                    else:
                        weight_shape = list(m.weight.shape)
                
                print(f'idx {layer_idx:<2} | type {layer_type:<6} | shape {str(weight_shape):<20} | '
                    f'bitops: {size_product:.3f}M * {m.abit} * {self.bitw}, '
                    f'memory: {memory_size:.3f}K * {m.abit}')
                
                sum_bitops += bitops
                sum_bita += bita
                layer_idx += 1
                
        return sum_bitops, sum_bita
class BasicBlockFixQ(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, 
                 conv_func=None, abit_iter=None, bitw=8):
        super(BasicBlockFixQ, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if conv_func is None:
            conv_func = qm.QuantActivConv2d

        conv_kwargs1 = {
            'kernel_size': 3, 'stride': stride, 'padding': dilation,
            'bias': False, 'dilation': dilation
        }
        conv_kwargs2 = {
            'kernel_size': 3, 'stride': 1, 'padding': dilation,
            'bias': False, 'dilation': dilation
        }

        self.conv1 = conv_func(inplanes, planes, **conv_kwargs1,
                               abit=next(abit_iter))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_func(planes, planes, **conv_kwargs2,
                               abit=next(abit_iter))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



