import torch
import torch.nn.functional as F
from collections import namedtuple
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

_GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, fc_to_avg=False):
        super(InceptionAux, self).__init__()
        self.fc_to_avg = fc_to_avg
        self.num_classes = num_classes

        if fc_to_avg:
            self.aux_conv = BasicConv2d(in_channels, num_classes, kernel_size=1)
            self.GlobalAvgPool2d = nn.AdaptiveAvgPool2d(1)
        else:
            self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
            self.fc1 = nn.Linear(2048, 1024)
            self.fc2_num = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))

        if self.fc_to_avg:
            x = self.aux_conv(x)
            x = self.GlobalAvgPool2d(x)
            x = x.view(x.size(0), self.num_classes)
        else:
            x = self.conv(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x), inplace=True)
            x = F.dropout(x, 0.7, training=self.training)
            x = self.fc2_num(x)

        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=True, fc_to_avg=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.fc_to_avg = fc_to_avg
        self.num_classes = num_classes
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes, self.fc_to_avg)
            self.aux2 = InceptionAux(528, num_classes, self.fc_to_avg)
        '''
        if fc_to_avg:
            self.conv4 = BasicConv2d(1024, num_classes, kernel_size=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.2)
            self.fc_fin = nn.Linear(1024, num_classes)
        '''
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc_fin = nn.Linear(1024, num_classes)

        # 初始化
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)

        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        '''
        if self.fc_to_avg:
            x = self.conv4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), self.num_classes)
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc_fin(x)
        '''
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc_fin(x)
        if self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        return x


def googlenet(pretrained=False, progress=True, num_classes=200, **kwargs):
    if pretrained:
        model = GoogLeNet(num_classes=200, **kwargs)

        # 读取预训练参数和模型初始化参数
        pretrained_dict = load_state_dict_from_url(model_urls['googlenet'], progress)
        model_dict = model.state_dict()

        # 去除预训练模型中与初始化模型中不一样的参数， 并更新pretrained_dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict)

    else:
        model = GoogLeNet(num_classes=num_classes, **kwargs)
    return model


if __name__ == '__main__':
    net = googlenet(pretrained=True, aux_logits=True, num_classes=200)
    a = torch.rand([1, 3, 448, 448])
    out = net(a)
    print(net)
    print(out, out.shape)