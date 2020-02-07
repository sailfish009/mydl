from collections import OrderedDict
import torch
from torch import nn
# from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, model_zoo, model_urls, resnet34
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, model_urls, resnet34
from torch.utils import model_zoo


class ResNet18MaxProtein(ResNet):
    def __init__(self, num_classes):
        super(ResNet18MaxProtein, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        self.conv1_y = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(512)),
            ('drop1', nn.Dropout(p=0.5)),
            ('linear1', nn.Linear(512, num_classes))
        ]))

    def forward(self, x):
        rgb = x[:, [0, 1, 2], :, :]
        y = x[:, [3], :, :]
        x = self.conv1(rgb)
        x_y = self.conv1_y(y)
        x = x + x_y
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ada_maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet18AvgProtein(ResNet):
    def __init__(self, num_classes):
        super(ResNet18AvgProtein, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        self.conv1_y = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(512)),
            ('drop1', nn.Dropout(p=0.5)),
            ('linear1', nn.Linear(512, num_classes))
        ]))

    def forward(self, x):
        rgb = x[:, [0, 1, 2], :, :]
        y = x[:, [3], :, :]
        x = self.conv1(rgb)
        x_y = self.conv1_y(y)
        x = x + x_y
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ada_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet18MaxAvgProtein(ResNet):
    def __init__(self, num_classes):
        super(ResNet18MaxAvgProtein, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        self.conv1_y = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(1024)),
            ('drop1', nn.Dropout(p=0.5)),
            ('linear1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(512)),
            ('drop2', nn.Dropout(p=0.5)),
            ('linear2', nn.Linear(512, num_classes))
        ]))

    def forward(self, x):
        rgb = x[:, [0, 1, 2], :, :]
        y = x[:, [3], :, :]
        x = self.conv1(rgb)
        x_y = self.conv1_y(y)
        x = x + x_y
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_x = self.ada_avgpool(x)
        max_x = self.ada_maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet183CProtein(ResNet):
    def __init__(self, num_classes):
        super(ResNet183CProtein, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(512)),
            ('drop1', nn.Dropout(p=0.5)),
            ('linear1', nn.Linear(512, num_classes))
        ]))

class ResNet18MaxAvgNoDropout(ResNet18MaxAvgProtein):
    def __init__(self, num_classes):
        super(ResNet18MaxAvgNoDropout, self).__init__(num_classes)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(1024)),
            ('linear1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(512)),
            ('linear2', nn.Linear(512, num_classes))
        ]))


class ResNet34MaxAvgProtein(ResNet):
    def __init__(self, num_classes):
        super(ResNet34MaxAvgProtein, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        self.conv1_y = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(1024)),
            ('drop1', nn.Dropout(p=0.5)),
            ('linear1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(512)),
            ('drop2', nn.Dropout(p=0.5)),
            ('linear2', nn.Linear(512, num_classes))
        ]))

    def forward(self, x):
        rgb = x[:, [0, 1, 2], :, :]
        y = x[:, [3], :, :]
        x = self.conv1(rgb)
        x_y = self.conv1_y(y)
        x = x + x_y
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_x = self.ada_avgpool(x)
        max_x = self.ada_maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet34MaxAvgNoDropout(ResNet34MaxAvgProtein):
    def __init__(self, num_classes):
        super(ResNet34MaxAvgNoDropout, self).__init__(num_classes)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(1024)),
            ('linear1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(512)),
            ('linear2', nn.Linear(512, num_classes))
        ]))


class ResNet34MaxProtein(ResNet):
    def __init__(self, num_classes):
        super(ResNet34MaxProtein, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        self.conv1_y = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(512)),
            ('drop1', nn.Dropout(p=0.5)),
            ('linear1', nn.Linear(512, num_classes))
        ]))

    def forward(self, x):
        rgb = x[:, [0, 1, 2], :, :]
        y = x[:, [3], :, :]
        x = self.conv1(rgb)
        x_y = self.conv1_y(y)
        x = x + x_y
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ada_maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet34Protein3C(ResNet):
    def __init__(self, num_classes):
        super(ResNet34Protein3C, self).__init__(BasicBlock, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(1024)),
            ('drop1', nn.Dropout(p=0.5)),
            ('linear1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(512)),
            ('drop2', nn.Dropout(p=0.5)),
            ('linear2', nn.Linear(512, num_classes))
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_x = self.ada_avgpool(x)
        max_x = self.ada_maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet50Protein(ResNet):
    def __init__(self, num_classes):
        super(ResNet50Protein, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        self.conv1_y = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ada_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(4096)),
            ('drop1', nn.Dropout(p=0.5)),
            ('linear1', nn.Linear(4096, 1024)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(1024)),
            ('drop2', nn.Dropout(p=0.5)),
            ('linear2', nn.Linear(1024, num_classes))
        ]))

    def forward(self, x):
        rgb = x[:, [0, 1, 2], :, :]
        y = x[:, [3], :, :]
        x = self.conv1(rgb)
        x_y = self.conv1_y(y)
        x = x + x_y
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_x = self.ada_avgpool(x)
        max_x = self.ada_maxpool(x)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def test():
    model = ResNet34MaxProtein(28)
    i = torch.randn((16, 4, 128, 128))
    o = model(i)
    print(o.size())

    for key, value in model.named_parameters():
        print(key)


if __name__ == '__main__':
    test()
