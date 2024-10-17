import torch
import torch.nn as nn
import torchvision.models as models


class Meso4(nn.Module):

    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        # flatten: x = x.view(x.size(0), -1)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, input):
        x = self.conv1(input)  # (8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 128, 128)

        x = self.conv2(x)  # (8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 64, 64)

        x = self.conv3(x)  # (16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling1(x)  # (16, 32, 32)

        x = self.conv4(x)  # (16, 32, 32)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling2(x)  # (16, 8, 8)

        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x)  # (Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MesoInception4(nn.Module):

    def __init__(self, num_classes=2):
        super(MesoInception4, self).__init__()
        self.num_classes = num_classes
        # InceptionLayer1
        self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)

        # InceptionLayer2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        # Normal Layer
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    # InceptionLayer
    def InceptionLayer1(self, input):
        x1 = self.Incption1_conv1(input)
        x2 = self.Incption1_conv2_1(input)
        x2 = self.Incption1_conv2_2(x2)
        x3 = self.Incption1_conv3_1(input)
        x3 = self.Incption1_conv3_2(x3)
        x4 = self.Incption1_conv4_1(input)
        x4 = self.Incption1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)

        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        x2 = self.Incption2_conv2_1(input)
        x2 = self.Incption2_conv2_2(x2)
        x3 = self.Incption2_conv3_1(input)
        x3 = self.Incption2_conv3_2(x3)
        x4 = self.Incption2_conv4_1(input)
        x4 = self.Incption2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)

        return y

    def forward(self, input):
        x = self.InceptionLayer1(input)  # (Batch, 11, 128, 128)
        x = self.InceptionLayer2(x)  # (Batch, 12, 64, 64)

        x = self.conv1(x)  # (Batch, 16, 64 ,64)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (Batch, 16, 32, 32)

        x = self.conv2(x)  # (Batch, 16, 32, 32)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x)  # (Batch, 16, 8, 8)

        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x)  # (Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Vgg(nn.Module):
    def __init__(self, num_class=2):
        super(Vgg, self).__init__()
        vgg16_net = models.vgg16_bn(weights=models.VGG16_BN_Weights)
        self.num_class = num_class
        self.feature = vgg16_net.features
        self.avgpool = vgg16_net.avgpool
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 200, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(200),
            nn.ReLU(True),

            nn.Conv2d(200, self.num_class, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        batchsize = x.size(0)
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = x.view(batchsize, -1)
        return x


class Resnet(nn.Module):
    def __init__(self, num_class=2):
        super(Resnet, self).__init__()
        self.num_class = num_class
        resnet_net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(
            *list(resnet_net.children())[:-1]
        )
        # self.classifier = nn.Linear(2048, self.num_class)  # RESNET50
        self.classifier = nn.Linear(512, self.num_class)  # RESNET18

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class Iception(nn.Module):
    def __init__(self, num_class=2):
        super(Iception, self).__init__()
        self.num_class = num_class
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights)
        aux_in_features = self.inception.AuxLogits.fc.in_features
        in_features = self.inception.fc.in_features
        self.inception.AuxLogits.fc = nn.Linear(aux_in_features, self.num_class)
        self.inception.fc = nn.Linear(in_features, self.num_class)

    def forward(self, x):
        x = self.inception(x)

        return x


class Efficientnet(nn.Module):
    def __init__(self, num_class=2):
        super(Efficientnet, self).__init__()
        self.num_class = num_class
        # resnet_net = models.resnet18(pretrained=True)
        resnet_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            *list(resnet_net.children())[:-1]
        )
        self.classifier = nn.Linear(1280, self.num_class)
        # efficientnet-b0-224 1280
        # efficientnet-b1-240 1280
        # efficientnet-b2-260 1408
        # efficientnet-b3-300 1536
        # efficientnet-b4-380 1792
        # efficientnet-b5-456 2048
        # efficientnet-b6-528 2304
        # efficientnet-b7-600 2560

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class Efficientnetv2(nn.Module):
    def __init__(self, num_class=2):
        super(Efficientnetv2, self).__init__()
        self.num_class = num_class
        resnet_net = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights)
        print(resnet_net)

        self.features = nn.Sequential(
            *list(resnet_net.children())[:-1]
        )

        self.classifier = nn.Linear(1280, self.num_class)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class Squeezenet(nn.Module):
    def __init__(self, num_class=2):
        super(Squeezenet, self).__init__()
        self.num_class = num_class
        net = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights)
        print(net)

        self.features = nn.Sequential(
            *list(net.children())[:-1]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, self.num_class, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x.view(x.size(0), self.num_class)


class VisionTransformer(nn.Module):
    def __init__(self, num_class=2):
        super(VisionTransformer, self).__init__()
        self.num_class = num_class
        self.net = models.vit_b_16(weights=models.ViT_B_16_Weights)
        print(self.net)

    def forward(self, x):
        x = self.net(x)

        return x
