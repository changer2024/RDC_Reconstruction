import argparse
import os

import torchvision
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
from torchattacks import *
from torchvision.utils import save_image

from network.classifier import *
from network.transform import *

os.environ['TORCH_HOME'] = '../pretrain_model'


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def main():
    args = parse.parse_args()
    test_path = args.test_path
    batch_size = args.batch_size
    model_path = args.model_path
    device = args.device
    adv_save_dir = args.adv_save_dir
    torch.backends.cudnn.benchmark = True

    test_dataset = torchvision.datasets.ImageFolder(test_path, transform=vgg_data_transforms['adv'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)

    mean = [0.5] * 3
    std = [0.5] * 3

    norm_layer = Normalize(mean=mean, std=std)

    test_model = Resnet()
    test_model.load_state_dict(torch.load(model_path, map_location=device))

    model = nn.Sequential(
        norm_layer,
        test_model
    )
    model = model.to(device).eval()

    att = FGSM(model, eps=0.05)

    count = 1
    prob_all_adv = []
    label_all = []

    for (image, labels) in test_loader:
        if labels.item() == 0:
            subdir = 'fake'
        else:
            subdir = 'real'

        output_path = os.path.join(adv_save_dir, subdir)
        os.makedirs(output_path, exist_ok=True)

        image = image.to(device)
        labels = labels.to(device)

        label_all.extend(labels)

        adv_image = att(image, labels)

        file_name = str(count) + '.png'

        save_path = os.path.join(output_path, file_name)
        save_image(adv_image.cpu(), save_path, normalize=True)
        print(save_path)

        adv_outputs = model(adv_image)
        _, adv_preds = torch.max(adv_outputs.data, 1)

        prob_all_adv.extend(adv_preds.cpu().numpy())

        count += 1

    print('*******************************')
    print(roc_auc_score(label_all, prob_all_adv))
    print(f1_score(label_all, prob_all_adv))
    print(accuracy_score(label_all, prob_all_adv))
    print(precision_score(label_all, prob_all_adv))
    print(recall_score(label_all, prob_all_adv))
    print(confusion_matrix(label_all, prob_all_adv))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', type=int, default=1)
    parse.add_argument('--test_path', type=str, default='./data')
    parse.add_argument('--device', type=str, default='cpu')
    parse.add_argument('--model_path', type=str, default='./output/Resnet18/best.pkl')
    parse.add_argument('--adv_save_dir', type=str, default= './fsgm')
    main()
