import argparse
import os

import torchvision
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, \
    confusion_matrix
from torch.utils.data import DataLoader

from network.classifier import *
from network.transform import *

os.environ['TORCH_HOME'] = '../pretrain_model'


def main():
    args = parse.parse_args()
    test_path = args.test_path
    batch_size = args.batch_size
    model_path = args.model_path
    device = args.device
    torch.backends.cudnn.benchmark = True

    test_dataset = torchvision.datasets.ImageFolder(test_path, transform=vgg_data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)

    model = Resnet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    prob_all = []
    label_all = []

    with torch.no_grad():
        for (image, labels) in test_loader:
            image = image.to(device)
            labels = labels.to(device)

            label_all.extend(labels)

            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            # _, preds = torch.max(outputs, 1)   # xception only
            prob_all.extend(preds.cpu().numpy())

        print('*******************************')
        print("AUC:", roc_auc_score(label_all, prob_all))
        print("F1 score:", f1_score(label_all, prob_all))
        print("Accuracy:", accuracy_score(label_all, prob_all))
        print("Precision:", precision_score(label_all, prob_all))
        print("Recall:", recall_score(label_all, prob_all))
        print("Confusion Matrix:\n", confusion_matrix(label_all, prob_all))

        # fpr, tpr, thresholds = roc_curve(label_all, prob_all)
        # print(fpr)
        # print(tpr)
        # plt.plot(fpr, tpr, color='blue', label='FGSM')
        # # plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('FPR')
        # plt.ylabel('TPR')
        # plt.title('Title')
        # plt.legend(loc="lower right")
        # plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--test_path', type=str, default='./data')
    parse.add_argument('--device', type=str, default='cpu')
    parse.add_argument('--model_path', type=str, default='./output/Resnet18/best.pkl')
    main()
