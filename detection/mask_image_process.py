import os

os.environ['TORCH_HOME'] = '../pretrain_model'

import argparse
from torchvision.utils import save_image
import torchvision
from pytorch_grad_cam import GradCAM

from network.classifier import *
from network.transform import *


def generate_mask_image(input_tensor, cam):
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = torch.from_numpy(grayscale_cam).unsqueeze(1)
    cam_mask = grayscale_cam.expand(input_tensor.size())

    cam_mask_image = torch.where(cam_mask >= 0.7, torch.ones(cam_mask.size()), input_tensor)

    return cam_mask_image


def main():
    args = parse.parse_args()
    test_path = args.test_path
    batch_size = args.batch_size
    model_path = args.model_path
    device = args.device
    cam_save_dir = args.cam_save_dir
    torch.backends.cudnn.benchmark = True

    test_dataset = torchvision.datasets.ImageFolder(test_path, transform=resnet_data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)

    model = Resnet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    layers = []

    for layer in model.named_modules():
        layers.append(layer)

    target_layers = [layers[66][1]]  # resnet18
    resnet18_cam = GradCAM(model=model, target_layers=target_layers)

    count = 1

    for (image, labels) in test_loader:
        if labels.item() == 0:
            subdir = 'fake'
        else:
            subdir = 'real'

        output_path = os.path.join(cam_save_dir, subdir)
        os.makedirs(output_path, exist_ok=True)

        cam_mask_image = generate_mask_image(image, resnet18_cam)

        cam_image_name = os.path.join(output_path, str(count)) + '.png'
        save_image(cam_mask_image, cam_image_name, normalize=True, value_range=(-1, 1))

        count += 1


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', type=int, default=2)
    parse.add_argument('--test_path', type=str, default='./data')
    parse.add_argument('--device', type=str, default='cpu')
    parse.add_argument('--model_path', type=str, default='./output/Resnet18/best.pkl')
    parse.add_argument('--cam_save_dir', type=str, default='./cam')
    main()
