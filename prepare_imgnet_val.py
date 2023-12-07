import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser('Get ImageNet validation set for FID/IS evaluation', add_help=False)
parser.add_argument('--data_path', default='./data/imagenet', type=str,
                    help='imagenet dataset path')
parser.add_argument('--output_dir', default='imagenet-val', type=str,
                    help='output directory')

args = parser.parse_args()

transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256)])
dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

nsamples = len(dataset_val)
indices = range(nsamples)
for i in tqdm(indices):
    sample = dataset_val[i]
    img = sample[0]
    sample_name = os.path.join(args.output_dir, '{}.png'.format(str(i).zfill(5)))
    img.save(sample_name)

