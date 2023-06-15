import argparse
from utils.tieredimagenet import TieredImagenetConcatDataset

parser = argparse.ArgumentParser(description='Extract embedding for few shot dataset')
parser.add_argument('-d', '--dataset', default='tieredimagenet', type=str, metavar='S',
                    help='name of dataset ( tieredimagenet)')

def download_datasets(dataset):

    transform = None
    print('Start to download {}'.format(dataset))
    if dataset == 'tieredimagenet':
        meta_train_dataset = TieredImagenetConcatDataset('../data', meta_train=True, transform=transform,
                                                         download=True)

def main():
    args = parser.parse_args()
    download_datasets(args.dataset)

if __name__ == '__main__':
    main()