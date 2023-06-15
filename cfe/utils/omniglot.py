import os
import json
import glob
import h5py
from PIL import Image, ImageOps

from torchmeta.utils.data import Dataset, ClassDataset
# from unsuper.dataset import CombinationMetaDataset
from torchvision.datasets.utils import list_dir, download_url
from torchmeta.datasets.utils import get_asset

import bisect

class OmniglotConcatDataset(Dataset):
    folder = 'omniglot'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    filename = 'data.hdf5'
    filename_labels = '{0}{1}_labels.json'

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, use_vinyals_split=True, transform=None, target_transform=None,
                 class_augmentations=None, download=False):
        super(OmniglotConcatDataset, self).__init__(root, target_transform=target_transform)

        self.transform = transform
        self.meta_train = meta_train
        self.meta_test = meta_test
        self.meta_val = meta_val
        self._meta_split = meta_split
        if self.meta_val and (not use_vinyals_split):
            raise ValueError('Trying to use the meta-validation without the '
                'Vinyals split. You must set `use_vinyals_split=True` to use '
                'the meta-validation split.')

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.use_vinyals_split = use_vinyals_split
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename)
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format('vinyals_' if use_vinyals_split else '',
            self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Omniglot integrity check failed')
        self._num_classes = len(self.labels)

        with h5py.File(self.split_filename, 'r') as dataset:
            unsuper_dataset = [dataset['/'.join(self.labels[i])] for i in range(self._num_classes)]
            self.cumulative_sizes = self.cumsum(unsuper_dataset)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        character_name = '/'.join(self.labels[dataset_idx])
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        with h5py.File(self.split_filename, 'r') as dataset:
            img0 = dataset[character_name][sample_idx]
        image = Image.fromarray(img0)
        # target = character_name
        target = dataset_idx

        if self.transform is not None:
            image = self.transform(image)

        return (image, target)


    @property
    def meta_split(self):
        if self._meta_split is None:
            if self.meta_train:
                self._meta_split = 'train'
            elif self.meta_val:
                self._meta_split = 'val'
            elif self.meta_test:
                self._meta_split = 'test'
            else:
                raise NotImplementedError()
        return self._meta_split
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.split_filename, 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):
        import zipfile
        import shutil

        if self._check_integrity():
            return

        for name in self.zips_md5:
            zip_filename = '{0}.zip'.format(name)
            filename = os.path.join(self.root, zip_filename)
            if os.path.isfile(filename):
                continue

            url = '{0}/{1}'.format(self.download_url_prefix, zip_filename)
            download_url(url, self.root, zip_filename, self.zips_md5[name])

            with zipfile.ZipFile(filename, 'r') as f:
                f.extractall(self.root)

        filename = os.path.join(self.root, self.filename)
        with h5py.File(filename, 'w') as f:
            for name in self.zips_md5:
                group = f.create_group(name)

                alphabets = list_dir(os.path.join(self.root, name))
                characters = [(name, alphabet, character) for alphabet in alphabets
                    for character in list_dir(os.path.join(self.root, name, alphabet))]

                split = 'train' if name == 'images_background' else 'test'
                labels_filename = os.path.join(self.root,
                    self.filename_labels.format('', split))
                with open(labels_filename, 'w') as f_labels:
                    labels = sorted(characters)
                    json.dump(labels, f_labels)

                for _, alphabet, character in characters:
                    filenames = glob.glob(os.path.join(self.root, name,
                        alphabet, character, '*.png'))
                    dataset = group.create_dataset('{0}/{1}'.format(alphabet,
                        character), (len(filenames), 105, 105), dtype='uint8')

                    for i, char_filename in enumerate(filenames):
                        image = Image.open(char_filename, mode='r').convert('L')
                        dataset[i] = ImageOps.invert(image)

                shutil.rmtree(os.path.join(self.root, name))

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename_labels.format(
                'vinyals_', split))
            data = get_asset(self.folder, '{0}.json'.format(split), dtype='json')

            with open(filename, 'w') as f:
                labels = sorted([('images_{0}'.format(name), alphabet, character)
                    for (name, alphabets) in data.items()
                    for (alphabet, characters) in alphabets.items()
                    for character in characters])
                json.dump(labels, f)

import numpy as np
class OmniglotCacheDataset(Dataset):


    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 transform=None):
        super(OmniglotCacheDataset, self).__init__(root)

        dataset = 'omniglot'
        data_folder = os.path.join(root, dataset)

        splits = ['train', 'val', 'test']
        filenames = {split: os.path.join(data_folder, '{}_cache_{}.npz'.format(dataset, split))
                     for split in splits}

        def get_XY(filename):
            data = np.load(filename)
            X, Y = data['X'], data['Y']
            return X, Y


        if meta_train:
            self.X, self.Y = get_XY(filenames['train'])
        elif meta_val:
            self.X, self.Y = get_XY(filenames['val'])
        elif meta_test:
            self.X, self.Y = get_XY(filenames['test'])
        else:
            raise NotImplementedError
        self.transform = transform


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img0 = self.X[idx]
        img = np.squeeze(img0)
        image = Image.fromarray(img)
        target = self.Y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return (image, target)



from torchvision.transforms import ToTensor, Resize, Compose
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
def main(args):

    if args.dataset == 'omniglot':
        transform = Compose([Resize(28), ToTensor()])
        class_augmentations = [Rotation([90, 180, 270])]

        meta_train_dataset = OmniglotConcatDataset(args.folder, meta_train=True, transform=transform,
             class_augmentations=class_augmentations,
            download=True)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['sinusoid', 'omniglot', 'miniimagenet'], default='omniglot',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default='./tmp',
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=1,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_false')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)