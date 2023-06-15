import os
import pickle
from PIL import Image
import h5py
import json

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_file_from_google_drive

import bisect

class MiniImagenetConcatDataset(Dataset):
    folder = 'miniimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    gz_filename = 'mini-imagenet.tar.gz'
    gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    pkl_filename = 'mini-imagenet-cache-{0}.pkl'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(MiniImagenetConcatDataset, self).__init__(root)

        self.meta_train = meta_train
        self.meta_test = meta_test
        self.meta_val = meta_val
        self._meta_split = meta_split
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('MiniImagenet integrity check failed')
        self._num_classes = len(self.labels)

        dataset = self.data
        unsuper_dataset = [self.data[self.labels[i]] for i in range(self._num_classes)]
        self.cumulative_sizes = self.cumsum(unsuper_dataset)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # class_name = self.labels[index % self.num_classes]
        # data = self.data[class_name]
        # transform = self.get_transform(index, self.transform)
        # target_transform = self.get_target_transform(index)
        #
        # return MiniImagenetDataset(data, class_name, transform=transform,
        #     target_transform=target_transform)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        character_name = self.labels[dataset_idx]
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        img0 = self.data[character_name][sample_idx]
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
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
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
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_file_from_google_drive(self.gdrive_id, self.root,
                                        self.gz_filename, md5=self.gz_md5)

        filename = os.path.join(self.root, self.gz_filename)
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            pkl_filename = os.path.join(self.root, self.pkl_filename.format(split))
            if not os.path.isfile(pkl_filename):
                raise IOError()
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f)
                images, classes = data['image_data'], data['class_dict']

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)

            if os.path.isfile(pkl_filename):
                os.remove(pkl_filename)
