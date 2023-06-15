import numpy as np
from PIL import Image
import h5py
import json
import os
import io
import pickle

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_file_from_google_drive
from .utils import download_file
import bisect

class TieredImagenetConcatDataset(Dataset):
    folder = 'tieredimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07'
    tar_filename = 'tiered-imagenet.tar'
    tar_md5 = 'e07e811b9f29362d159a9edd0d838c62'
    tar_folder = 'tiered-imagenet'

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
        super(TieredImagenetConcatDataset, self).__init__(root)

        self.meta_train = meta_train
        self.meta_test = meta_test
        self.meta_val = meta_val
        self._meta_split = meta_split
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self._data_file = None
        self._data = None
        self._labels_specific = None

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('TieredImagenet integrity check failed')
        self._num_classes = len(self.labels_specific)

        unsuper_dataset = [self.data[self.labels_specific[i]] for i in range(self._num_classes)]
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
        character_name = self.labels_specific[dataset_idx]
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        img0 = self.data[character_name][sample_idx]
        image = Image.open(io.BytesIO(img0))
        # target = character_name
        target = dataset_idx

        if self.transform is not None:
            image = self.transform(image)

        return (image, target, img0)

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
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels_specific(self):
        if self._labels_specific is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels_specific = json.load(f)
        return self._labels_specific


    @property
    def num_classes(self):
        return self._num_classes

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def download(self):
        import tarfile
        import shutil
        from tqdm import tqdm

        if self._check_integrity():
            return

        if os.path.isfile(os.path.join(self.root, self.tar_filename)):
            filename = os.path.join(self.root, self.tar_filename)
            with tarfile.open(filename, 'r') as f:
                f.extractall(self.root)
        else:
            print('Downloading tiered ImageNet. (12Gb) Please be patient.')
            try:
                archive_dir = os.path.join(self.root, self.tar_folder)
                # archive_dir = self.root
                os.makedirs(archive_dir, exist_ok=True)
                files_to_download = [
                    'https://zenodo.org/record/7978538/files/tiered-imagenet-class_names.txt',
                    'https://zenodo.org/record/7978538/files/tiered-imagenet-synsets.txt',
                    'https://zenodo.org/record/7978538/files/tiered-imagenet-test_images_png.pkl',
                    'https://zenodo.org/record/7978538/files/tiered-imagenet-test_labels.pkl',
                    'https://zenodo.org/record/7978538/files/tiered-imagenet-train_images_png.pkl',
                    'https://zenodo.org/record/7978538/files/tiered-imagenet-train_labels.pkl',
                    'https://zenodo.org/record/7978538/files/tiered-imagenet-val_images_png.pkl',
                    'https://zenodo.org/record/7978538/files/tiered-imagenet-val_labels.pkl',
                ]
                for file_url in files_to_download:
                    file_dest = os.path.join(
                        archive_dir,
                        os.path.basename(file_url).replace('tiered-imagenet-', '')
                    )
                    download_file(
                        source=file_url,
                        destination=file_dest,
                    )
            except Exception:
                download_file_from_google_drive(self.gdrive_id, self.root,
                                            self.tar_filename, md5=self.tar_md5)
                filename = os.path.join(self.root, self.tar_filename)
                with tarfile.open(filename, 'r') as f:
                    f.extractall(self.root)


        tar_folder = os.path.join(self.root, self.tar_folder)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            images_filename = os.path.join(tar_folder, '{0}_images_png.pkl'.format(split))
            if not os.path.isfile(images_filename):
                raise IOError(images_filename)
            with open(images_filename, 'rb') as f:
                images = pickle.load(f, encoding='bytes')

            labels_filename = os.path.join(tar_folder, '{0}_labels.pkl'.format(split))
            if not os.path.isfile(labels_filename):
                raise IOError()
            with open(labels_filename, 'rb') as f:
                labels = pickle.load(f, encoding='latin1')

            labels_str = labels['label_specific_str']
            general_labels_str = labels['label_general_str']
            general_labels = labels['label_general']
            with open(os.path.join(self.root, self.filename_labels.format(split)), 'w') as f:
                json.dump(labels_str, f)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.uint8)
                for i, label in enumerate(tqdm(labels_str, desc=filename)):
                    indices, = np.where(labels['label_specific'] == i)
                    dataset = group.create_dataset(label, (len(indices),), dtype=dtype)
                    general_idx = general_labels[indices[0]]
                    dataset.attrs['label_general'] = (general_labels_str[general_idx]
                                                      if general_idx < len(general_labels_str) else '')
                    dataset.attrs['label_specific'] = label
                    for j, k in enumerate(indices):
                        dataset[j] = np.squeeze(images[k])

        if os.path.isdir(tar_folder):
            shutil.rmtree(tar_folder)


class TieredImagenetDataset(Dataset):
    def __init__(self, index, data, general_class_name, specific_class_name,
                 transform=None, target_transform=None):
        super(TieredImagenetDataset, self).__init__(index, transform=transform,
                                                    target_transform=target_transform)
        self.data = data
        self.general_class_name = general_class_name
        self.specific_class_name = specific_class_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.data[index]))
        target = (self.general_class_name, self.specific_class_name)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
