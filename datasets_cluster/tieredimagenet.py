import numpy as np
from PIL import Image
import h5py
import json
import os
import io
import pickle

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
# QKFIX: See torchmeta.datasets.utils for more informations
from torchmeta.datasets.utils import download_file_from_google_drive


class TieredImagenet(CombinationMetaDataset):
    """
    The Tiered-Imagenet dataset, introduced in [1]. This dataset contains images
    of 608 different classes from the ILSVRC-12 dataset (Imagenet challenge).

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `tieredimagenet` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way"
        classification.

    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        `meta_val` and `meta_test` if all three are set to `False`.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root
        directory (under the `tieredimagenet` folder). If the dataset is already
        available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from [this repository]
    (https://github.com/renmengye/few-shot-ssl-public/). The dataset contains
    images from 34 categories. The meta train/validation/test splits are over
    20/6/8 categories. Each category contains between 10 and 30 classes. The
    splits over categories (instead of over classes) ensures that all the training
    classes are sufficiently distinct from the test classes (unlike Mini-Imagenet).

    References
    ----------
    .. [1] Ren, M., Triantafillou, E., Ravi, S., Snell, J., Swersky, K.,
           Tenenbaum, J.B., Larochelle, H. and Zemel, R.S. (2018). Meta-learning
           for semi-supervised few-shot classification. International Conference
           on Learning Representations. (https://arxiv.org/abs/1803.00676)
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = TieredImagenetClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(TieredImagenet, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class TieredImagenetClassDataset(ClassDataset):
    folder = 'cfe_encodings' #'tieredimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07'
    tar_filename = 'tiered-imagenet.tar'
    tar_md5 = 'e07e811b9f29362d159a9edd0d838c62'
    tar_folder = 'tiered-imagenet'

    # filename = '{0}_data.hdf5'
    # filename_labels = '{0}_labels.json'
    filename = 'tieredimagenet_128_K_500_%s.npz'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(TieredImagenetClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename)
        # self.split_filename_labels = os.path.join(self.root,
        #     self.filename_labels.format('vinyals_' if use_vinyals_split else '',
        #     self.meta_split))
        data = np.load(self.split_filename % self.meta_split)
        X = data["X"]
        Y = data["cluster_label"]
        data_dict = {}
        data_list = []
        label_names = []
        Y_u = np.unique(Y)
        for y in Y_u:
            # data_dict[str(y)]=X[Y==y]
            # label_names.append(str(y))
            data_list.append(X[Y == y])
            label_names.append(y)

        self._data = data_list
        self._labels = label_names

        if not self._check_integrity():
            raise RuntimeError('TieredImagenet integrity check failed')
        self._num_classes = len(self.labels)

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    def __getitem__(self, index):
        specific_class_name = self.labels[index % self.num_classes]
        data = self.data[specific_class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return TieredImagenetDataset(index, data,
             specific_class_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes


    def _check_integrity(self):
        return os.path.isfile(self.split_filename % self.meta_split)


class TieredImagenetDataset(Dataset):
    def __init__(self, index, data, specific_class_name,
                 transform=None, target_transform=None):
        super(TieredImagenetDataset, self).__init__(index, transform=transform,
                                                    target_transform=target_transform)
        self.data = data
        self.specific_class_name = specific_class_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        target0 = self.specific_class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target0)

        return (image, target, target0)
