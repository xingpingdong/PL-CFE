import os
import pickle
from PIL import Image
import h5py
import json

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
# QKFIX: See torchmeta.datasets.utils for more informations
from torchmeta.datasets.utils import download_file_from_google_drive
import numpy as np

class MiniImagenet(CombinationMetaDataset):
    """
    The Mini-Imagenet dataset, introduced in [1]. This dataset contains images
    of 100 different classes from the ILSVRC-12 dataset (Imagenet challenge).
    The meta train/validation/test splits are taken from [2] for reproducibility.

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `miniimagenet` exists.

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
        directory (under the `miniimagenet` folder). If the dataset is already
        available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from [this repository]
    (https://github.com/renmengye/few-shot-ssl-public/). The meta train/
    validation/test splits are over 64/16/20 classes.

    References
    ----------
    .. [1] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016).
           Matching Networks for One Shot Learning. In Advances in Neural
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)

    .. [2] Ravi, S. and Larochelle, H. (2016). Optimization as a Model for
           Few-Shot Learning. (https://openreview.net/forum?id=rJY0-Kcll)
    """

    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = MiniImagenetClassDataset(root, meta_train=meta_train,
                                           meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
                                           transform=transform, class_augmentations=class_augmentations,
                                           download=download)
        super(MiniImagenet, self).__init__(dataset, num_classes_per_task,
                                           target_transform=target_transform, dataset_transform=dataset_transform)


class MiniImagenetClassDataset(ClassDataset):
    folder = 'cfe_encodings' #'miniimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    gz_filename = 'mini-imagenet.tar.gz'
    gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    pkl_filename = 'mini-imagenet-cache-{0}.pkl'

    # filename = '{0}_data.hdf5'
    filename = 'miniimagenet_128_K_500_%s.npz'
    # filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(MiniImagenetClassDataset, self).__init__(meta_train=meta_train,
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

        # if download:
        #     self.download()

        if not self._check_integrity():
            raise RuntimeError('MiniImagenet integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return MiniImagenetDataset(index, data, class_name,
                                   transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    def _check_integrity(self):
        return os.path.isfile(self.split_filename % self.meta_split)


class MiniImagenetDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(MiniImagenetDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = self.data[index]
        target0 = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target0)

        return (image, target, target0)
