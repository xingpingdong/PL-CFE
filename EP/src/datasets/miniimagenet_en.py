from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle as pkl

class NonEpisodicMiniimagenet(Dataset):
    tasks_type = "clss"
    name = "miniimagenet"
    split_paths = {"train": "train", "val":"val", "valid": "val", "test": "test"}
    episodic=False
    c = 1
    h = 28
    w = 28

    def __init__(self, data_root, split, transforms, **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = os.path.join(data_root, "miniimagenet_128_K_500_%s.npz")
        data = np.load(self.data_root % self.split_paths[split])
        self.features = data["X"]
        self.labels = data["cluster_label"]
        # self.cluster_centers = data['cluster_centers']
        self.transforms = transforms

    def next_run(self):
        pass

    def __getitem__(self, item):
        image = self.transforms(np.transpose(self.features[item], (1, 2, 0)))
        image = image * 2 - 1
        return image, self.labels[item]

    def __len__(self):
        return len(self.features)

class RotatedNonEpisodicMiniimagenet(Dataset):
    tasks_type = "clss"
    name = "miniimagenet"
    split_paths = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    episodic = False
    c = 1
    h = 28
    w = 28

    def __init__(self, data_root, split, transforms, rotation_labels=[0, 1, 2, 3], **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = os.path.join(data_root, "miniimagenet_128_K_500_%s.npz")
        data = np.load(self.data_root % self.split_paths[split])
        self.features = data["X"]
        self.labels = data["cluster_label"]
        # self.cluster_centers = data['cluster_centers']
        self.transforms = transforms
        self.size = len(self.features)
        self.rotation_labels = rotation_labels

    def next_run(self):
        pass

    def rotate_img(self, img, rot):
        if rot == 0:  # 0 degrees rotation
            return img
        elif rot == 90:  # 90 degrees rotation
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:  # 90 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270:  # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, item):
        image = np.transpose(self.features[item], (1, 2, 0))
        image = np.uint8(image)
        if np.random.randint(2):
            image = np.fliplr(image).copy()
        cat = [self.transforms(image)]
        if len(self.rotation_labels) > 1:
            image_90 = self.transforms(self.rotate_img(image, 90))
            image_180 = self.transforms(self.rotate_img(image, 180))
            image_270 = self.transforms(self.rotate_img(image, 270))
            cat.extend([image_90, image_180, image_270])
        images = torch.stack(cat) * 2 - 1
        return images, torch.ones(len(self.rotation_labels), dtype=torch.long)*int(self.labels[item]), torch.LongTensor(self.rotation_labels)

    def __len__(self):
        return self.size

import torchvision
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # from src.tools.plot_episode import plot_episode
    import time
    # sampler = FewShotSampler(5, 5, 15, 0)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                 torchvision.transforms.ToTensor(),
                                                ])
    dataset = NonEpisodicMiniimagenet('../../data/mini-imagenet', 'train', transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    for batch in loader:
        print(batch[0][1])
        # print(np.unique(batch[0]["targets"].view(20, 5).numpy()))
        # plot_episode(batch[0], classes_first=False)
        # time.sleep(1)