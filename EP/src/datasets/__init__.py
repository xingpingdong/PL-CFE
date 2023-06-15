import torchvision

from .episodic_dataset import FewShotSampler
from .episodic_miniimagenet import EpisodicMiniImagenet, EpisodicMiniImagenetPkl
from .miniimagenet import NonEpisodicMiniImagenet, RotatedNonEpisodicMiniImagenet, RotatedNonEpisodicMiniImagenetPkl
from .episodic_tiered_imagenet import EpisodicTieredImagenet
from .tiered_imagenet import RotatedNonEpisodicTieredImagenet
from .cub import RotatedNonEpisodicCUB, NonEpisodicCUB
from .episodic_cub import EpisodicCUB
from .episodic_tiered_imagenet import EpisodicTieredImagenet
from .tiered_imagenet import RotatedNonEpisodicTieredImagenet, NonEpisodicTieredImagenet
# from .omniglot import NonEpisodicOmniglot, RotatedNonEpisodicOmniglot
# from .episodic_omniglot import EpisodicOmniglot

from .omniglot_en import RotatedNonEpisodicOmniglot as RotatedNonEpisodicOmniglotEn
from .episodic_omniglot_en import EpisodicOmniglot as EpisodicOmniglotEn
from .miniimagenet_en import RotatedNonEpisodicMiniimagenet as RotatedNonEpisodicMiniimagenetEn
from .episodic_miniimagenet_en import EpisodicMiniimagenet as EpisodicMiniimagenetEn





from .tieredimagenet_en import RotatedNonEpisodicTieredimagenet as RotatedNonEpisodicTieredimagenetEn
from .episodic_tieredimagenet_en import EpisodicTieredimagenet as EpisodicTieredimagenetEn

def get_dataset(dataset_name, 
                data_root,
                split, 
                transform, 
                classes,
                support_size,
                query_size, 
                unlabeled_size,
                n_iters):

    transform_func = get_transformer(transform, split)
    if dataset_name == "rotated_omniglot":
        dataset = RotatedNonEpisodicOmniglot(data_root,
                                                 split,
                                                 transform_func)
    elif dataset_name == "omniglot":
        dataset = NonEpisodicOmniglot(data_root,
                                          split,
                                          transform_func)
    elif dataset_name == "episodic_omniglot":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicOmniglot(data_root=data_root,
                                       split=split,
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func)
    elif dataset_name == "rotated_omniglot_en":
        dataset = RotatedNonEpisodicOmniglotEn(data_root,
                                          split,
                                          transform_func)
    elif dataset_name == "episodic_omniglot_en":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicOmniglotEn(data_root=data_root,
                                       split=split,
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func)
    elif dataset_name == "rotated_miniimagenet_en":
        dataset = RotatedNonEpisodicMiniimagenetEn(data_root,
                                          split,
                                          transform_func)
    elif dataset_name == "episodic_miniimagenet_en":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicMiniimagenetEn(data_root=data_root,
                                       split=split,
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func)
    elif dataset_name == "rotated_tiered-imagenet_en":
        dataset = RotatedNonEpisodicTieredimagenetEn(data_root,
                                          split,
                                          transform_func)
    elif dataset_name == "episodic_tiered-imagenet_en":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicTieredimagenetEn(data_root=data_root,
                                       split=split,
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func)

    elif dataset_name == "rotated_miniimagenet":
        dataset = RotatedNonEpisodicMiniImagenet(data_root,
                                                 split,
                                                 transform_func)

    elif dataset_name == "miniimagenet":
        dataset = NonEpisodicMiniImagenet(data_root,
                                          split,
                                          transform_func)
    elif dataset_name == "episodic_miniimagenet":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicMiniImagenet(data_root=data_root,
                                       split=split, 
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func)
    elif dataset_name == "rotated_episodic_miniimagenet_pkl":
        dataset = RotatedNonEpisodicMiniImagenetPkl(data_root=data_root,
                                          split=split,
                                          transforms=transform_func)
    elif dataset_name == "episodic_miniimagenet_pkl":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicMiniImagenetPkl(data_root=data_root,
                                       split=split, 
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func)
    elif dataset_name == "cub":
        dataset = NonEpisodicCUB(data_root, split, transform_func)
    elif dataset_name == "rotated_cub":
        dataset = RotatedNonEpisodicCUB(data_root, split, transform_func)
    elif dataset_name == "episodic_cub":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicCUB(data_root=data_root,
                              split=split, 
                              sampler=few_shot_sampler,
                              size=n_iters,
                              transforms=transform_func)
    elif dataset_name == "tiered-imagenet":
        dataset = NonEpisodicTieredImagenet(data_root, split, transform_func)
    elif dataset_name == "rotated_tiered-imagenet":
        dataset = RotatedNonEpisodicTieredImagenet(data_root, split, transform_func)
    elif dataset_name == "episodic_tiered-imagenet":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicTieredImagenet(data_root,
                                         split=split, 
                                         sampler=few_shot_sampler,
                                         size=n_iters,
                                         transforms=transform_func)
    return dataset


# ===================================================
# helpers
def get_transformer(transform, split):
    if transform == "data_augmentation":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84,84)),
                                                    torchvision.transforms.ToTensor()])

        return transform 

    if "{}_{}".format(transform, split) == "cifar_train":
        transform = torchvision.transforms.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transform

    if "{}_{}".format(transform, split) == "cifar_test" or "{}_{}".format(transform, split) == "cifar_val":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transform

    if transform == "basic":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((84,84)),
                                                    torchvision.transforms.ToTensor()])

        return transform 
    
    if transform == "omni":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((28,28)),
                                                    torchvision.transforms.ToTensor()])

        return transform 

    if transform == "wrn_pretrain_train":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.RandomResizedCrop((80, 80), scale=(0.08, 1)),
                                                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                    torchvision.transforms.ToTensor()
                                                            ])
        return transform
    elif transform == "wrn_finetune_train":
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((92, 92)),
                                                    torchvision.transforms.CenterCrop(80),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor()
                                                    ])
        return transform

    elif "wrn" in transform:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    torchvision.transforms.Resize((92, 92)),
                                                    torchvision.transforms.CenterCrop(80),
                                                    torchvision.transforms.ToTensor()])
        return transform

    raise NotImplementedError
                                                    
