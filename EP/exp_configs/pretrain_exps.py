from haven import haven_utils as hu

conv4 = {
    "name": "pretraining",
    'backbone':'conv4',
    "depth": 4,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}
conv4_1 = {
    "name": "pretraining",
    'backbone':'conv4_1',
    "depth": 4,
    "width": 1,
    "transform_train": "omni",
    "transform_val": "omni",
    "transform_test": "omni"
}

wrn = {
    "name": "pretraining",
    "backbone": 'wrn',
    "depth": 28,
    "width": 10,
    "transform_train": "wrn_pretrain_train",
    "transform_val": "wrn_val",
    "transform_test": "wrn_val"
}

resnet12 = {
    "name": "pretraining",
    "backbone": 'resnet12',
    "depth": 12,
    "width": 1,
    "transform_train": "basic",
    "transform_val": "basic",
    "transform_test": "basic"
}

omniglot = {
    "dataset": "omniglot",
    "dataset_train": "rotated_omniglot",
    "dataset_val": "episodic_omniglot",
    "dataset_test": "episodic_omniglot",
    "n_classes": 1100,
    "data_root": "omniglot-c"
}


omniglot_en_acai = {
    "dataset": "omniglot_en_acai",
    "dataset_train": "rotated_omniglot_en_acai",
    "dataset_val": "episodic_omniglot",
    "dataset_test": "episodic_omniglot",
    "n_classes": 500,
    "data_root": "omniglot-c"
}

miniimagenet = {
    "dataset": "miniimagenet",
    "dataset_train": "rotated_episodic_miniimagenet_pkl",
    "dataset_val": "episodic_miniimagenet_pkl",
    "dataset_test": "episodic_miniimagenet_pkl",
    "n_classes": 64,
    "data_root": "mini-imagenet"
}


omniglot_en = {
    "dataset": "omniglot_en",
    "dataset_train": "rotated_omniglot_en",
    "dataset_val": "episodic_omniglot_en",
    "dataset_test": "episodic_omniglot_en",
    "n_classes": 500,
    "data_root": "cfe_encodings"
}

miniimagenet_en = {
    "dataset": "miniimagenet_en",
    "dataset_train": "rotated_miniimagenet_en",
    "dataset_val": "episodic_miniimagenet_en",
    "dataset_test": "episodic_miniimagenet_en",
    "n_classes": 500,
    "data_root": "cfe_encodings"
}


tiered_imagenet_en = {
    "dataset": "tiered-imagenet_en",
    "n_classes": 500,
    "dataset_train": "rotated_tiered-imagenet_en",
    "dataset_val": "episodic_tiered-imagenet_en",
    "dataset_test": "episodic_tiered-imagenet_en",
    "data_root": "cfe_encodings",
}

tiered_imagenet = {
    "dataset": "tiered-imagenet",
    "n_classes": 351,
    "dataset_train": "rotated_tiered-imagenet",
    "dataset_val": "episodic_tiered-imagenet",
    "dataset_test": "episodic_tiered-imagenet",
    "data_root": "tiered-imagenet",
}

cub = {
    "dataset": "cub",
    "n_classes": 100,
    "dataset_train": "rotated_cub",
    "dataset_val": "episodic_cub",
    "dataset_test": "episodic_cub",
    "data_root": "CUB_200_2011"
}

EXP_GROUPS = {"pretrain": []}

classes=5
for dataset in [omniglot, miniimagenet]:
    for backbone in [conv4]:#[conv4_1, conv4, resnet12, wrn]:
        for lr in [0.1]:#[0.2, 0.1]:
            EXP_GROUPS['pretrain'] += [{"model": backbone,

                                        # Hardware
                                        "ngpu": 4,
                                        "random_seed": 42,

                                        # Optimization
                                        "batch_size": 128,
                                        "target_loss": "val_accuracy",
                                        "lr": lr,
                                        "min_lr_decay": 0.0001,
                                        "weight_decay": 0.0005,
                                        "patience": 10,
                                        "max_epoch": 200,
                                        "train_iters": 600,
                                        "val_iters": 600,
                                        "test_iters": 600,
                                        "tasks_per_batch": 1,

                                        # Model
                                        "dropout": 0.1,
                                        "avgpool": True,

                                        # Data
                                        'n_classes': dataset["n_classes"],
                                        "collate_fn": "default",
                                        "transform_train": backbone["transform_train"],
                                        "transform_val": backbone["transform_val"],
                                        "transform_test": backbone["transform_test"],

                                        "dataset_train": dataset["dataset_train"],
                                        "dataset_train_root": dataset["data_root"],
                                        "classes_train": classes,
                                        "support_size_train": 5,
                                        "query_size_train": 15,
                                        "unlabeled_size_train": 0,

                                        "dataset_val": dataset["dataset_val"],
                                        "dataset_val_root": dataset["data_root"],
                                        "classes_val": classes,
                                        "support_size_val": 5,
                                        "query_size_val": 15,
                                        "unlabeled_size_val": 0,

                                        "dataset_test": dataset["dataset_test"],
                                        "dataset_test_root": dataset["data_root"],
                                        "classes_test": classes,
                                        "support_size_test": 5,
                                        "query_size_test": 15,
                                        "unlabeled_size_test": 0,
                                        

                                        # Hparams
                                        "embedding_prop": True,
                                        "cross_entropy_weight": 1,
                                        "few_shot_weight": 0,
                                        "rotation_weight": 1,
                                        "active_size": 0,
                                        "distance_type": "labelprop",
                                        "kernel_bound": "",
                                        "rotation_labels": [0, 1, 2, 3]
                                        }]

EXP_GROUPS["pretrain_en_omni"]=[]
base_setting = EXP_GROUPS['pretrain'][0]
for dataset in [omniglot_en]:#, miniimagenet_en]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4_1]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            # for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        # "lr": 0.2,
                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        'dataset_train_root': dataset["data_root"],
                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],
                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['pretrain_en_omni'] += [new_setting]

EXP_GROUPS["pretrain_en_imagenet"]=[]
base_setting = EXP_GROUPS['pretrain'][0]
for dataset in [miniimagenet_en, tiered_imagenet_en]:#, miniimagenet_en]:#[, semi_miniimagenet, miniimagenet, tiered_imagenet, cub]:
    for backbone in [conv4]:#[ conv4semi_test,resnet12_semi_norm, wrn_semi_norm]:#[conv4semi_test,conv4semi_test_basic,conv4semi_test]:#[[conv4, resnet12, wrn]:
            # for npc in [20, 100, 200]:#[20, 60, 120, 300, 540]:
                    new_setting = base_setting.copy()
                    new_dict={
                        "model": backbone,
                        "ngpu": 1,
                        # Data
                        'n_classes': dataset["n_classes"],
                        "transform_train": backbone["transform_train"],
                        "transform_val": backbone["transform_val"],
                        "transform_test": backbone["transform_test"],
                        "dataset_train": dataset["dataset_train"],
                        'dataset_train_root': dataset["data_root"],
                        "dataset_val": dataset["dataset_val"],
                        'dataset_val_root': dataset["data_root"],
                        "dataset_test": dataset["dataset_test"],
                        'dataset_test_root': dataset["data_root"],
                    }
                    new_setting.update(new_dict)
                    EXP_GROUPS['pretrain_en_imagenet'] += [new_setting]