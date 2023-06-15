import torch
import torch.nn.functional as F
import os
import json

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets import Omniglot, MiniImagenet, TieredImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical
from torchvision.transforms import ToTensor, Resize, Compose

from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from maml.metalearners import ModelAgnosticMetaLearning
from maml.utils import ToTensor1D

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.folder is not None:
        config['folder'] = args.folder
    if args.num_steps > 0:
        config['num_steps'] = args.num_steps
    if args.num_batches > 0:
        config['num_batches'] = args.num_batches
    if args.num_shots > 0:
        config['num_shots'] = args.num_shots
    if args.model_path is not '0':
        config['model_path']=args.model_path
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')
    print(device)

    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=config['num_shots'],
                                      num_test_per_class=config['num_shots_test'])
    if config['dataset'] == 'sinusoid':
        transform = ToTensor1D()
        meta_test_dataset = Sinusoid(config['num_shots'] + config['num_shots_test'],
            num_tasks=1000000, transform=transform, target_transform=transform,
            dataset_transform=dataset_transform)
        model = ModelMLPSinusoid(hidden_sizes=[40, 40])
        loss_function = F.mse_loss

    elif config['dataset'] == 'omniglot':
        transform = Compose([Resize(28), ToTensor()])
        meta_test_dataset = Omniglot(config['folder'], transform=transform,
            target_transform=Categorical(config['num_ways']),
            num_classes_per_task=config['num_ways'], meta_test=True,
            dataset_transform=dataset_transform, download=True)
        model = ModelConvOmniglot(config['num_ways'],
                                  hidden_size=config['hidden_size'])
        loss_function = F.cross_entropy

    elif config['dataset'] == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])
        meta_test_dataset = MiniImagenet(config['folder'], transform=transform,
            target_transform=Categorical(config['num_ways']),
            num_classes_per_task=config['num_ways'], meta_test=True,
            dataset_transform=dataset_transform, download=True)
        model = ModelConvMiniImagenet(config['num_ways'],
                                      hidden_size=config['hidden_size'])
        loss_function = F.cross_entropy
    elif config['dataset'] == 'tieredimagenet':
        transform = Compose([Resize(84), ToTensor()])
        meta_test_dataset = TieredImagenet(config['folder'], transform=transform,
            target_transform=Categorical(config['num_ways']),
            num_classes_per_task=config['num_ways'], meta_test=True,
            dataset_transform=dataset_transform, download=True)
        model = ModelConvMiniImagenet(config['num_ways'],
                                      hidden_size=config['hidden_size'])
        loss_function = F.cross_entropy
    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(config['dataset']))

    with open(config['model_path'], 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))

    meta_test_dataloader = BatchMetaDataLoader(meta_test_dataset,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    metalearner = ModelAgnosticMetaLearning(model,
        first_order=config['first_order'], num_adaptation_steps=config['num_steps'],
        step_size=config['step_size'], loss_function=loss_function, device=device)

    results = metalearner.evaluate(meta_test_dataloader,
                                   max_batches=config['num_batches'],
                                   verbose=args.verbose,
                                   desc='Test')
    print('results_{}_ways_{}_shots_model_{}: {}'.format(config['num_ways'], config['num_shots'],config['model_path'].split('/')[-1],results))
    # Save results
    dirname = os.path.dirname(config['model_path'])
    with open(os.path.join(dirname, 'results_{}_ways_{}_shots_model_{}.json'.format(config['num_ways'], config['num_shots'], config['model_path'].split('/')[-1])), 'w') as f:
        json.dump(results, f)

import random
import numpy as np
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')
    parser.add_argument('config', type=str,
        help='Path to the configuration file returned by `train.py`.')
    parser.add_argument('--folder', type=int, default=None,
        help='Path to the folder the data is downloaded to. '
        '(default: path defined in configuration file).')

    # Optimization
    parser.add_argument('--num-steps', type=int, default=-1,
        help='Number of fast adaptation steps, ie. gradient descent updates '
        '(default: number of steps in configuration file).')
    parser.add_argument('--num-batches', type=int, default=-1,
        help='Number of batch of tasks per epoch '
        '(default: number of batches in configuration file).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=8,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed')

    parser.add_argument('--num-shots', type=int, default=-1,
                        help='Number of fast adaptation steps, ie. gradient descent updates '
                             '(default: number of steps in configuration file).')

    parser.add_argument('--model-path', type=str, default='0',
                        help='model path.')

    args = parser.parse_args()

    seed_torch(args.seed)
    main(args)
