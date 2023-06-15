import os.path

import torch
import random
import numpy as np
import requests
import tqdm

EPSILON = 1e-8
CHUNK_SIZE = 1 * 1024 * 1024

def download_file(source, destination, size=None):
    if size is None:
        size = 0
    req = requests.get(source, stream=True)
    with open(destination, 'wb') as archive:
        for chunk in tqdm.tqdm(
            req.iter_content(chunk_size=CHUNK_SIZE),
            total=size // CHUNK_SIZE,
            leave=False,
        ):
            if chunk:
                archive.write(chunk)

def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))

def select_index(n1, n2):
    # draw n2 samples from a pool with length of n1
    if n1>=n2:
        ind = random.sample(range(n1), n2)
        # cl0_select = cl0[ind]
    else:
        ind1 = random.sample(range(n1), n1)
        ind2 = random.choices(ind1,k=n2-n1)
        ind = ind1+ind2
    return ind

def load_clusters(dataloader):
    # if dataset == 'mini-imagenet':
    #     root = os.path.join(root, 'cfe_encodings/mini_imagenet_128_K_500_train_clusters.npz')
    # elif root=='omniglot':
    #     root = os.path.join(root, 'cfe_encodings/omniglot_64_K_500_train_clusters.npz')
    # elif root=='tiered-imagenet':
    #     root = os.path.join(root, 'cfe_encodings/tiered_imagenet_128_K_500_train_clusters.npz')
    root = dataloader.dataset.dataset.root
    filename = dataloader.dataset.dataset.filename
    cluster_root = os.path.join(root, filename % 'train_clusters')
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    data = np.load(cluster_root)
    np.load = np_load_old
    c_list = dataloader.dataset.dataset.data
    c_centers = data['cluster_centers']
    c = torch.tensor(c_centers).cuda(non_blocking=True)

    dist = pairwise_distances(c, c, 'l2')
    _, ind_sorted = torch.sort(dist, dim=1)
    return c_list, c_centers, ind_sorted

def momentum_update(model_q, model_k, beta = 0.999, replace=False):
    param_k = model_k.state_dict()
    if replace:
        param_q = model_q.state_dict()
        model_k.load_state_dict(param_q)
    else:
        param_q = model_q.named_parameters()
        for n, q in param_q:
            if n in param_k:
                param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
        model_k.load_state_dict(param_k)