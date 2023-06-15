import torch
from collections import OrderedDict
import copy
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
        ).pow(2).mean(dim=2)
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


def random_select_cluster(simi, knn):
    # simi: N_samples x N_c
    _, sub = torch.sort(simi, descending=True)
    n, m = sub.shape
    ind = torch.zeros(n, dtype=torch.long)
    knn = min(knn, m)
    for i in range(n):
        s = sub[i]
        i0 = torch.randint(0, knn, (1,))
        ind[i] = s[i0]

    return ind

def get_parameters(model):
    t = model.state_dict()
    parameters = copy.deepcopy(t)
    # parameters = OrderedDict()
    # for name, p in model.named_parameters():
    #     parameters[name] = p.data.clone()
    return parameters

def average_params_pool(params_pool):
    params = copy.deepcopy(params_pool[0])
    l = len(params_pool)
    if l>1:
        for i in range(1,l):
            p1 = params_pool[i]
            for k, v in params.items():
                params[k].data = v.data + p1[k].data.clone()
        for k, v in params.items():
            params[k].data = v.data / l
    return params
# pos_simi = torch.bmm(pos_samples, samples.view(s[0],-1,1))
# _, sub = torch.sort(pos_simi.squeeze(), dim=1, descending=True)
# sub = sub[:, :knn]

## random select positive samples
# sub = torch.randint(0, m, (n, knn))
#
# c_t = torch.tensor(range(0, m * n, m)).view(n, 1)
# d = c_t.repeat(1, knn)
# ind = sub + d
# ind = ind.view(-1)
# pos_samples_select = pos_samples.view(-1,n_d)[ind].view(n_pos,knn, -1)
# pos_samples_select = pos_samples_select.mean(dim=1).squeeze()


from math import log


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 样本数
    labelCounts = {}  # 该数据集每个类别的频数
    for featVec in dataSet:  # 对每一行样本
        currentLabel = featVec[-1]  # 该样本的标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算p(xi)
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt

def do_RWR(features, labels):
    """
    calculate the reach probalilty with restart random walk
    :param features n x d: the feature of all node
    :param labels n x 1: the labels for all node, -1 means unlabel
    :return:
    """
    ## construct affinity matric for each node
    # fea_norm = torch.norm(features,p=2,dim=1)
    # features = torch.div(features.T, fea_norm)
    # # features = features.T
    # simi = torch.mm(features.T, features) # cos similarity
    # W = (simi+1)/2# normlize to [0,1]
    dis = pairwise_distances(features, features,'l2')
    sigma = 60
    EPSILON = 1e-5
    W = torch.exp(-dis*sigma)+EPSILON
    # print(W)
    n = W.shape[0]
    # Random Walks with Restart (RWR)
    c = 0.001
    E = torch.diag(torch.ones(n))
    P = torch.div(W, W.sum(dim=1))
    P = P.T
    # get init seed
    u_label = torch.unique(labels)
    u_label = u_label[u_label>=0]
    K = len(u_label)
    lines = torch.zeros(n,K)
    for i,k in enumerate(u_label):
        idx = labels==k
        Mk = float(torch.sum(idx))
        lines[idx,i] = 1.0/Mk

    P_inv = c*(E-(1-c)*P).inverse()
    R = torch.mm(P_inv, lines)
    # Estimat likelihoods
    likelihoods = torch.div(R,R.sum(dim=0)) # n x K
    # Estimate posteriors
    prob = torch.div(likelihoods.T, likelihoods.T.sum(dim=0)) # K x n
    prob = prob.T
    # print(prob)
    return prob


if __name__ == '__main__':
    n=10
    features = torch.randn(n,4)
    # labels = torch.tensor([0,1,2,-1,-1,-1,-1,-1,-1,-1])
    labels = -1*torch.ones(n)
    labels[0] = 0
    labels[1] = 1
    labels[2] = 2
    do_RWR(features,labels)