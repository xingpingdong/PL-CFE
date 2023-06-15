# PL-CFE
Implementation code for

[Rethinking Clustering-Based Pseudo-Labeling for Unsupervised Meta-Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800160.pdf).
In Proceedings of the European Conference on Computer Vision (ECCV), 2022.

By Xingping Dong, Jianbing Shen, and Ling Shao.

Any comments, please email: xingping.dong@gmail.com, shenjianbingcg@gmail.com

If you use this software for academic research, please consider to cite the following paper:
```
@inproceedings{dong2022rethinking,
  title={Rethinking Clustering-Based Pseudo-Labeling for Unsupervised Meta-Learning},
  author={Dong, Xingping and Shen, Jianbing and Shao, Ling},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XX},
  pages={169--186},
  year={2022},
  organization={Springer}
}
```


### Getting started

#### Requirements
 - Python 3.6 or above
 - PyTorch 1.2
 - Torchvision 0.4
 - Torchmeta 1.2

To avoid any conflict with your existing Python setup, it is suggested to work in a virtual environment with [`Miniconda`](https://docs.conda.io/en/latest/miniconda.html#miniconda/). 

Create a new environment (PL-CFE) with conda
```bash
bash creat_env.sh
```

### Datasets
#### ImageNet
1. Download the [Imagenet dataset](https://image-net.org/download.php) into $imagenet_dir.
2. Copy extract_ILSVRC.sh to $imagenet_dir
3. Go to $imagenet_dir and run extract_ILSVRC.sh
```bash
cd $imagenet_dir
bash extract_ILSVRC.sh 
```
#### Omniglot and tieredImageNet
We have provide the cache files for Omniglot dataset in './data/omniglot'
Please extract it on './data' and make sure the structure of files is the same to 'data/omniglot/omniglot_cache_train.npz'
```bash
cd data
tar -xf omniglot.tar.xz 
cd ..
```
#### miniImageNet
We download these miniImageNet ([Google Drive](https://drive.google.com/open?id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY))
from [few-shot-ssl](https://github.com/renmengye/few-shot-ssl-public).
Then place 'mini-imagenet.tar.gz' into ./data/miniimagenet/

#### tieredImageNet
You can use 'cfe/download_datasets' to download tieredImageNet.
```bash
cd cfe
python download_datasets.py -d tieredimagenet
cd ..
```
### Usage
We provide shell scripts in ['scripts](scripts) to reproduce our results.
Here, we provide the procedure to train a model on omniglot. (Similar precedure for other datasets)
#### Step 1: Unsupervised training model with our CFE (multiple GPUs)

Notice that this step requires multiple GPUs. 
If you use one GPU for training, you may suffer from performance degradation
```bash
cd scripts
bash train_cfe_omni.sh
# For miniImagenet, please download the ImageNet and then modify the dir_imagenet in train_cfe_mini
# bash train_cfe_imagenet
```
#### Step 2: Extract the embeddings for the few shot dataset
After training, you can use this model to extract the embeddings and make clustering by using ```cfe/cfe_encoding.py```, and save the results as new datasets.
Please do data transform in this step. In the following steps, we would not do any transform. 

We also provide a script as following. Please modify $pretrain_path to select the embedding model.
We provide the pre-trained model in [Google Drive](https://drive.google.com/drive/folders/1veOQ8SfwwMqohfBsJ2kkc4fANDvg7107?usp=sharing)
```bash
bash cfe_encoding.sh
```
#### Step 3: Run the supervised method, such as MAML, on the pseudo labeling dataset based on clustering in our CFE embedding.
```bash
bash omni-maml-pem.sh
```
Please refer 'EP/scripts' to evaluate the EP method.
### Acknowledgement
Some codes are borrowed from the following open-source codes. Thanks for the contributions of these authors.

[pytorch-maml](https://github.com/tristandeleu/pytorch-maml),
[MoCo](https://github.com/facebookresearch/moco), [EP](https://github.com/ServiceNow/embedding-propagation)
