# EEGNet-tutorial
This is a naive Pytorch implementation of [EEGNet](https://arxiv.org/abs/1611.08024)
We don't consider data preprocessing, visualization , ablation test, within/cross-subject experiments.

## Setup environment
1. From conda (**recommended**)
```
conda env create -f environment.yml
```

2. From pip
```
pip install -r requirements.txt
```

## Run
1. 
```
python run.py
```
2. Distributed tuning hyperparameters 
* Using [ray tune](https://docs.ray.io/en/latest/tune.html)
```
python tuning.py
```
