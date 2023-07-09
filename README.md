This repository can be used to reproduce plots and experiments from our ACL'23 paper:
[Grokking of Hierarchical Structure in Vanilla Transformers](https://arxiv.org/abs/2305.18741). Almost all of the structure of this codebase is borrowed from our earlier work on [intrinsic compositionality](https://github.com/MurtyShikhar/TreeProjections).


## Dependencies:
Note that this code has been tested with `python 3.8.10`.
1. Install conda environment: 

```
conda env create -f environment.yml
conda activate tree-reg
pip install -e .
```

## Data:
Datasets used in this work can be found under data_utils. More specificially, `data_utils/dyck_data` contains Dyck, `data_utils/question_formation_data` contains Question-Formation and `data_utils/tense_inflection_data` contains Tense-Inflection.

## Training
The script in `train_transformers.py` may be used to train transformer LMs of various sizes. Below trains a 6 layer transformer LM on question-formation.
```
# checkpoints saved under /path/to/save/dir
python train_transformers.py --dataset lm --save_dir /path/to/save/dir --encoder_depth 6
```

To modify number of training steps / checkpoint saving frequency, please take a look at `trainin_utils.py`. 

## Computing Tree Projections
For computing tree projections, take a look at `transformer_lm_sci.py`. We provide a minimal implementation of the tree projections method (**for auto-regressive LMs**) in the `tree_projections` folder. 

> :warning: Please note that this implementation is specifically made for models with **causal masking** and will not work for bi-directional models. If you want an implementation for bidirectional models, check out the original [Tree Projections codebase](https://github.com/MurtyShikhar/TreeProjections).

## Reproducing Plots
We provide all the raw data for reproducing plots in our paper. Please run the notebooks in the `notebooks` folder, after unzipping `sci.tar.gz`.

## Citing this work
If you use ideas from this paper in your work, we kindly ask you to cite us as:
```
@inproceedings{
murty2023structure,
title={Grokking of Hierarchical Structure in Vanilla Transformers},
author={Shikhar Murty and Pratyusha Sharma and Jacob Andreas and Christopher D Manning},
booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
year={2023},
}
```


## Acknowledgements
A lot of our code is built on top of the following repositories:
- [The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers](https://github.com/RobertCsordas/transformer_generalization)
- [Does syntax need to grow on trees? Sources of hierarchical inductive bias in sequence to sequence networks](https://github.com/tommccoy1/rnn-hierarchical-biases.git)

