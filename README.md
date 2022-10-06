# Semi-Supervised Semantic Segmentation with Uncertainty-guided Self Cross Supervision

## Getting Started

### Data Preparation 

We employ the dataset (VOC) provided by [CPS](https://github.com/charlesCXK/TorchSemiSeg):
```
datasets/data/
|-- pascal_voc
```

### Training && testing on PASCAL VOC:
Please modify the parameter ```test_only``` in ```config_semi.yml ```for train or test.

```shell
$ python train.py
```

We train VOC for 30K iters with the batch size set to 16 for both labeled and unlabeled data. The epochs are different for different partitions.
We list the epochs for partitions in the below.

| Dataset    | 1/16 | 1/8  | 1/4  | 1/2  |
| ---------- | ---- | ---- | ---- | ---- |
| VOC        | 48   | 52   | 60   | 90   |


###Checkpoints
The models trained on VOC with Resnet50-deeplabv3+ will soon be published.


##Acknowledgement
* [CPS](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Semi-Supervised_Semantic_Segmentation_With_Cross_Pseudo_Supervision_CVPR_2021_paper.pdf): https://github.com/charlesCXK/TorchSemiSeg
* [MixMo](https://arxiv.org/abs/2010.06610): https://github.com/alexrame/mixmo-pytorch
* [CutMix](https://arxiv.org/abs/1905.04899): https://github.com/ildoonet/cutmix