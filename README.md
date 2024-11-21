# Deep-compressed domain image classification

- [Install](#install)
- [Usage](#usage)
- [Licenses](#license)


## Install
**Requirements**:
- Python
- PyTorch >= 2.0 : https://pytorch.org/get-started/locally
- [compressai](https://github.com/InterDigitalInc/CompressAI): `pip install compressai`

For training & utilities:
- [timm](https://github.com/huggingface/pytorch-image-models): `pip install timm`
- [wandb](https://wandb.ai/site/): `pip install wandb`


## Usage
### Pre-trained models
TBD

### Evaluation
TBD

### Training

Commands are shown for single-GPU training.

Without teacher model:
```bash
python train.py --s1 cheng5 --s3 res50-aa --batch_size 128 --equiv_bs 256 --workers 8 --epochs 16 --fixseed --wbmode online
```

With teacher model:
```bash
python train.py --s1 cheng5 --s3 res50-aa --guide l2 --batch_size 128 --equiv_bs 256 --workers 8 --epochs 16 --fixseed --wbmode online
```


## License
TBD