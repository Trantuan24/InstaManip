# Unleashing In-context Learning of Autoregressive Models for Few-shot Image Manipulation

### [Project Page](https://bolinlai.github.io/) | [Paper](https://drive.google.com/file/d/1JF4sUdfFAh9ujMQ-P30eTFvejUTxkA5D/view?usp=drive_link)

### Updating...

<img src="https://bolinlai.github.io/projects/InstaManip/figures/teaser.png"/>

### TODO

- [ ] Release the code for single image inference.
- [ ] Release the code for multi-shot inference.
- [ ] Update README of training and evaluation.
- [ ] Release model weights.


## Contents

- [Setup](#setup)
- [Model Weights](#model-weights)
- [Quick Start](#quick-start)
- [Traininng](#training)
- [Evaluation](#evaluation)
- [BibTex](#bibtex)
- [Acknowledge](#acknowledgement)


## Setup

### Environment

```shell
conda env create -f environment.yaml  # THe env name is "instamanip"
```

### Dataset

### Pre-trained Checkpoints


## Model Weights (Coming)


## Quick Start


## Training

```shell
bash scripts/train.sh
```


## Evaluation

Go to the checkpont directory your want to evaluate.
 Convert the model weights.
```shell
python zero_to_fp32.py . ./pytorch_model.bin
```

Go back to the project root directory.

```shell
# In distribution
python src/inference/eval_model.py --ckpt ./train_out/your_path/checkpoints-xxxx/pytorch_model.bin -- setting in_dist

# Out of distribution
python src/inference/eval_model.py --ckpt ./train_out/your_path/checkpoints-xxxx/pytorch_model.bin -- setting out_of_dist
```


## Metrics

```shell
python src/metrics/metrics.py  --gen_path ./train_output/your_path/checkpoints-xxxx/inference-xxxx-xx
```


## BibTex

If you find our paper helpful to your work, please cite with this BibTex.

```BibTex
@article{lai2024unleashing,
  title={Unleashing In-context Learning of Autoregressive Models for Few-shot Image Manipulation},
  author={Lai, Bolin and Juefei-Xu, Felix and Liu, Miao and Dai, Xiaoliang and Mehta, Nikhil and Zhu, Chenguang and Huang, Zeyi and Rehg, James M and Lee, Sangmin and Zhang, Ning and others},
  journal={arXiv preprint arXiv:2412.01027},
  year={2024}
}
```


## Acknowledgement
