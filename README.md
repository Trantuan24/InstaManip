# Unleashing In-context Learning of Autoregressive Models for Few-shot Image Manipulation

### [Project Page](https://bolinlai.github.io/) | [Paper](https://drive.google.com/file/d/1JF4sUdfFAh9ujMQ-P30eTFvejUTxkA5D/view?usp=drive_link)

### Updating...

<img src="https://bolinlai.github.io/projects/InstaManip/figures/teaser.png"/>

### TODO

- [ ] Release the code for single image inference.
- [x] Release the code for multi-shot inference.
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
conda env create -f environment.yaml  # The env name is "instamanip"
```

### Dataset

Download the dataset collected in the work [InstructPix2Pix](https://instruct-pix2pix.eecs.berkeley.edu/clip-filtered-dataset/). Unzip all the 30 zip files into the path `./data/ip2p/`.

### Pre-trained Checkpoints


## Model Weights (Coming)


## Quick Start


## Training

Run the following command to train the model on 8 GPUs. You can change the number of GPUs by updating `--nproc_per_node` in `train.sh`.

```shell
bash scripts/train.sh
```


## Evaluation

Go to the checkpont directory that you want to evaluate. Convert the model weights.

```shell
python zero_to_fp32.py . ./pytorch_model.bin
```

Go back to the project root directory and run the following commands. The inference results will be saved in `checkpoint-xxxx/inference-xxxx-xx`.

Using one pair of examplar images (1-shot):

```shell
# In distribution
python src/inference/eval_model.py --ckpt ./train_output/your_path/checkpoint-xxxx/pytorch_model.bin --setting in_dist

# Out of distribution
python src/inference/eval_model.py --ckpt ./train_output/your_path/checkpoint-xxxx/pytorch_model.bin --setting out_of_dist
```

Using multiple examplar images (few-shot):


```shell
# In distribution
python src/inference/eval_model_multishot.py --ckpt ./train_output/your_path/checkpoint-xxxx/pytorch_model.bin --example_num 2 --setting in_dist

# Out of distribution
python src/inference/eval_model_multishot.py --ckpt ./train_output/your_path/checkpoint-xxxx/pytorch_model.bin --example_num 2 --setting out_of_dist

# Out of distribution (diverse)
python src/inference/eval_model_multishot.py --ckpt ./train_output/your_path/checkpoint-xxxx/pytorch_model.bin --example_num 2 --setting out_of_dist_diverse
```

Most instructions have 3-4 instances in the dataset of IP2P. The model will use duplicate exemplar images if ``example_num`` is set above the available instances.


## Metrics

Run the following command.

```shell
python src/metrics/metrics.py  --gen_path ./train_output/your_path/checkpoint-xxxx/inference-xxxx-xx
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
