# Unleashing In-context Learning of Autoregressive Models for Few-shot Image Manipulation

### CVPR 2025 (Highlight)

### [Project Page](https://bolinlai.github.io/projects/InstaManip/) | [Paper](https://arxiv.org/pdf/2412.01027) | [HuggingFace](https://huggingface.co/bolinlai/InstaManip)

#### [Bolin Lai](https://bolinlai.github.io/), [Felix Juefei-Xu](https://xujuefei.com/), [Miao Liu](https://aptx4869lm.github.io/), [Xiaoliang Dai](https://sites.google.com/view/xiaoliangdai/), [Nikhil Mehta](https://hockeybro12.github.io/), [Chenguang Zhu](https://cs.stanford.edu/~cgzhu/), [Zeyi Huang](https://oodbag.github.io/), [James M. Rehg](https://rehg.org/), [Sangmin Lee](https://sites.google.com/view/sangmin-lee), [Ning Zhang](https://n-zhang.github.io/), [Tong Xiao](http://xiaotong.me/)


<img src="https://bolinlai.github.io/projects/InstaManip/figures/teaser.png"/>


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

Download the following pre-trained checkpoints and save them under `./pretrained`.

- [SEED-X-17B](https://huggingface.co/AILab-CVC/SEED-X-17B/tree/main)
- [Stable-Diffusion-XL-Base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main)
- [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat/tree/main)

Move  `cvlm_llama2_tokenizer_100img_and_224loc_addpatch`, `seed_detokenizer` and `seed_x` from `SEED-X-17B` to `./pretrained`.

Replace the `added_tokens.json` under `cvlm_llama2_tokenizer_100img_and_224loc_addpatch` with our released json file in `./pretrained`.

```shell
mv ./pretrained/added_tokens.json ./pretrained/cvlm_llama2_tokenizer_100img_and_224loc_addpatch/
```

Please run the following script to save the weights of visual encoder of `Qwen-VL-Chat` to `./pretrained/QwenViT`.

```shell
python src/tools/reload_qwen_vit.py
```

Finally, you should have the following directories under `./pretrained`. We don't need the other files.

```
./pretrained
     |
     |- QwenViT
     |- cvlm_llama2_tokenizer_100img_and_224loc_addpatch
     |- seed_detokenizer
     |- seed_x
     |- stable-diffusion-xl-base-1.0
```


## Model Weights

Our model weights are available on [HuggingFace](https://huggingface.co/bolinlai/InstaManip). There are four models released in this repo.

- InstaManip-17B-1shot: model trained specifically for 1-shot image manipulation.
- InstaManip-17B-2shot: model trained specifically for 2-shot image manipulation.
- InstaManip-17B-3shot: model trained specifically for 3-shot image manipulation.
- InstaManip-17B-dynamic: model trained for arbitrary amount of exemplar image pairs.

## Quick Start

We provide a few examples in `./demo` for a quick start of our model. After setting up the environment and downloading all pre-trained checkpoints and our model weight, run the following command to edit a given image.

```shell
# 1-shot
python src/inference/run_model.py --ckpt ./train_output/your_path/checkpoint-xxxx/pytorch_model.bin

# multi-shot
python src/inference/run_model_multishot.py --ckpt ./train_output/your_path/checkpoint-xxxx/pytorch_model.bin
```

You can try different examples or use your own image by updating `source_image_path`, `exemplar_source_image_path`, `exemplar_target_image_path` and `instruction` in `src/inference/run_model.py` and `src/inference/run_model_multishot.py`.


## Training

Run the following command to train the model on 8 GPUs. You can change the number of GPUs by updating `--nproc_per_node` in `train.sh`.

```shell
bash scripts/train.sh
```

You can use different hyperparameters in `scripts/train.sh` (e.g., learning rate, iterateions) and `configs/data/dataset.yaml` (e.g., batch size, number of exemplar images).

We also enable `torch.multiprocessing.set_start_method("spawn")` in `scripts/train.sh` for training on H100. If you run the code on A100, this line can be commented out for faster training.


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

Our work was developed based on [SEED-X](https://github.com/AILab-CVC/SEED-X). We appreciate the contributors for their awesome codebase.
