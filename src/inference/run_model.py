import hydra
import torch
import os
import re
import pyrootutils
import numpy as np
import random
import xformers.ops as xops

from argparse import ArgumentParser
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Setting this as False may impact the performance of cudnn
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


set_seed(42)

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

parser = ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoints, e.g. /data/home/bolinlai/Projects/FreeIm2Im_Dev/train_output/edit_ip2p/checkpoint-02000/pytorch_model.bin")
args = parser.parse_args()


BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'
BOE_TOKEN = '<edit>'
EOE_TOKEN = '</edit>'
EDIT_TOKEN = '<edit_{:03d}>'

multi_resolution = False
resolution_grids = ['1x1']
base_resolution = 448

device = 'cuda'
dtype = torch.float16
num_img_in_tokens = 64
num_img_out_tokens = 64
num_latent_edit_tokens = 30
instruction_prompt = '[INST] {instruction} [/INST]\n'
generated_resolution = (1024, 1024)

tokenizer_cfg_path = 'configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml'
image_transform_cfg_path = 'configs/processer/qwen_448_transform.yaml'
visual_encoder_cfg_path = 'configs/visual_encoder/qwen_vitg_448.yaml'
llm_cfg_path = 'configs/clm_models/llm_seed_x_lora.yaml'
agent_cfg_path = 'configs/clm_models/agent_seed_x.yaml'
adapter_cfg_path = 'configs/sdxl_adapter/sdxl_qwen_vit_resampler_l4_q64_full_with_latent_image_pretrain_no_normalize.yaml'
discrete_model_cfg_path = 'configs/discrete_model/discrete_identity.yaml'

diffusion_model_path = 'pretrained/stable-diffusion-xl-base-1.0'

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)

image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
image_transform = hydra.utils.instantiate(image_transform_cfg)

visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device, dtype=dtype)
print('Init visual encoder done')

llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)
print('Init llm done.')

agent_model_cfg = OmegaConf.load(agent_cfg_path)
if args.ckpt is not None:
    agent_model_cfg['pretrained_model_path'] = args.ckpt
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)
print('Agent model args:', agent_model_cfg)

agent_model.eval().to(device, dtype=dtype)
print('Init agent mdoel Done')

noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
print('init vae')
vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device, dtype=dtype)
print('init unet')
unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device, dtype=dtype)

adapter_cfg = OmegaConf.load(adapter_cfg_path)
adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device, dtype=dtype).eval()

discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device).eval()
print('Init adapter done')

adapter.init_pipe(vae=vae,
                  scheduler=noise_scheduler,
                  visual_encoder=visual_encoder,
                  image_transform=image_transform,
                  dtype=dtype,
                  device=device)
print('Init adapter pipe done')

boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

# Add <edit> and </edit> +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
boe_token_id = tokenizer.encode(BOE_TOKEN, add_special_tokens=False)[0]  # 32330
eoe_token_id = tokenizer.encode(EOE_TOKEN, add_special_tokens=False)[0]  # 32331
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Examples for a quick start -----------------------------------
source_image_path = "./demo/source_1.jpg"
instruction = "Make the person a toddler"
exemplar_source_image_path = "./demo/exemplar_source_1.jpg"
exemplar_target_image_path = "./demo/exemplar_target_1.jpg"

# source_image_path = "./demo/source_1.jpg"
# instruction = "Make it a tropical landscape"
# exemplar_source_image_path = "./demo/exemplar_source_2.jpg"
# exemplar_target_image_path = "./demo/exemplar_target_2.jpg"

# source_image_path = "./demo/source_1.jpg"
# instruction = "Make it sepia toned"
# exemplar_source_image_path = "./demo/exemplar_source_3.jpg"
# exemplar_target_image_path = "./demo/exemplar_target_3.jpg"

# source_image_path = "./demo/source_1.jpg"
# instruction = "Have him be the joker"
# exemplar_source_image_path = "./demo/exemplar_source_4.jpg"
# exemplar_target_image_path = "./demo/exemplar_target_4.jpg"

# source_image_path = "./demo/source_1.jpg"
# instruction = "Make him wear a plaid clothing"
# exemplar_source_image_path = "./demo/exemplar_source_5.jpg"
# exemplar_target_image_path = "./demo/exemplar_target_5.jpg"
# --------------------------------------------------------------

image = Image.open(source_image_path).convert('RGB')
source_image = image.resize(generated_resolution)
exemplar_source_image = Image.open(exemplar_source_image_path).convert('RGB')
exemplar_source_image = exemplar_source_image.resize(generated_resolution)
exemplar_target_image = Image.open(exemplar_target_image_path).convert('RGB')
exemplar_target_image = exemplar_target_image.resize(generated_resolution)

image_tensor = image_transform(image).unsqueeze(0)
embeds_cmp_mask = torch.tensor([True]).to(device, dtype=torch.bool)
exemplar_source_image_tensor = image_transform(exemplar_source_image).unsqueeze(0)
exemplar_source_embeds_cmp_mask = torch.tensor([True]).to(device, dtype=torch.bool)
exemplar_target_image_tensor = image_transform(exemplar_target_image).unsqueeze(0)
exemplar_target_embeds_cmp_mask = torch.tensor([True]).to(device, dtype=torch.bool)

image_tokens = ''
image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)]) + EOI_TOKEN
exemplar_source_image_tokens = ''
exemplar_source_image_tokens += BOI_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN
exemplar_target_image_tokens = ''
exemplar_target_image_tokens += BOI_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN

patch_position = exemplar_source_patch_position = exemplar_target_patch_position = None

image_tensor = image_tensor.to(device, dtype=dtype)
exemplar_source_image_tensor = exemplar_source_image_tensor.to(device, dtype=dtype)
exemplar_target_image_tensor = exemplar_target_image_tensor.to(device, dtype=dtype)

quoted_instruction = f'"{instruction}"'
latent_edit_tokens = BOE_TOKEN + ''.join([EDIT_TOKEN.format(int(item)) for item in range(num_latent_edit_tokens)]) + EOE_TOKEN
icl_instruction = f'Here is an image manipulation instruction {quoted_instruction}, which can edit source image {exemplar_source_image_tokens} to target image {exemplar_target_image_tokens}. The editing is embedded in {latent_edit_tokens}. Learn from the instruction with the exemplar pairs and apply the same manipulation to this image {image_tokens}.'
prompt = instruction_prompt.format_map({'instruction': icl_instruction})

input_ids = tokenizer.encode(prompt, add_special_tokens=False)
input_ids = [tokenizer.bos_token_id] + input_ids
input_ids = torch.tensor(input_ids).to(device, dtype=torch.long)

ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
ids_exemplar_target_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
scope_mask = xops.LowerTriangularMask().materialize(shape=(input_ids.shape[0], input_ids.shape[0]), dtype=dtype, device=device)

boi_indices = torch.where(input_ids == boi_token_id)[0].tolist()
eoi_indices = torch.where(input_ids == eoi_token_id)[0].tolist()

for boi_idx, eoi_idx in zip(boi_indices, eoi_indices):
    ids_cmp_mask[boi_idx+1:eoi_idx] = True

for boi_idx, eoi_idx in zip(boi_indices[1:-1:2], eoi_indices[1:-1:2]):
    ids_exemplar_target_mask[boi_idx+1:eoi_idx] = True

# Add masks for group self-attention +++++++++++++++++++++++++++++++++++++++++++++++++
boe_indices = torch.where(input_ids == boe_token_id)[0].tolist()
eoe_indices = torch.where(input_ids == eoe_token_id)[0].tolist()
for boe_idx, eoe_idx in zip(boe_indices, eoe_indices):
    last_exemplar_target_eoi = eoi_indices[1:-1:2][-1]
    scope_mask[eoe_idx+1:, :last_exemplar_target_eoi+1] = -float('inf')
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

input_ids = input_ids.unsqueeze(0)
ids_cmp_mask = ids_cmp_mask.unsqueeze(0)
ids_exemplar_target_mask = ids_exemplar_target_mask.unsqueeze(0)
scope_mask = scope_mask.unsqueeze(0)

with torch.no_grad():
    image_embeds = visual_encoder(image_tensor)
    exemplar_source_image_embeds = visual_encoder(exemplar_source_image_tensor)
    exemplar_target_image_embeds = visual_encoder(exemplar_target_image_tensor)
    output = agent_model.generate(tokenizer=tokenizer,
                                  input_ids=input_ids,
                                  image_embeds={'new': image_embeds, 'exemplar_source': exemplar_source_image_embeds, 'exemplar_target': exemplar_target_image_embeds},
                                  embeds_cmp_mask={'new': embeds_cmp_mask, 'exemplar_source': exemplar_source_embeds_cmp_mask, 'exemplar_target': exemplar_target_embeds_cmp_mask},
                                  patch_positions=None if patch_position is None else {'new': patch_position, 'exemplar_source': exemplar_source_patch_position, 'exemplar_target': exemplar_target_patch_position},
                                  ids_cmp_mask=ids_cmp_mask,
                                  ids_exemplar_target_mask=ids_exemplar_target_mask,
                                  scope_mask=scope_mask,
                                  max_new_tokens=500,
                                  num_img_gen_tokens=num_img_out_tokens)
text = re.sub('<[^>]*>', '', output['text'])
print(text)

if output['has_img_output']:
    images = adapter.generate(image_embeds=output['img_gen_feat'], latent_image=source_image, num_inference_steps=50)
    save_path = f"./demo/output/{os.path.basename(source_image_path).split('.')[0]}/{instruction.replace(' ', '_')}_by_{os.path.basename(exemplar_source_image_path).replace('source', 'pair')}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    images[0].save(save_path)

torch.cuda.empty_cache()
