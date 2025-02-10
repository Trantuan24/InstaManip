import hydra
import pyrootutils
import os
import torch
import numpy as np
import random
import argparse
import gc
import logging
import transformers
import torch.nn.functional as F

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from typing import Optional
from dataclasses import dataclass, field, asdict, is_dataclass
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, SequentialReadingService
from torch.utils.data import DataLoader, Dataset

print('============= train code =============')

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.train.schedular import get_scheduler
from src.train.dist_utils import all_gather
from src.metrics.clip_similarity import ClipSimilarity


log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)


@dataclass
class ConfigPathArguments:
    image_transform: Optional[str] = field(default=None, metadata={"help": "config path of image transform"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    visual_encoder: Optional[str] = field(default=None, metadata={"help": "config path of visual encoder"})
    llm_model: Optional[str] = field(default=None, metadata={"help": "config path of llm"})
    agent_model: Optional[str] = field(default=None, metadata={"help": "config path of agent"})
    train_dataset: Optional[str] = field(default=None, metadata={"help": "config path of training dataset"})
    fsdp_plugin: Optional[str] = field(default=None, metadata={"help": "config path of fsdp plugin"})
    deepspeed_plugin: Optional[str] = field(default=None, metadata={"help": "config path of deepspeed plugin"})


@dataclass
class TrainingArguments:
    output_dir: str = field(metadata={"help": "The output directory where the model predictions and checkpoints will be written."}, )
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "The path to a folder with a valid checkpoint for your model."})
    resume_steps: Optional[int] = field(default=None, metadata={"help": "The training sterps of saved checkpoint"})
    batch_size: Optional[int] = field(default=60, metadata={"help": "The training batch size"})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    mixed_precision: Optional[str] = field(default='no', metadata={"help": "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU."})
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(default=-1, metadata={"help": "Total number of training steps to perform. "})
    save_steps: int = field(default=10000, metadata={"help": "Number of updates steps before two checkpoint saves."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The scheduler type to use."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    min_lr_ratio: float = field(default=0.01, metadata={"help": "Minimal learning rate ratio."})
    dataloader_num_workers: int = field(default=8, metadata={"help": "The number of workers to use for data loading."})
    project_name: str = field(default="ContinuousVLM", metadata={"help": "The name of experiment"})
    expr_name: str = field(default="", metadata={"help": "The name of experiment"})


def build_dataloader(dataset_cfg, image_transform, tokenizer, batch_size, dataloader_num_workers=4):
    dataset = hydra.utils.instantiate(dataset_cfg, image_transform=image_transform, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=dataloader_num_workers)
    return dataloader


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


def get_metric(output):
    metric = {}
    for key, value in output.items():
        if 'loss' in key:
            gathered_metric = torch.stack(all_gather(value)).mean()
            metric[key] = gathered_metric.item()
        if 'acc' in key:
            metric[key] = value.item()
    return metric


def merge_config(**kwargs):
    config = {}
    for key, value in kwargs.items():
        if isinstance(value, argparse.Namespace):
            config[key] = vars(value)
        elif isinstance(value, DictConfig):
            config[key] = OmegaConf.to_object(value)
        elif is_dataclass(value):
            config[key] = asdict(value)
        elif isinstance(value, (int, str, float, dict)) or value is None:
            config[key] = value
        else:
            logger.error(f'key: {key}, value: {value} will not be merged.')
    return config


def trainable_params(model):
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count += param.numel()
    return count


def train():

    parser = transformers.HfArgumentParser((ConfigPathArguments, TrainingArguments))
    cfg_path, args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)  # return_remaining_strings returns the unknown args
    remaining_args = {remaining_args[i][2:]: remaining_args[i+1] for i in range(len(remaining_args)) if remaining_args[i][:2] == '--'}

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, 'logs'))

    if 'seed' in remaining_args:
        set_seed(int(remaining_args['seed']))


    assert int(cfg_path.fsdp_plugin is not None) + int(cfg_path.deepspeed_plugin is not None) <= 1
    if cfg_path.fsdp_plugin is not None:  # - None
        fsdp_plugin_cfg = OmegaConf.load(cfg_path.fsdp_plugin)
        fsdp_plugin = hydra.utils.instantiate(fsdp_plugin_cfg)
        logger.info('Use FSDP plugin')
    else:
        fsdp_plugin = None

    if cfg_path.deepspeed_plugin is not None:
        deepspeed_plugin_cfg = OmegaConf.load(cfg_path.deepspeed_plugin)
        deepspeed_plugin = hydra.utils.instantiate(deepspeed_plugin_cfg)
        logger.info('Use deepspeed plugin')
    else:
        deepspeed_plugin = None

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_config=project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin
    )
    accelerator.wait_for_everyone()
    logger.info('Init accelerator done.')

    if cfg_path.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)

    image_transform_cfg = OmegaConf.load(cfg_path.image_transform)
    image_transform = hydra.utils.instantiate(image_transform_cfg)

    tokenizer_cfg = OmegaConf.load(cfg_path.tokenizer)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    train_dataset_cfg = OmegaConf.load(cfg_path.train_dataset)

    visual_encoder_cfg = OmegaConf.load(cfg_path.visual_encoder)
    visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
    logger.info('Load visual encoder done.')

    llm_model_cfg = OmegaConf.load(cfg_path.llm_model)
    llm_model = hydra.utils.instantiate(llm_model_cfg, torch_dtype=accelerator.mixed_precision)
    llm_model.gradient_checkpointing_enable()
    llm_model.config.use_cache = False
    logger.info('Load llm model done.')

    agent_model_cfg = OmegaConf.load(cfg_path.agent_model)
    agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm_model)
    logger.info('Load agent model done.')

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":  # True
        weight_dtype = torch.bfloat16

    visual_encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info('Freeze visual encoder...')
    visual_encoder.requires_grad_(False)

    if cfg_path.fsdp_plugin is not None:  # = None
        agent_model = accelerator.prepare(agent_model)

    optimizer = torch.optim.AdamW(agent_model.parameters(),
                                  lr=args.learning_rate,
                                  betas=[args.adam_beta1, args.adam_beta2],
                                  eps=args.adam_epsilon,
                                  weight_decay=args.weight_decay)
    logger.info('Init optimizer done.')
    scheduler = get_scheduler(name=args.lr_scheduler_type,
                              optimizer=optimizer,
                              num_warmup_steps=args.warmup_steps,
                              num_training_steps=args.max_steps,
                              min_lr_ratio=args.min_lr_ratio)
    train_dataloader = build_dataloader(dataset_cfg=train_dataset_cfg, 
                                        image_transform=image_transform,
                                        tokenizer=tokenizer,
                                        batch_size=args.batch_size,
                                        dataloader_num_workers=args.dataloader_num_workers)
    
    if cfg_path.fsdp_plugin is not None:
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    else:
        agent_model, optimizer, scheduler = accelerator.prepare(agent_model, optimizer, scheduler)
    logger.info('Prepare accelerator done.')

    config_record = merge_config(agent_model=agent_model_cfg,
                                 llm_model=llm_model,
                                 visual_encoder=visual_encoder_cfg,
                                 image_transform=image_transform_cfg,
                                 tokenizer=tokenizer_cfg,
                                 train_dataset=train_dataset_cfg,
                                 train_args=args)
    accelerator.init_trackers(project_name="InstaManip",
                              init_kwargs={"wandb": {
                                  "config": config_record,
                                  "name": args.expr_name,
                                  "dir": args.output_dir,
                                  "resume": 'allow'
                              }})

    if args.resume_from_checkpoint is not None:
        logger.info(f'Load checkpoint from {args.resume_from_checkpoint}')
        accelerator.load_state(args.resume_from_checkpoint)
        torch.cuda.empty_cache()
        gc.collect()

    num_params = trainable_params(agent_model)
    logger.info("***** Running training *****")
    logger.info(f"  Total optimization steps = {args.max_steps}")
    logger.info(f"  Total trainable params = {num_params}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    if args.resume_steps is not None:
        global_step = args.resume_steps
        progress_bar.update(args.resume_steps)

    for epoch in range(args.num_train_epochs):
        agent_model.train()
        logger.info('Start new epoch')

        for step, batch in enumerate(train_dataloader):
    
            with accelerator.accumulate(agent_model):
                # Squeeze --------------------------------
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor) and batch[k].size(0) == 1:
                        batch[k] = batch[k].squeeze(0)
                # ----------------------------------------

                images = batch['images'].to(accelerator.device) if batch['images'] is not None else None
                if images is not None:
                    embeds_gen_mask = batch['embeds_gen_mask'].to(accelerator.device)
                    embeds_cmp_mask = batch['embeds_cmp_mask'].to(accelerator.device)

                    # filter out patch embedding in embeds_gen (only useful for mutli-resolution)
                    embeds_valid_mask = torch.logical_or(embeds_gen_mask, embeds_cmp_mask)
                    embeds_gen_mask = embeds_gen_mask[embeds_valid_mask]  # only select the elements at the position where the value is true in embeds_valid_mask
                    embeds_cmp_mask = embeds_cmp_mask[embeds_valid_mask]
                    images = images[embeds_valid_mask]

                    if 'patch_position' in batch:  # single resolution: False, multi resolution: True
                        patch_position = batch['patch_position'].to(accelerator.device) 
                        patch_position = patch_position[embeds_valid_mask]
                    else:
                        patch_position = None

                    if images.shape[0] == 0:
                        images = None
                
                with torch.no_grad():
                    # Calculate editing similarity matrix [Remember to check] ------------------------------------------------
                    clip = ClipSimilarity()
                    edits = [item[0] for item in batch["edit"]]
                    edits_clip_embeds = clip.encode_text(edits)
                    edits_sim_matrix = edits_clip_embeds @ edits_clip_embeds.transpose(0, 1)
                    edits_clip_embeds=edits_clip_embeds.to(weight_dtype).to(accelerator.device),  # (B, 768)
                    edits_sim_matrix=edits_sim_matrix.to(weight_dtype).to(accelerator.device)  # (B, B)
                    
                    # [or] Not use similarity matrix -----------------------------------------------------
                    # edits_clip_embeds = edits_sim_matrix = None
                    # ------------------------------------------------------------------------------------

                    if images is not None:
                        image_embeds = visual_encoder(images, batch['patch_position'].to(accelerator.device) if 'patch_position' in batch else None)
                    else:
                        image_embeds = None

                output = agent_model(
                    input_ids=batch['input_ids'].to(accelerator.device),  # (B, L)
                    attention_mask=batch['attention_mask'].to(accelerator.device),  # (B, L)
                    labels=batch['labels'].to(accelerator.device),  # (B, L)
                    image_embeds=image_embeds,  # (B', 256, 4096)
                    patch_positions=patch_position if images is not None else None,  # (B', 2)
                    embeds_gen_mask=embeds_gen_mask if batch['embeds_gen_mask'] is not None else None,  # (B')
                    embeds_cmp_mask=embeds_cmp_mask if batch['embeds_cmp_mask'] is not None else None,  # (B')
                    ids_gen_mask=batch['ids_gen_mask'].to(accelerator.device),  # (B, L)
                    ids_cmp_mask=batch['ids_cmp_mask'].to(accelerator.device),  # (B, L)
                    ids_exemplar_source_mask=batch['ids_exemplar_source_mask'].to(accelerator.device),  # (B, L)
                    ids_exemplar_target_mask=batch['ids_exemplar_target_mask'].to(accelerator.device),  # (B, L)
                    ids_latent_edit_mask=batch['ids_latent_edit_mask'].to(accelerator.device),  # (B, L)
                    scope_mask=batch['scope_mask'].to(weight_dtype).to(accelerator.device),  # (B, L, L)
                    edits_clip_embeds=edits_clip_embeds,
                    edits_sim_matrix=edits_sim_matrix
                )
                loss = output['total_loss']

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(agent_model.parameters(), max_norm=args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step:05d}")
                    accelerator.save_state(save_path)

            metric = get_metric(output)
            metric['lr'] = optimizer.param_groups[0]['lr']
            accelerator.log(metric, step=global_step)
            metric = {key: (format(value, ".6f") if isinstance(value, float) else value) for key, value in metric.items()}
            if accelerator.is_main_process:
                tqdm.write(str(metric))

            if global_step >= args.max_steps:
                break

    accelerator.end_training()


if __name__ == '__main__':
    # Enable spawn to avoid errors on H100 cluster ---------------------------
    torch.multiprocessing.set_start_method("spawn")
    # ------------------------------------------------------------------------
    train()
