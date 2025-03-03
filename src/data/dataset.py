import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import numpy as np
import torch
import torch.distributed as dist
import os
import random
from braceexpand import braceexpand
import hydra
import json
import xformers.ops as xops

import pyrootutils


pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

dynamic_padding = False

BOI_TOKEN = '<img>'
BOP_TOKEN = '<patch>'
EOI_TOKEN = '</img>'
EOP_TOKEN = '</patch>'
IMG_TOKEN = '<img_{:05d}>'
BOE_TOKEN = '<edit>'
EOE_TOKEN = '</edit>'
EDIT_TOKEN = '<edit_{:03d}>'

gen_prompt_response = [
    "Here is a picture.",
    "I have designed an image.",
    "Here is a photo.",
    "I have generated an image.",
    "Here's a painting.",
    "Here's a drawing.",
    "Enjoy this illustration.",
    "Take a look at this image.",
    "Here is a picture.",
    "I have created a photo.",
    "Enjoy this photo.",
    "I have generated a picture.",
    "Here is a photograph.",
    "Here's an image.",
    "Certainly, here's an image.",
    "Absolutely, here is a painting.",
    "Sure, here is a picture.",
    "Of course, here is a photo.",
    "Certainly, please enjoy this picture.",
    "Sure, please enjoy this illustration.",
    "",
]


def build_multi_datapipes(datapipes, tokenizer=None, image_transform=None, sample_weights=None):
    if sample_weights is None:
        sample_weights = [1] * len(datapipes)
    else:
        assert len(sample_weights) == len(datapipes)

    datapipes = [hydra.utils.instantiate(datapipe, tokenizer=tokenizer, image_transform=image_transform) for datapipe in datapipes]

    datasets_to_weights_dict = {}
    for dataset, sample_weight in zip(datapipes, sample_weights):
        datasets_to_weights_dict[dataset] = sample_weight
    datapipe = dp.iter.SampleMultiplexer(datasets_to_weights_dict, seed=42 + dist.get_rank())

    return datapipe


def filter_data_with_image_ids(item):
    if ('images' not in item):
        return False
    elif 'input_ids' not in item:
        return False
    else:
        return True


def decode_in_context_learning_data(item,
                                    image_dir,
                                    data_group, 
                                    tokenizer,
                                    image_transform=None,
                                    max_length=128,
                                    min_resolution=400,
                                    instruction_prompt='[INST] {instruction} [/INST]\n',
                                    system_message='',
                                    min_aspect_ratio=0.666,
                                    textual_instruction_drop_ratio=0.0,
                                    use_polite_response=True,
                                    num_img_in_tokens=64,
                                    num_img_out_tokens=64,
                                    num_latent_edit_tokens=30,
                                    num_exemplar_pair=1,
                                    dynamic_exemplar_num=False):
    key, value = item
    if 'source_image' not in value or 'target_image' not in value or 'instruction' not in value:
        return {}

    source_image_path = os.path.join(image_dir, value['source_image'])
    target_image_path = os.path.join(image_dir, value['target_image'])

    source_image = Image.open(source_image_path).convert('RGB')
    target_image = Image.open(target_image_path).convert('RGB')
    width, height = source_image.size

    group_id = source_image_path.split('/')[-2]
    image_id = source_image_path.split('/')[-1].split('_')[0]
    exemplar_candidates = [item for item in data_group[group_id] if item != image_id]
    max_num_exemplar_pair = num_exemplar_pair  # maximal example number for dynamic shot
    if len(exemplar_candidates) > 0:
        if dynamic_exemplar_num:
            num_exemplar_pair = random.choice(list(range(1, num_exemplar_pair+1)))

        if num_exemplar_pair == 1:
            exemplar_image_id = random.choice(exemplar_candidates)
            exemplar_source_image_path = os.path.join(image_dir, group_id, f'{exemplar_image_id}_0.jpg')
            exemplar_target_image_path = os.path.join(image_dir, group_id, f'{exemplar_image_id}_1.jpg')
            exemplar_source_image = Image.open(exemplar_source_image_path).convert('RGB')
            exemplar_target_image = Image.open(exemplar_target_image_path).convert('RGB')
            have_exemplar_images = True
        else:
            exemplar_image_ids = random.sample(exemplar_candidates, k=min(num_exemplar_pair, len(exemplar_candidates)))  # random select without returning
            if len(exemplar_image_ids) < num_exemplar_pair:
                exemplar_image_ids.extend(random.choices(exemplar_candidates, k=num_exemplar_pair-len(exemplar_image_ids)))
            random.shuffle(exemplar_image_ids)
            exemplar_images = list()
            for im_id in exemplar_image_ids:
                exemplar_source_image_path = os.path.join(image_dir, group_id, f'{im_id}_0.jpg')
                exemplar_target_image_path = os.path.join(image_dir, group_id, f'{im_id}_1.jpg')
                exemplar_images.append(Image.open(exemplar_source_image_path).convert('RGB'))
                exemplar_images.append(Image.open(exemplar_target_image_path).convert('RGB'))
                have_exemplar_images = True
    else:  # don't use in-context learning if there's only one image in the group
        have_exemplar_images = False
        return {}

    aspect_ratio = height / width
    if height < min_resolution or width < min_resolution:
        print(f'filtered because resolution: ({width}, {height})')
        return {}
    if aspect_ratio < min_aspect_ratio or aspect_ratio > 1 / min_aspect_ratio:
        print(f'filtered because aspect ratio: ({width}, {height})')
        return {}
        
    if have_exemplar_images is True:
        if num_exemplar_pair == 1:
            images = [exemplar_source_image, exemplar_target_image, source_image, target_image]
        else:
            images = [*exemplar_images, source_image, target_image]
    else:
        images = [source_image, target_image]

    if image_transform is not None:
        images = [image_transform(image) for image in images]
        images = torch.stack(images, dim=0)
        expected_num_images = max_num_exemplar_pair * 2 + 2  # num_exemplar_pair * 2 + source + target
        if dynamic_exemplar_num and images.shape[0] < expected_num_images:  # fill with black image placeholders for batch training
            placeholder = torch.zeros(size=(expected_num_images - images.shape[0], images.shape[1], images.shape[2], images.shape[3]), dtype=images.dtype, device=images.device)
            images = torch.concat([images, placeholder], dim=0)

        image_tokens = ''
        image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)]) + EOI_TOKEN

        embeds_cmp_mask = [True, False]
        embeds_gen_mask = [False, True]

        if have_exemplar_images is True:
            exemplar_source_image_tokens = ''
            exemplar_source_image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)]) + EOI_TOKEN
            exemplar_target_image_tokens = ''
            exemplar_target_image_tokens += BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)]) + EOI_TOKEN

            embeds_cmp_mask = [True, True] * num_exemplar_pair + embeds_cmp_mask
            embeds_gen_mask = [False, False] * num_exemplar_pair + embeds_gen_mask

            if dynamic_exemplar_num and len(embeds_cmp_mask) < expected_num_images:
                embeds_cmp_mask = embeds_cmp_mask + [False] * (expected_num_images - len(embeds_cmp_mask))
                embeds_gen_mask = embeds_gen_mask + [False] * (expected_num_images - len(embeds_gen_mask))

    input_ids = []
    labels = []
    input_text = ''

    if system_message != '':
        if not system_message.endswith('\n'):
            system_message += '\n'
        input_text += system_message
        item_ids = tokenizer.encode(system_message, add_special_tokens=False)
        item_labels = [-100] * len(item_ids)
        input_ids.extend(item_ids)
        labels.extend(item_labels)

    if 'instruction_new' in value and 'response' in value:
        instruction = value['instruction_new']
        response = value['response']
    else:
        instruction = value['instruction']
        response = random.choice(gen_prompt_response)

    quoted_instruction = f'"{instruction}"'
    drop_prompt = np.random.uniform(0, 1) < textual_instruction_drop_ratio
    if drop_prompt is True:
        instruction = ''
        quoted_instruction = ''
    if not use_polite_response:
        response = ''

    if have_exemplar_images is True:
        image_in_start = np.random.uniform(0, 1) < 0.5
        latent_edit_tokens = BOE_TOKEN + ''.join([EDIT_TOKEN.format(int(item)) for item in range(num_latent_edit_tokens)]) + EOE_TOKEN
        if num_exemplar_pair == 1:
            icl_instruction = f'Here is an image manipulation instruction {quoted_instruction}, which can edit source image {exemplar_source_image_tokens} to target image {exemplar_target_image_tokens}. The editing is embedded in {latent_edit_tokens}. Learn from the instruction with the exemplar pairs and apply the same manipulation to this image {image_tokens}.'
        else:
            snippet = f" source image {exemplar_source_image_tokens} to target image {exemplar_target_image_tokens}" * num_exemplar_pair
            icl_instruction = f'Here is an image manipulation instruction {quoted_instruction}, which can edit{snippet}. The editing is embedded in {latent_edit_tokens}. Learn from the instruction with the exemplar pairs and apply the same manipulation to this image {image_tokens}.'
    else:
        image_in_start = np.random.uniform(0, 1) < 0.5
        if image_in_start:
            icl_instruction = f'Apply the instruction "{instruction}" to this image {image_tokens}.'
        else:
            icl_instruction = f'{image_tokens} Apply the instruction "{instruction}" to this image.'

    instruction_prompt = instruction_prompt.format_map({'instruction': icl_instruction})

    image_gen_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(num_img_out_tokens)]) + EOI_TOKEN
    response = response + image_gen_tokens
    
    item_ids = tokenizer.encode(instruction_prompt, add_special_tokens=False)
    item_labels = [-100] * len(item_ids)
    input_text += instruction_prompt
    input_ids.extend(item_ids)
    labels.extend(item_labels)

    item_ids = tokenizer.encode(response, add_special_tokens=False)
    item_labels = item_ids
    input_text += response
    input_ids.extend(item_ids)
    labels.extend(item_labels)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] + labels + [tokenizer.eos_token_id]

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]  # 32100
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]  # 32101
    ids_cmp_mask = [False] * len(input_ids)
    ids_gen_mask = [False] * len(input_ids)
    ids_exemplar_target_mask = [False] * len(input_ids)

    # Use manipulation tokens =====================================================================
    boe_token_id = tokenizer.encode(BOE_TOKEN, add_special_tokens=False)[0]  # 32330
    eoe_token_id = tokenizer.encode(EOE_TOKEN, add_special_tokens=False)[0]  # 32331
    ids_latent_edit_mask = [False] * len(input_ids)
    # ============================================================================================

    if len(input_ids) >= max_length:
        print('An edit sample has been removed because of max length.', len(input_ids))
        return {}
    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length
        ids_exemplar_target_mask = ids_exemplar_target_mask + [False] * padding_length
        ids_latent_edit_mask = ids_latent_edit_mask + [False] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
    ids_exemplar_target_mask = torch.tensor(ids_exemplar_target_mask, dtype=torch.bool)
    ids_latent_edit_mask = torch.tensor(ids_latent_edit_mask, dtype=torch.bool)
    embeds_cmp_mask = torch.tensor(embeds_cmp_mask) if embeds_cmp_mask is not None else None
    embeds_gen_mask = torch.tensor(embeds_gen_mask) if embeds_gen_mask is not None else None

    scope_mask = xops.LowerTriangularMask().materialize(shape=(max_length, max_length), dtype=images.dtype)

    boi_indices = torch.where(input_ids == boi_token_id)[0].tolist()
    eoi_indices = torch.where(input_ids == eoi_token_id)[0].tolist()
    for boi_idx, eoi_idx in zip(boi_indices[:-1], eoi_indices[:-1]):
        ids_cmp_mask[boi_idx+1:eoi_idx] = True
    ids_gen_mask[boi_indices[-1]+1:eoi_indices[-1]] = True

    for boi_idx, eoi_idx in zip(boi_indices[1:-2:2], eoi_indices[1:-2:2]):
        ids_exemplar_target_mask[boi_idx+1:eoi_idx] = True

    # Add masks for group self-attention ****************************************
    boe_indices = torch.where(input_ids == boe_token_id)[0].tolist()
    eoe_indices = torch.where(input_ids == eoe_token_id)[0].tolist()
    for boe_idx, eoe_idx in zip(boe_indices, eoe_indices):
        ids_latent_edit_mask[boe_idx+1:eoe_idx] = True
        if len(eoi_indices) > 2:  # have exemplar images
            last_exemplar_target_eoi = eoi_indices[1:-2:2][-1]
            scope_mask[eoe_idx+1:, :last_exemplar_target_eoi+1] = -float('inf')
        else:  # no exemplar images
            scope_mask[eoe_idx+1:, :boe_indices[0]+1] = -float('inf')
    # ***************************************************************************

    labels[boi_indices[-1]+1:eoi_indices[-1]+1] = -100

    ret = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'ids_exemplar_target_mask': ids_exemplar_target_mask,
        'ids_latent_edit_mask': ids_latent_edit_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        'scope_mask': scope_mask,
        'images': images,
        'text': input_text,
        'edit': instruction
    }
    
    return ret


def single_turn_edit_collate(batch):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            if key in ['embeds_gen_mask', 'embeds_cmp_mask', 'images']:
                results[key] = torch.cat(cur, dim=0)
            else:
                results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    return results


def build_in_context_learning_datapipes(data_dir,
                                        image_dir,
                                        data_group_path,
                                        tokenizer=None,
                                        max_length=77,
                                        batch_size=None,
                                        min_resolution=180,
                                        image_transform=None,
                                        instruction_prompt='[INST] {instruction} [INST]\n',
                                        system_message='',
                                        min_aspect_ratio=0.666,
                                        textual_instruction_drop_ratio=0.0,
                                        use_polite_response=True,
                                        num_img_in_tokens=64,
                                        num_img_out_tokens=64,
                                        num_latent_edit_tokens=30,
                                        num_exemplar_pair=1,
                                        dynamic_exemplar_num=False,
                                        cycle_count=None,
                                        dataset_name=None):
    """
    Datapipe of ICL dataset (such as InstructPix2Pix) with webdataset format
    """

    with open(data_group_path, 'r') as infile:
        data_group = json.load(infile)

    # Adujst max_length and batch_size adaptively (for 80G GPU memory) ----------------------
    assert num_exemplar_pair > 0, "You need to set a positive number of exemplar pairs."
    if num_exemplar_pair == 1:
        pass
    elif num_exemplar_pair == 2:
        batch_size = 40
        max_length = 650
    else:
        batch_size = 40 - (num_exemplar_pair - 2) * 5
        max_length = 650 + (num_exemplar_pair - 2) * 100
    # ---------------------------------------------------------------------------------------

    decode_partial = functools.partial(decode_in_context_learning_data,
                                       image_dir=image_dir,
                                       data_group=data_group,
                                       tokenizer=tokenizer,
                                       image_transform=image_transform,
                                       max_length=max_length,
                                       instruction_prompt=instruction_prompt,
                                       system_message=system_message,
                                       min_resolution=min_resolution,
                                       min_aspect_ratio=min_aspect_ratio,
                                       textual_instruction_drop_ratio=textual_instruction_drop_ratio,
                                       use_polite_response=use_polite_response,
                                       num_img_in_tokens=num_img_in_tokens,
                                       num_img_out_tokens=num_img_out_tokens,
                                       num_latent_edit_tokens=num_latent_edit_tokens,
                                       num_exemplar_pair=num_exemplar_pair,
                                       dynamic_exemplar_num=dynamic_exemplar_num)

    filter_partial = functools.partial(filter_data_with_image_ids)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))
    datapipe = dp.iter.FileLister(root=data_dir, masks='*.jsonl', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.shuffle(buffer_size=512)
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(filter_partial)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate(collate_fn=single_turn_edit_collate)
    return datapipe
