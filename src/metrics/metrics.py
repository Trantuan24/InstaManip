import os
import time
import sys
# sys.path.append("./")

import json
import torch
import transformers
import torchvision.transforms.functional as F

from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from clip_similarity import ClipSimilarity
from torchvision import transforms
# from torchvision.transforms._transforms_video import NormalizeVideo
# from torchvision.transforms.functional import InterpolationMode
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import AutoFeatureExtractor, AutoModel


def read_jsonl(file_path):
    with open(file_path, 'r') as infile:
        data = [json.loads(line) for line in infile]
    return data


def compute_fid(image_path, gen_path):
    fid = FrechetInceptionDistance()
    gen_images, target_images = list(), list()

    for category in tqdm(os.listdir(gen_path), desc='FID'):
        if ".json" in category:
            continue
        for group in tqdm(os.listdir(os.path.join(gen_path, category)), leave=False):
            for sample in os.listdir(os.path.join(gen_path, category, group)):
                if '_gen' not in sample:
                    continue
                gen_image_path = os.path.join(gen_path, category, group, sample)
                target_image_path = os.path.join(image_path, group.split('_')[0], sample.replace('_gen', '_1'))

                gen_image = Image.open(gen_image_path)
                target_image = Image.open(target_image_path)
                gen_image = F.pil_to_tensor(gen_image)  # The range of values is 0~255, required by fid
                target_image = F.pil_to_tensor(target_image)  # The range of values is 0~255, required by fid

                gen_images.append(gen_image)
                target_images.append(target_image)

    fid.update(torch.stack(gen_images, dim=0), real=False)
    fid.update(torch.stack(target_images, dim=0), real=True)

    score = fid.compute()
    score = score.numpy().tolist()

    return score


def compute_clip_score(image_path, gen_path, instruct_path, group_path):
    clip_similarity = ClipSimilarity().cuda()
    count = 0
    avg_sim_0, avg_sim_1, avg_sim_direction, avg_sim_image = 0, 0, 0, 0
    avg_sim_exemplar = 0

    with open(group_path, 'r') as infile:
        groups = json.load(infile)

    for item in tqdm(os.listdir(instruct_path), desc='Clip Score'):
        lines = read_jsonl(os.path.join(instruct_path, item))

        for line in tqdm(lines, leave=False):
            instruction = line['instruction']
            caption_before = line['caption_before']
            caption_after = line['caption_after']
            source_image_path = os.path.join(image_path, line['source_image'])
            # For generated images --------------------------------------------------------------------------------------------------------------------------------------
            gen_image_path = os.path.join(gen_path, item.split('.')[0], f'{line["source_image"].split("/")[-2]}_{instruction.replace(" ", "_").replace(".", "")}', line['source_image'].split('/')[-1].replace('_0', '_gen'))
            # [or] For ground truth -------------------------------------------------------------------------------------------------------------------------------------
            # gen_image_path = os.path.join(gen_path, item.split('.')[0], f'{line["source_image"].split("/")[-2]}_{instruction.replace(" ", "_").replace(".", "")}', line['source_image'].split('/')[-1].replace('_0', '_1'))
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------

            group_id = line['source_image'].split('/')[0]
            candidates = groups[group_id]
            image_id = line['source_image'].split('/')[1].split('_')[0]
            idx = candidates.index(image_id)
            support_image_idx = idx + 1 if idx+1 < len(candidates) else 0  # use the next image as support image
            support_source_image_path = os.path.join(image_path, group_id, f'{candidates[support_image_idx]}_0.jpg')
            support_target_image_path = os.path.join(image_path, group_id, f'{candidates[support_image_idx]}_1.jpg')

            source_image = Image.open(source_image_path)
            gen_image = Image.open(gen_image_path)
            support_source_image = Image.open(support_source_image_path)
            support_target_image = Image.open(support_target_image_path)
            source_image = F.to_tensor(source_image).float().cuda()  # the range of values are 0~1
            gen_image = F.to_tensor(gen_image).float().cuda()  # the range of values are 0~1
            support_source_image = F.to_tensor(support_source_image).float().cuda()
            support_target_image = F.to_tensor(support_target_image).float().cuda()
            sim_0, sim_1, sim_direction, sim_image = clip_similarity(image_0=source_image[None], image_1=gen_image[None], text_0=[caption_before], text_1=[caption_after])
            sim_exemplar = clip_similarity.exemplar_direction_similarity(exemplar_image_0=support_source_image[None], exemplar_image_1=support_target_image[None], image_0=source_image[None], image_1=gen_image[None])
            avg_sim_0 += sim_0.cpu().numpy().tolist()[0]
            avg_sim_1 += sim_1.cpu().numpy().tolist()[0]
            avg_sim_direction += sim_direction.cpu().numpy().tolist()[0]
            avg_sim_image += sim_image.cpu().numpy().tolist()[0]
            avg_sim_exemplar += sim_exemplar.cpu().numpy().tolist()[0]
            count += 1
    
    avg_sim_0 /= count
    avg_sim_1 /= count
    avg_sim_direction /= count
    avg_sim_image /= count
    avg_sim_exemplar /= count

    return avg_sim_0, avg_sim_1, avg_sim_direction, avg_sim_image, avg_sim_exemplar


def compute_psnr(image_path, gen_path):
    psnr = PeakSignalNoiseRatio(data_range=1.0)

    score_sum, item_count = 0, 0
    for category in tqdm(os.listdir(gen_path), desc='PSNR'):
        if ".json" in category:
            continue
        for group in tqdm(os.listdir(os.path.join(gen_path, category)), leave=False):
            for sample in os.listdir(os.path.join(gen_path, category, group)):
                if '_gen' not in sample:
                    continue
                gen_image_path = os.path.join(gen_path, category, group, sample)
                target_image_path = os.path.join(image_path, group.split('_')[0], sample.replace('_gen', '_1'))
                gen_image = Image.open(gen_image_path)
                target_image = Image.open(target_image_path)
                gen_image = F.to_tensor(gen_image.resize((target_image.size), Image.Resampling.LANCZOS))  # values in range of [0, 1]
                target_image = F.to_tensor(target_image)  # values in range of [0, 1]
                score = psnr(target_image[None], gen_image[None])
                score_sum += float(score.numpy())
                item_count += 1
    
    return score_sum / item_count


def compute_lpips(image_path, gen_path):
    lpips_squeeze = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', reduction='mean').cuda()
    lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean').cuda()
    lpips_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='mean').cuda()

    score_squeeze_sum, score_vgg_sum, score_alex_sum, item_count = 0, 0, 0, 0
    for category in tqdm(os.listdir(gen_path), desc='LPIPS'):
        if ".json" in category:
            continue
        for group in tqdm(os.listdir(os.path.join(gen_path, category)), leave=False):
            for sample in os.listdir(os.path.join(gen_path, category, group)):
                if '_gen' not in sample:
                    continue
                gen_image_path = os.path.join(gen_path, category, group, sample)
                target_image_path = os.path.join(image_path, group.split('_')[0], sample.replace('_gen', '_1'))
                gen_image = Image.open(gen_image_path)
                target_image = Image.open(target_image_path)
                gen_image = F.to_tensor(gen_image.resize((target_image.size), Image.Resampling.LANCZOS))  # values in range of [0, 1]
                target_image = F.to_tensor(target_image)  # values in range of [0, 1]
                gen_image = gen_image.cuda()
                target_image = target_image.cuda()
                score_squeeze = lpips_squeeze(target_image[None], gen_image[None])
                score_vgg = lpips_vgg(target_image[None], gen_image[None])
                score_alex = lpips_alex(target_image[None], gen_image[None])
                score_squeeze_sum += float(score_squeeze.detach().cpu().numpy())
                score_vgg_sum += float(score_vgg.detach().cpu().numpy())
                score_alex_sum += float(score_alex.detach().cpu().numpy())
                item_count += 1
    
    return score_squeeze_sum / item_count, score_vgg_sum / item_count, score_alex_sum / item_count


def compute_dino(image_path, gen_path):
    # Load the DINO model and feature extractor
    model_name = "facebook/dino-vits16"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda()

    dino_sum, item_count = 0, 0

    for category in tqdm(os.listdir(gen_path), desc='DINO'):
        if ".json" in category:
            continue
        for group in tqdm(os.listdir(os.path.join(gen_path, category)), leave=False):
            for sample in os.listdir(os.path.join(gen_path, category, group)):
                if '_gen' not in sample:
                    continue
                gen_image_path = os.path.join(gen_path, category, group, sample)
                source_image_path = os.path.join(image_path, group.split('_')[0], sample.replace('_gen', '_0'))
                gen_image = Image.open(gen_image_path)
                source_image = Image.open(source_image_path)
                gen_image = F.to_tensor(gen_image)  # values in range of [0, 1]
                source_image = F.to_tensor(source_image)  # values in range of [0, 1]
                
                # Preprocess the image using the feature extractor
                gen_image = feature_extractor(images=gen_image, return_tensors="pt")
                source_image = feature_extractor(images=source_image, return_tensors="pt")
                for k in gen_image.keys():
                    gen_image[k] = gen_image[k].cuda()
                    source_image[k] = source_image[k].cuda()
                
                # Extract features using the DINO model
                with torch.no_grad():
                    gen_image_outputs = model(**gen_image)
                    source_image_outputs = model(**source_image)
                
                # Extract the CLS token output from the last hidden state (usually index 0)
                # This is a common practice for Vision Transformers (ViTs)
                gen_image_features = gen_image_outputs.last_hidden_state[:, 0, :]
                source_image_features = source_image_outputs.last_hidden_state[:, 0, :]
                
                # Normalize the features (L2 normalization is commonly used with cosine similarity)
                gen_image_features = gen_image_features / gen_image_features.norm(dim=1, keepdim=True)
                source_image_features = source_image_features / source_image_features.norm(dim=1, keepdim=True)

                # Compute cosine similarity between the two feature vectors
                cosine_similarity = torch.mm(gen_image_features, source_image_features.T)

                dino_sum += float(cosine_similarity.cpu()[0, 0])
                item_count += 1
    
    return dino_sum / item_count


def compute_l1(image_path, gen_path):
    score_sum, item_count = 0, 0
    for category in tqdm(os.listdir(gen_path), desc='L1'):
        if ".json" in category:
            continue
        for group in tqdm(os.listdir(os.path.join(gen_path, category)), leave=False):
            for sample in os.listdir(os.path.join(gen_path, category, group)):
                if '_gen' not in sample:
                    continue
                gen_image_path = os.path.join(gen_path, category, group, sample)
                target_image_path = os.path.join(image_path, group.split('_')[0], sample.replace('_gen', '_1'))
                gen_image = Image.open(gen_image_path)
                target_image = Image.open(target_image_path)
                gen_image = F.to_tensor(gen_image.resize((target_image.size), Image.Resampling.LANCZOS))  # values in range of [0, 1]
                target_image = F.to_tensor(target_image)  # values in range of [0, 1]
                score = torch.abs(gen_image - target_image).mean()
                score_sum += float(score.numpy())
                item_count += 1
    
    return score_sum / item_count


def compute_metrics(image_path, gen_path, instruct_path, group_path, cmpt_fid=True, cmpt_psnr=True, cmpt_lpips=True, cmpt_clip=True, cmpt_dino=True, cmpt_l1=True, save_res=True):    
    content = dict()

    if cmpt_clip:
        print('Computing CLIP...')
        sim_0, sim_1, sim_direction, sim_image, sim_exemplar = compute_clip_score(image_path, gen_path, instruct_path, group_path)
        print('clip_t2i_source:', sim_0)
        print('clip_t2i_target:', sim_1)
        print('clip_direction:', sim_direction)
        print('clip_image:', sim_image)
        print('clip_exemplar_alignment', sim_exemplar)
        content['clip_t2i_source'] = sim_0
        content['clip_t2i_target'] = sim_1
        content['clip_direction'] = sim_direction
        content['clip_image'] = sim_image
        content['clip_exemplar_alignment'] = sim_exemplar

    if cmpt_fid:
        print('Computing FID...')
        fid = compute_fid(image_path, gen_path)
        print('\nFID:', fid)
        content['fid'] = fid

    if cmpt_l1:
        print("Computing L1...")
        l1 = compute_l1(image_path, gen_path)
        print("\nL1:", l1)
        content["l1"] = l1

    if cmpt_psnr:
        print('Computing PSNR')
        psnr = compute_psnr(image_path, gen_path)
        print('\nPSNR:', psnr)
        content['psnr'] = psnr

    if cmpt_lpips:
        print('Computing LPIPS')
        lpips_squeeze, lpips_vgg, lpips_alex = compute_lpips(image_path, gen_path)
        print('\nLPIPS (SENet):', lpips_squeeze)
        print('LPIPS (VGG):', lpips_vgg)
        print('LPIPS (ALEX):', lpips_alex)
        content['lpips_squeeze'] = lpips_squeeze
        content['lpips_vgg'] = lpips_vgg
        content['lpips_alex'] = lpips_alex

    if cmpt_dino:
        print("Computing DINO...")
        dino = compute_dino(image_path, gen_path)
        print("\nDINO:", dino)
        content["dino"] = dino

    if save_res:
        save_path = os.path.join(gen_path, 'metric.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            with open(save_path, 'r') as infile:
                save = json.load(infile)
        else:
            save = dict()
        save.update(content)
        with open(save_path, 'w') as outfile:
            json.dump(save, outfile, indent=4)


def compute_clip_on_single_pair(exemplar_source_path, exemplar_target_path, image1_path, image2_path, caption1, caption2):
    clip_similarity = ClipSimilarity().cuda()
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    image1 = F.to_tensor(image1).float().cuda()  # the range of values are 0~1
    image2 = F.to_tensor(image2).float().cuda()  # the range of values are 0~1
    exemplar_source_image = Image.open(exemplar_source_path)
    exemplar_target_image = Image.open(exemplar_target_path)
    exemplar_source_image = F.to_tensor(exemplar_source_image).float().cuda()  # the range of values are 0~1
    exemplar_target_image = F.to_tensor(exemplar_target_image).float().cuda()  # the range of values are 0~1
    # instruction = "Make the scene look like a painting by Van Gogh"
    sim_0, sim_1, sim_direction, sim_image = clip_similarity(image_0=image1[None], image_1=image2[None], text_0=[caption1], text_1=[caption2])
    sim_exemplar = clip_similarity.exemplar_direction_similarity(exemplar_image_0=exemplar_source_image[None], exemplar_image_1=exemplar_target_image[None], image_0=image1[None], image_1=image2[None])
    print(sim_0, sim_1, sim_direction, sim_image, sim_exemplar)


if __name__ == '__main__':
    start = time.time()

    parser = ArgumentParser()
    parser.add_argument("--gen_path", type=str, help="Path to generated images, e.g. /data/home/bolinlai/Projects/FreeIm2Im_Dev/train_output/edit_ip2p_s1/checkpoint-02000/inference-02000")
    args = parser.parse_args()

    # image_path = '/fsx-project/bolinlai/Datasets/ip2p'
    image_path = '/scratch/users/bolinlai/Datasets/ip2p'
    # instruct_path = "/data/home/bolinlai/Projects/DataPreprocess/hold_out_eval/split_2/hold_out_test"
    instruct_path = "/home/bolinlai/Projects/DataPreprocess/hold_out_eval/split_2/hold_out_test"
    # group_path = '/data/home/bolinlai/Projects/DataPreprocess/ip2p_group_low_level.json'
    group_path = "/home/bolinlai/Projects/DataPreprocess/ip2p_group_low_level.json"

    compute_metrics(
        image_path=image_path,
        gen_path=args.gen_path,
        instruct_path=instruct_path,
        group_path=group_path,
        cmpt_clip=True,
        cmpt_l1=True,
        cmpt_psnr=True,
        cmpt_lpips=True,
        cmpt_fid=True,
        cmpt_dino=True,
        save_res=True
    )

    # compute_clip_on_single_pair(
    #     # exemplar_source_path=os.path.join(image_path, "0245161/1514571971_0.jpg"),
    #     # exemplar_target_path=os.path.join(image_path, "0245161/1514571971_1.jpg"),
    #     # image1_path=os.path.join(image_path, "0245161/1444327862_0.jpg"),
    #     # # image2_path="/fsx-project/bolinlai/Models/Ip2p/train_base/results/epoch=000000-step=000000999/e=0_st=5.0_si=2.0/van_gogh/0245161_make_the_scene_look_like_a_painting_by_Van_Gogh/1444327862_gen.jpg",
    #     # # image2_path="/fsx-project/bolinlai/Models/FreeIm2Im/checkpoints/icl_edit_embed30/checkpoint-14000/inference-14000/van_gogh/0245161_make_the_scene_look_like_a_painting_by_Van_Gogh/1444327862_gen.jpg",
    #     # image2_path="/fsx-project/bolinlai/Models/PromptDiffusion/released/network-step=04999-9.0-1/van_gogh/0245161_make_the_scene_look_like_a_painting_by_Van_Gogh/1444327862_gen.jpg",
    #     # caption1="Temple Crag Best Places to Camp in California",
    #     # caption2="Temple Crag Best Places to Camp in California, Painting by Vincent van Gogh"

    #     exemplar_source_path=os.path.join(image_path, "0224299/3491663431_0.jpg"),
    #     exemplar_target_path=os.path.join(image_path, "0224299/3491663431_1.jpg"),
    #     image1_path=os.path.join(image_path, "0224299/3014702808_0.jpg"),
    #     # image2_path="/fsx-project/bolinlai/Models/Ip2p/train_base/results/epoch=000000-step=000000999/e=0_st=5.0_si=2.0/everest/0224299_move_the_Milky_Way_to_the_top_of_Mt_Everest/3014702808_gen.jpg",
    #     # image2_path="/fsx-project/bolinlai/Models/PromptDiffusion/released/network-step=04999-9.0-1/everest/0224299_move_the_Milky_Way_to_the_top_of_Mt_Everest/3014702808_gen.jpg",
    #     # image2_path="/fsx-project/bolinlai/Models/FreeIm2Im/checkpoints/icl_edit_embed30/checkpoint-14000/inference-14000/everest/0224299_move_the_Milky_Way_to_the_top_of_Mt_Everest/3014702808_gen.jpg",
    #     image2_path=os.path.join(image_path, "0224299/3014702808_1.jpg"),
    #     caption1="Latidude 62.4 by GiulioCobianchiPhoto - Capture The Milky Way Photo Contest",
    #     caption2="Latidude 62.4 by GiulioCobianchiPhoto - Capture The Milky Way Atop Mt. Everest Photo Contest"
    # )

    print(time.time() - start)
