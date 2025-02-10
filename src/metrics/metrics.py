import os
import time
import json
import torchvision.transforms.functional as F

from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from clip_similarity import ClipSimilarity


def read_jsonl(file_path):
    with open(file_path, 'r') as infile:
        data = [json.loads(line) for line in infile]
    return data


def compute_clip_score(image_path, gen_path, instruct_path, group_path):
    clip_similarity = ClipSimilarity().cuda()
    count = 0
    avg_sim_0, avg_sim_1, avg_sim_direction, avg_sim_image, avg_sim_exemplar_alignment = 0, 0, 0, 0, 0

    with open(group_path, 'r') as infile:
        groups = json.load(infile)

    for item in tqdm(os.listdir(instruct_path), desc='Clip Score'):
        lines = read_jsonl(os.path.join(instruct_path, item))

        for line in tqdm(lines, leave=False):
            instruction = line['instruction']
            caption_before = line['caption_before']
            caption_after = line['caption_after']
            source_image_path = os.path.join(image_path, line['source_image'])
            gen_image_path = os.path.join(gen_path, item.split('.')[0], f'{line["source_image"].split("/")[-2]}_{instruction.replace(" ", "_").replace(".", "")}', line['source_image'].split('/')[-1].replace('_0', '_gen'))

            group_id = line['source_image'].split('/')[0]
            candidates = groups[group_id]
            image_id = line['source_image'].split('/')[1].split('_')[0]
            idx = candidates.index(image_id)
            exemplar_image_idx = idx + 1 if idx+1 < len(candidates) else 0  # use the next image as exemplar image by default
            exemplar_source_image_path = os.path.join(image_path, group_id, f'{candidates[exemplar_image_idx]}_0.jpg')
            exemplar_target_image_path = os.path.join(image_path, group_id, f'{candidates[exemplar_image_idx]}_1.jpg')

            source_image = Image.open(source_image_path)
            gen_image = Image.open(gen_image_path)
            exemplar_source_image = Image.open(exemplar_source_image_path)
            exemplar_target_image = Image.open(exemplar_target_image_path)
            source_image = F.to_tensor(source_image).float().cuda()
            gen_image = F.to_tensor(gen_image).float().cuda()
            exemplar_source_image = F.to_tensor(exemplar_source_image).float().cuda()
            exemplar_target_image = F.to_tensor(exemplar_target_image).float().cuda()
            sim_0, sim_1, sim_direction, sim_image = clip_similarity(image_0=source_image[None], image_1=gen_image[None], text_0=[caption_before], text_1=[caption_after])
            sim_exemplar = clip_similarity.exemplar_direction_similarity(exemplar_image_0=exemplar_source_image[None], exemplar_image_1=exemplar_target_image[None], image_0=source_image[None], image_1=gen_image[None])
            avg_sim_0 += sim_0.cpu().numpy().tolist()[0]
            avg_sim_1 += sim_1.cpu().numpy().tolist()[0]
            avg_sim_direction += sim_direction.cpu().numpy().tolist()[0]
            avg_sim_image += sim_image.cpu().numpy().tolist()[0]
            avg_sim_exemplar_alignment += sim_exemplar.cpu().numpy().tolist()[0]
            count += 1

    avg_sim_0 /= count
    avg_sim_1 /= count
    avg_sim_direction /= count
    avg_sim_image /= count
    avg_sim_exemplar_alignment /= count

    return avg_sim_0, avg_sim_1, avg_sim_direction, avg_sim_image, avg_sim_exemplar_alignment


def compute_metrics(image_path, gen_path, instruct_path, group_path, save_res=True):
    content = dict()

    print('Computing CLIP...')
    sim_0, sim_1, sim_direction, sim_image, sim_exemplar = compute_clip_score(image_path, gen_path, instruct_path, group_path)
    print('clip_t2i_source:', sim_0)
    print('clip_t2i_target:', sim_1)  # CLIP-T
    print('clip_direction:', sim_direction)  # CLIP-Dir
    print('clip_image:', sim_image)  # CLIP-I
    print('clip_exemplar_alignment', sim_exemplar)  # CLIP-Vis
    content['clip_t2i_source'] = sim_0
    content['clip_t2i_target'] = sim_1
    content['clip_direction'] = sim_direction
    content['clip_image'] = sim_image
    content['clip_exemplar_alignment'] = sim_exemplar

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


if __name__ == '__main__':
    start = time.time()

    parser = ArgumentParser()
    parser.add_argument("--gen_path", type=str, help="Path to generated images, e.g. /your_path_to/checkpoint-xxxx/inference-xxxx-in-dist")
    args = parser.parse_args()

    image_path = './data/ip2p'
    instruct_path = "./data/eval"
    group_path = "./data/ip2p_group_instruct.json"

    compute_metrics(
        image_path=image_path,
        gen_path=args.gen_path,
        instruct_path=instruct_path,
        group_path=group_path,
        save_res=True
    )

    print(time.time() - start)
