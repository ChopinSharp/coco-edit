import json
import random
import os
import sys
import torch
from multiprocessing import Pool
from tqdm import tqdm
import time
from argparse import ArgumentParser
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.spice.spice import Spice
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from utils.edit_misc import edit_distance
from utils.config import COCO_CAPTION_PATH


CLIP_TOP = 300
CLIP_RESAMPLE = 30
BLEU_RESMAPLE = 6
BLEU_2_LOW = 0.4
BLEU_3_LOW = 0.3
SPICE_HIGH = 0.35


def calculate_edit_distance(sent1, sent2):
    return edit_distance(sent1.split(), sent2.split())[-1][-1]


def tokenize(tokenizer, sentences):
    return tokenizer.tokenize({0: [{'caption': sent} for sent in sentences]})


def sample_caption(args):
    bleu = Bleu(3)
    spice = Spice(pid=os.getpid())
    tokenizer = PTBTokenizer()
    bleu_cand = []
    gts = tokenize(tokenizer, args['gt_list'])
    for sent in args['neg']:
        bleu_scores = bleu.compute_score(gts, tokenize(tokenizer, [sent]))[0]
        bleu_2, bleu_3 = bleu_scores[1], bleu_scores[2]
        if bleu_2 > BLEU_2_LOW and bleu_3 > BLEU_3_LOW:
            bleu_cand.append({
                'sent': sent,
                'bleu_2': bleu_2,
                'bleu_3': bleu_3
            })
            if len(bleu_cand) == BLEU_RESMAPLE:
                break
    if not bleu_cand:
        return None
    for cand in bleu_cand:
        cand['spice'] = spice.compute_score(gts, tokenize(tokenizer, [cand['sent']]))[1][0]['All']['pr']
    spice_cand = list(filter(lambda x: x['spice'] < SPICE_HIGH, bleu_cand))
    if not spice_cand:
        return None
    for cand in spice_cand:
        min_dist = 2333
        min_idx = None
        for gt_i, gt_sent in enumerate(args['gt_list']):
            this_dist = calculate_edit_distance(cand['sent'], gt_sent)
            if this_dist < min_dist:
                min_dist = this_dist
                min_idx = gt_i
        cand['dist'] = min_dist
        cand['ratio'] = min_dist / (len(cand['sent'].split()) + len(args['gt_list'][min_idx].split()))
        cand['gt_idx'] = min_idx
    final_cand = sorted(spice_cand, key=lambda x:x['ratio'])
    return final_cand


def main():
    parser = ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--split', type=str, required=True)
    args = parser.parse_args()

    t = time.time()

    # Load COCO Captions
    print('loading COCO ...')
    with open(COCO_CAPTION_PATH) as f:
        coco_captions = json.load(f)
    coco_images = coco_captions['images']

    print('loading CLIP similarities ...')
    img_cap = torch.load('cache/img_cap_sim_{}.pt'.format(args.split))
    with open('cache/imgid_list_{}.json'.format(args.split)) as f:
        imgid_list = json.load(f)
    
    start, end = args.start, args.end
    if end is None:
        end = len(imgid_list)
    print('sampling from {} to {} ...'.format(start, end))

    # Building samples
    N = end - start
    clip_sampled_data = []
    for img_idx in range(start, end):
        img_i = imgid_list[img_idx]
        img = coco_images[img_i]
        assert img_i == img['imgid']
        gt_list = [s['raw'].strip() for s in img['sentences']]
        clip_cand = random.sample(torch.argsort(img_cap[img_idx], descending=True)[:CLIP_TOP].tolist(), CLIP_RESAMPLE)
        clip_sampled_data.append({
            'img_i': img_i,
            'filename': img['filename'],
            'gt_list': gt_list,
            'neg': [coco_images[imgid_list[c // 5]]['sentences'][c % 5]['raw'].strip() for c in clip_cand]
        })

    neg_list = []
    with Pool(processes=30) as pool:
        with tqdm(total=N, ascii=True, desc='sampling', file=sys.stdout) as pbar:
            for item in pool.imap(sample_caption, clip_sampled_data, chunksize=1):
                neg_list.append(item)
                pbar.update(1)

    assert len(neg_list) == N

    print('time elapsed:', (time.time() - t) // 60, 'min')

    sampled_data = [{
        'img_i': sample['img_i'],
        'filename': sample['filename'],
        'gt': sample['gt_list'],
        'neg': neg_cand
    } for sample, neg_cand in zip(clip_sampled_data, neg_list) if neg_cand is not None]

    save_path = 'output/cocoedit_full_mp_{}_{}_{}.json'.format(args.split, start, end)
    print('saving results to', save_path)
    with open(save_path, 'w') as f:
        json.dump(sampled_data, f)


main()
