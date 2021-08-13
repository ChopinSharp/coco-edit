import torch
import clip
from PIL import Image
import json
import tqdm
import os
from utils.config import CUDA_DEVICE, COCO_CAPTION_PATH, COCO_IMAGE_PATH


def shard_matmul(A, B, shard_size=5000):
    A_len, B_len = A.shape[0], B.shape[0]
    B = B.T
    A_num = A_len // shard_size + (1 if A_len % shard_size != 0 else 0)
    B_num = B_len // shard_size + (1 if B_len % shard_size != 0 else 0)
    result = torch.zeros(A_len, B_len, dtype=torch.float)
    pbar = tqdm.tqdm(total=A_num*B_num, desc='shard matmul')
    for i in range(A_num):
        A_shard = A[i * shard_size: (i+1) * shard_size].to(device=CUDA_DEVICE)
        for j in range(B_num):
            B_shard = B[:, j * shard_size: (j+1) * shard_size].to(device=CUDA_DEVICE)
            shard_result = (A_shard @ B_shard).to(device='cpu')
            result[i * shard_size: (i+1) * shard_size, j * shard_size: (j+1) * shard_size] = shard_result
            pbar.update(1)
    pbar.close()
    return result


def main():
    im_bz = 512
    cap_bz = 1024

    model, preprocess = clip.load("ViT-B/32", device=CUDA_DEVICE)

    print('loading captions...')
    with open(COCO_CAPTION_PATH) as f:
        coco_caps = json.load(f)

    for split in ['val']:  # , 'test', 'train']:
        print('# SPLIT:', split)
        imgid_list = [img['imgid'] for img in coco_caps['images'] if img['split'] == split]
        if split == 'train':
            assert len(imgid_list) == 82783
        else:
            assert len(imgid_list) == 5000
        image_folder = os.path.join(COCO_IMAGE_PATH, 'train2014' if split == 'train' else 'val2014')
        image_files = [os.path.join(image_folder, coco_caps['images'][imgid]['filename']) for imgid in imgid_list]
        captions = [sent['raw'] for imgid in imgid_list for sent in coco_caps['images'][imgid]['sentences'][:5]]
        img_num, cap_num = len(image_files), len(captions)
        print('number of images: {}, number of captions: {}'.format(img_num, cap_num))
        assert len(image_files) * 5 == len(captions)

        image_list = []
        for image_path in tqdm.tqdm(image_files, desc='preprocessing images', ascii=True):
            image_list.append(preprocess(Image.open(image_path)).unsqueeze(0))
        image = torch.cat(image_list, dim=0)

        print('preprocessing captions ...')
        text = clip.tokenize(captions)

        with torch.no_grad():
            image_feat_list = []
            im_bn = img_num // im_bz + (1 if img_num % im_bz != 0 else 0)
            for i in tqdm.trange(im_bn, desc='encoding image', ascii=True):
                image_feat_list.append(model.encode_image(image[i*im_bz:(i+1)*im_bz].to(CUDA_DEVICE)).cpu())
            image_features = torch.cat(image_feat_list, dim=0)
            
            text_feat_list = []
            cap_bn = cap_num // cap_bz + (1 if cap_num % cap_bz != 0 else 0)
            for i in tqdm.trange(cap_bn, desc='encoding text', ascii=True):
                text_feat_list.append(model.encode_text(text[i*cap_bz:(i+1)*cap_bz].to(CUDA_DEVICE)).cpu())
            text_features = torch.cat(text_feat_list, dim=0)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            print('caculating similarities ...')
            sims = shard_matmul(image_features, text_features)

        sims = torch.sigmoid(sims)
        for img_i in range(sims.shape[0]):
            sims[img_i][5*img_i:5*(img_i+1)] = 0.

        print('saving similarities ...')
        torch.save(sims, 'cache/img_cap_sim_{}.pt'.format(split))

        with open('cache/imgid_list_{}.json'.format(split), 'w') as f:
            json.dump(imgid_list, f)


main()
