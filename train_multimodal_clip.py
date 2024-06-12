import warnings
warnings.filterwarnings('ignore')
import torch
from mmpt.utils import load_config, set_seed
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from viclip import ViCLIP
from laion_clap import CLAP_Module
from torch.utils.data import DataLoader, Dataset
from clip import clip
import os
import pickle
import re
import json
import argparse
import librosa
import time
import sys
sys.path.append('./models')
from tqdm import tqdm
import open_clip
import math
import argparse
from models.clip import clip
from transformers import AutoTokenizer
from config_loader import load_config_from_path
from torch.utils.data.dataloader import default_collate
from collections import OrderedDict, defaultdict
from models import DenseCLIP_multimodal_clip
from dataset.data_helper import *
from dataset.multimodal_dataset import get_audio_features, multimodal_dataset_train, video_k400_distill_test, \
                                        image_coco_distill_test, image_voc_distill_test, audio_esc50_distill_test, \
                                        video_k600_distill_test, video_k700_distill_test, image_nuswide_distill_test, \
                                        image_mini_distill_test, image_obj_distill_test, image_voc2012_distill_test

def to_ctx(data, ctx=0, dtype=None):
    if isinstance(data, dict):
        for key in data:
            if torch.is_tensor(data[key]):
                if dtype is not None and data[key].dtype == torch.float32:
                    data[key] = data[key].to(dtype)
                data[key] = data[key].to(ctx)
    else:
        data = data.to(ctx)

    return data

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def load_video_clip_to_cpu(model_path):

    model = ViCLIP(size='b')

    # print(f"Load video_clip pretrained weights from {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')['model']
    msg = model.load_state_dict(state_dict, strict=False)
    # print(msg)

    return model

def load_audio_clip_to_cpu(model_path):

    # print(f"Load audio_clip pretrained weights from {model_path}")
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt(ckpt=model_path, verbose=False)
    # print("matching")

    return model

def load_image_clip_to_cpu(model_path):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=model_path)
    return model

def average_precision(output, target):

    epsilon = 1e-8

    indices = output.argsort()[::-1]
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    if np.size(preds) == 0:
        print('Size is 0!')
        return 0
    ap = np.zeros((preds.shape[1]))

    for k in range(preds.shape[1]):
        scores = preds[:, k]
        targets = targs[:, k]
        ap[k] = average_precision(scores, targets)
        
    return 100 * ap.mean()

def acc(targs, preds, cates):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    if np.size(preds) == 0:
        print('Size is 0!')
        return 0

    cnt_1 = 0
    cnt_5 = 0
    cnt_10 = 0

    cls2acc = {}

    for c in cates:
        cls2acc[c] = [0,0,0,0]

    for k in range(preds.shape[0]):
        pred,indices = torch.topk(preds[k],10)
        target = targs[k]
        target_name = cates[target]
        if target in indices[:1]:
            cnt_1 += 1
            cls2acc[target_name][1]+=1
        if target in indices[:5]:
            cnt_5 += 1
            cls2acc[target_name][2]+=1
        if target in indices:
            cnt_10 += 1
            cls2acc[target_name][3]+=1
        cls2acc[target_name][0]+=1

    return [100 * (cnt_1/preds.shape[0]),100 * (cnt_5/preds.shape[0]),100 * (cnt_10/preds.shape[0])]

def collate_fn_ESC50(batch):
    audio_file_list = [item[0] for item in batch]
    one_hot_list = [item[1][None,:] for item in batch]
    one_hot_list = torch.cat(one_hot_list)
        
    audio_input = []
    for f in audio_file_list:
        audio_waveform, _ = librosa.load(f, sr=48000)           
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        temp_dict = {}
        temp_dict = get_audio_features(
            temp_dict, audio_waveform, 480000, 
            data_truncating='fusion' if audio_clip_model.enable_fusion else 'rand_trunc', 
            data_filling='repeatpad',
            audio_cfg=audio_clip_model.model_cfg['audio_cfg'],
            require_grad=audio_waveform.requires_grad
        )
        audio_input.append(temp_dict)
        
    input_dict = {}
    keys = audio_input[0].keys()
    for k in keys:
        input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in audio_input], dim=0)
        
    return {'input_dict':input_dict, 'one_hot' : one_hot_list}

def collate_fn_multimodal(batch):

    video_input_ids = [item['video'][0] for item in batch if item['modal_type'] == 'video']
    image_input_ids = [item['image'][0] for item in batch if item['modal_type'] == 'image']

    audio_input_ids = [torch.tensor(item['audio']['input_ids']) for item in batch if item['modal_type'] == 'audio']
    audio_attention_mask = [torch.tensor(item['audio']['attention_mask']) for item in batch if item['modal_type'] == 'audio']

    video_labels = torch.stack([item['label'] for item in batch if item['modal_type'] == 'video'])
    image_labels = torch.stack([item['label'] for item in batch if item['modal_type'] == 'image'])
    audio_labels = torch.stack([item['label'] for item in batch if item['modal_type'] == 'audio'])

    max_length = max(len(ids) for ids in audio_input_ids)
    
    audio_input_ids = audio_input_ids = [torch.nn.functional.pad(ids.clone(), (0, max_length - len(ids))) for ids in audio_input_ids]
    audio_attention_mask = [torch.nn.functional.pad(mask.clone(), (0, max_length - len(mask))) for mask in audio_attention_mask]

    return {
        'video_input_ids': torch.stack(video_input_ids),
        'image_input_ids': torch.stack(image_input_ids),
        'audio_input_ids': torch.stack(audio_input_ids),
        'audio_attention_mask': torch.stack(audio_attention_mask),
        'video_labels': video_labels,
        'image_labels': image_labels,
        'audio_labels': audio_labels,
        }

def video_test_model(model, test_loader, st, end, device, verbose, video):

    model.eval()
    pred_list = []
    pred_aux_list = []
    ground_truth_list = []

    if video in ['mini']:
        is_video = False
        is_image = True
    else:
        is_video = True
        is_image = False

    iid2fea = []

    if verbose:
        progression_bar = tqdm(test_loader, leave=True, ncols=80, desc=f"testing video {video}")
    else:
        progression_bar = test_loader

    for img,target in progression_bar:

        ground_truth_list.extend(target.tolist())
        img = img.to(device, non_blocking=True).float()

        with torch.no_grad():
            output, video_fea = model(test_input = img, is_video=is_video, is_image=is_image, if_test = True)
            output = output[:,st:end]
            pred_list.append(output.detach().cpu())
            iid2fea.append(video_fea.detach().cpu())

    pred = torch.cat(pred_list)
    ground_truth = np.array(ground_truth_list)
    iid2fea = torch.cat(iid2fea)

    return acc(ground_truth, pred, test_loader.dataset.clsname), np.array(ground_truth_list), torch.cat(pred_list)

def audio_test_model(model, test_loader, st, end, device, verbose, audio):

    model = model.to(device)
    model.eval()

    iid2fea = []
    ground_truth = []
    
    if verbose:
        progression_bar = tqdm(test_loader, leave=True, ncols=80, desc=f"testing {audio}")
    else:
        progression_bar = test_loader
        
    total = 0
    correct = 0
    for test_batch in progression_bar:
        total += len(test_batch['one_hot'])
        with torch.no_grad():
            output, audio_fea = model(test_input = test_batch['input_dict'], is_audio=True, if_test=True)
            output = output[:,st:end]
            predict = torch.argmax(output.detach().cpu(), dim=-1)
            iid2fea.append(audio_fea.detach().cpu())
            ground_truth.append(torch.argmax(test_batch['one_hot'],dim=1))

            correct += int(test_batch['one_hot'][torch.arange(test_batch['one_hot'].shape[0]), predict].sum())

    iid2fea = torch.cat(iid2fea)
    ground_truth = torch.cat(ground_truth)

    return 1.0 * correct / total

def image_test_model(model, test_loader, st, end, device, verbose, image):

    model.eval()
    total_num = 0
    pred_list = []
    pred_aux_list = []
    ground_truth_list = []
    test_loss = 0

    iid2fea = []

    if verbose:
        progression_bar = tqdm(test_loader, leave=True, ncols=80, desc=f"testing {image}")
    else:
        progression_bar = test_loader

    for img, wb_cap in progression_bar:
        total_num += img.shape[0]
        ground_truth_list.append(wb_cap[:,0])
        with torch.no_grad():
            output, image_fea = model(test_input = img.to(device), is_image=True, if_test = True)
            output = output[:,st:end]
            pred_list.append(output.detach().cpu())
            iid2fea.append(image_fea.detach().cpu())

    pred = torch.cat(pred_list).numpy()
    iid2fea = torch.cat(iid2fea)
    ground_truth = torch.cat(ground_truth_list).numpy()

    save_path = f"output/obj_fea.json"
    json.dump([iid2fea.tolist(), ground_truth.tolist()], open(save_path,'w'))

    return mAP(ground_truth, pred), torch.cat(ground_truth_list), torch.cat(pred_list)

def train(config, cls_num, model, train_loader, test_loader_01=None, test_loader_02=None, test_loader_03=None, max_epoch = 10, device = 'cuda'):
    model = model.to(device)
    optimizer = torch.optim.SGD(
        [
            {'params' : model.video_feature_learner.parameters(), 'lr': config.video_lr},
            {'params' : model.image_feature_learner.parameters(), 'lr': config.image_lr},
            {'params' : model.audio_feature_learner.parameters(), 'lr': config.audio_lr},
            ],
    )
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(max_epoch)
            )
    start_time = time.time()

    total_step = 0

    save_path = f"output/modal_prompt_k400_obj_esc50/"
    os.makedirs(save_path, exist_ok=True)

    best_image_map = 0
    best_video_acc = 0
    for epoch in range(max_epoch):
        print('='*40)
        if config.verbose:
            progression_bar = tqdm(train_loader, leave=True, ncols=160, ascii=True)
        else:
            progression_bar = train_loader
        for batch in progression_bar:

            total_step += 1
            if total_step % 10 == 1:
                video_prompt = model.video_feature_learner.text_features.tolist()
                image_prompt = model.video_feature_learner.text_features.tolist()
                audio_prompt = model.video_feature_learner.text_features.tolist()

                json.dump([video_prompt,image_prompt,audio_prompt],open(save_path + f"step_{total_step}.json",'w'))

            batch = to_ctx(batch,ctx=device)

            loss_ranking_i, loss_ranking_v, loss_ranking_a, loss_cont = model(**batch, if_test=False)
            loss = (loss_ranking_i + loss_ranking_v + loss_ranking_a) + loss_cont
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if config.verbose:
                progression_bar.set_description(f"{loss.item()}, {loss_ranking_i.item()}, {loss_ranking_v.item()}, {loss_ranking_a.item()}, {loss_cont.item()}")

        if config.use_k400:
            video_k400_test_mAP, ground_truth_list, pred_list = video_test_model(model, test_loader_01, st=0, end=cls_num[0], device=device, verbose=config.verbose, video='k400')
            print(f'test video_k400: {video_k400_test_mAP}')

        if config.use_k600:
            video_k600_test_mAP, ground_truth_list, pred_list = video_test_model(model, test_loader_01, st=0, end=cls_num[0], device=device, verbose=config.verbose, video='k600')
            print(f'test video_k600: {video_k600_test_mAP}')

        if config.use_k700:
            video_k700_test_mAP, ground_truth_list, pred_list = video_test_model(model, test_loader_01, st=0, end=cls_num[0], device=device, verbose=config.verbose, video='k700')
            print(f'test video_k700: {video_k700_test_mAP}')

        if config.use_coco:
            image_coco_test_mAP, ground_truth_list, pred_list = image_test_model(model, test_loader_02, st=cls_num[0], end=cls_num[0] + cls_num[1], device=device, verbose=config.verbose, image='coco')
            print(f'test image_coco: {image_coco_test_mAP}')

        if config.use_voc:
            image_voc_test_mAP, ground_truth_list, pred_list = image_test_model(model, test_loader_02, st=cls_num[0], end=cls_num[0] + cls_num[1], device=device, verbose=config.verbose, image='voc')
            print(f'test image_voc: {image_voc_test_mAP}')

        if config.use_voc2012:
            image_voc2012_test_mAP, ground_truth_list, pred_list = image_test_model(model, test_loader_02, st=cls_num[0], end=cls_num[0] + cls_num[1], device=device, verbose=config.verbose, image='voc2012')
            print(f'test image_voc2012: {image_voc2012_test_mAP}')

        if config.use_nuswide:
            image_nuswide_test_mAP, ground_truth_list, pred_list = image_test_model(model, test_loader_02, st=cls_num[0], end=cls_num[0] + cls_num[1], device=device, verbose=config.verbose, image='nuswide')
            print(f'test image_nuswide: {image_nuswide_test_mAP}')

        if config.use_obj:
            image_obj_test_mAP, ground_truth_list, pred_list = image_test_model(model, test_loader_02, st=cls_num[0], end=cls_num[0] + cls_num[1], device=device, verbose=config.verbose, image='obj')
            print(f'test image_obj: {image_obj_test_mAP}')

        if config.use_mini:
            image_mini_test_mAP, ground_truth_list, pred_list = video_test_model(model, test_loader_02, st=cls_num[0], end=cls_num[0] + cls_num[1], device=device, verbose=config.verbose, video='mini')
            print(f'test image_mini: {image_mini_test_mAP}')

        if config.use_esc50:
            audio_esc50_test_mAP = audio_test_model(model, test_loader_03, st=cls_num[0] + cls_num[1], end=10000, device=device, verbose=config.verbose, audio='esc50')
            print(f'test audio_esc50: {audio_esc50_test_mAP}')

        if config.use_us8k:
            audio_us8k_test_mAP = audio_test_model(model, test_loader_03, st=cls_num[0] + cls_num[1], end=10000, device=device, verbose=config.verbose, audio='us8k')
            print(f'test audio_us8k: {audio_us8k_test_mAP}')

        # model.save_model('save_model/DenseCLIP_skip_te_modified/' + f'DenseCLIP_skip_te_modified_{max_epoch}_{epoch}.pt')

        used_time = time.time() - start_time
        print(f'Epoch[{epoch+1}/{max_epoch}], time:{used_time:.2f}s, eta:{used_time * (max_epoch - epoch -1)/(epoch+1):.2f}s')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_lr', type=float, default=5e-4)
    parser.add_argument('--image_lr', type=float, default=5e-2)
    parser.add_argument('--audio_lr', type=float, default=5e-2)
    parser.add_argument('--verbose', default=True)
    args = parser.parse_args()

    set_seed(202)

    config = load_config_from_path('./configs/Caption_distill_double_config_video_clip.json')

    cls_num = []

    if config.use_k400:
        cls_num.append(400)

    if config.use_k600:
        cls_num.append(600)

    if config.use_k700:
        cls_num.append(700)

    if config.use_coco:
        cls_num.append(80)

    if config.use_voc:
        cls_num.append(20)

    if config.use_voc2012:
        cls_num.append(20)

    if config.use_nuswide:
        cls_num.append(81)

    if config.use_mini:
        cls_num.append(100)

    if config.use_obj:
        cls_num.append(365)

    if config.use_esc50:
        cls_num.append(50)

    if config.use_us8k:
        cls_num.append(10)
    
    config.video_lr = args.video_lr
    config.image_lr = args.image_lr
    config.audio_lr = args.audio_lr
    config.verbose = args.verbose

    print('='*40)
    print(f"use_image_video_con: {config.use_image_video_con}")
    print(f"use_image_audio_con: {config.use_image_audio_con}")
    print(f"use_video_audio_con: {config.use_video_audio_con}\n")

    print(f"max_epoch: {config.epoch}\n")

    print(f"cache_file: {config.cache_file}\n")

    print(f"video_lr: {config.video_lr}")
    print(f"image_lr: {config.image_lr}")
    print(f"audio_lr: {config.audio_lr}\n")

    print(f"si_directional: {config.si_directional}")
    print('='*40)

    image_clip_model = load_image_clip_to_cpu(config.image_clip_ckpt_path)
    video_clip_model = load_video_clip_to_cpu(config.video_clip_ckpt_path)
    audio_clip_model = load_audio_clip_to_cpu(config.audio_clip_ckpt_path)

    model = DenseCLIP_multimodal_clip(config, cls_num, image_clip_model, video_clip_model, audio_clip_model)
    model.float()

    max_txt_l = video_clip_model.max_txt_l
    multimodal_tokenizer = {
        'video':video_clip_model.text_encoder.tokenize,
        'audio':audio_clip_model.tokenize,
        'image':open_clip.get_tokenizer('ViT-B-32')
    }

    multimodal_dataset_train = multimodal_dataset_train(cfg=config, cache_file = config.cache_file, clip_prompt = "a photo of {}", tokenizer=multimodal_tokenizer, max_txt_l=max_txt_l)
    multimodal_train_loader = DataLoader(multimodal_dataset_train, batch_size=768, shuffle=True, num_workers=8, collate_fn=collate_fn_multimodal)

    if config.use_k400:
        video_test_dataset = video_k400_distill_test(cfg=config, cache_file='cache/video_k400_val.pkl')
    if config.use_k600:
        video_test_dataset = video_k600_distill_test(cfg=config, cache_file='cache/video_k600_test.pkl')
    if config.use_k700:
        video_test_dataset = video_k700_distill_test(cfg=config, cache_file='cache/video_k700_test.pkl')

    if config.use_coco:
        image_test_dataset = image_coco_distill_test(cfg=config, cache_file='cache/image_coco_test.pkl')
    if config.use_voc:
        image_test_dataset = image_voc_distill_test(cfg=config, cache_file='cache/image_voc_test.pkl')
    if config.use_nuswide:
        image_test_dataset = image_nuswide_distill_test(cfg=config, cache_file='cache/image_nuswide_test.pkl')
    if config.use_mini:
        image_test_dataset = image_mini_distill_test(cfg=config, cache_file='cache/image_mini_test.pkl')
    if config.use_obj:
        image_test_dataset = image_obj_distill_test(cfg=config, cache_file='cache/image_obj_test.pkl')
    if config.use_voc2012:
        image_test_dataset = image_voc2012_distill_test(cfg=config, cache_file='cache/image_voc2012_test.pkl')

    if config.use_esc50:
        audio_test_dataset = audio_esc50_distill_test(cfg=config, cache_file='cache/audio_esc50_test.pkl')
    if config.use_us8k:
        audio_test_dataset = audio_us8k_distill_test(cfg=config, cache_file='cache/audio_us8k_test.pkl')

    video_test_loader = DataLoader(video_test_dataset, batch_size=16, shuffle=False, num_workers=4)
    image_test_loader = DataLoader(image_test_dataset, batch_size=16, shuffle=False, num_workers=4)
    audio_test_loader = DataLoader(audio_test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn_ESC50)

    train(config, cls_num, model, multimodal_train_loader, test_loader_01=video_test_loader, test_loader_02=image_test_loader, test_loader_03=audio_test_loader, max_epoch=config.epoch)