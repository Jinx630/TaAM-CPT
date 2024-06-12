import torch
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('../..')
import nltk
import pandas as pd
import torch.nn.functional as F
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
    RandomApply, GaussianBlur, RandomGrayscale, RandomResizedCrop,
    RandomHorizontalFlip, 
)
import torchvision.transforms as transforms
import cv2
import os
import sys
import pickle
from PIL import Image
from tqdm import tqdm
from typing import Any, Union, List
from PIL import Image, ImageDraw
import random
import math
import numpy as np
import random
import json
import torchaudio
from contextlib import suppress

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

from .data_helper import *

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std

def get_mel(audio_data, audio_cfg):
    # mel shape: (n_mels, T)
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=audio_cfg['mel_bins'],
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    ).to(audio_data.device)
    
    mel = mel_tf(audio_data)
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T, n_mels)

def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg, require_grad=False):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    require_grad: whether to require gradient for audio data.
        This is useful when we want to apply gradient-based classifier-guidance.
    """
    grad_fn = suppress if require_grad else torch.no_grad
    with grad_fn():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                # fusion
                mel = get_mel(audio_data, audio_cfg)
                # split to three parts
                chunk_frames = max_len // audio_cfg['hop_size'] + 1  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is
                    # larger than max_len but smaller than max_len+hop_size.
                    # In this case, we just use the whole audio.
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                    #       'len(audio_data):', len(audio_data),
                    #       'chunk_frames:', chunk_frames,
                    #       'total_frames:', total_frames)
                    if len(ranges[1]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[2] = [0]
                    # randomly choose index for each part
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    # select mel
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                    # shrink the mel
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, audio_cfg['mel_bins']])(mel[None])[0]
                    # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                    # stack
                    mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            # random crop to max_len (for compatibility)
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx: idx + max_len]

        else:  # padding if too short
            if len(audio_data) < max_len:  # do nothing if equal
                if data_filling == "repeatpad":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data

    return sample

class multimodal_dataset_train(Dataset):
    def __init__(self, cfg=None, cache_file = None, clip_prompt = prompt_template, tokenizer=None, max_txt_l=32):

        if cache_file is not None and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.captions, self.word_based_captions = pickle.load(f)

        self.video_tokenizer = tokenizer['video']
        self.audio_tokenizer = tokenizer['audio']
        self.image_tokenizer = tokenizer['image']

        self.max_txt_l = max_txt_l

        print("train datum nums:", len(self.word_based_captions))
        
    def __len__(self):
        return len(self.word_based_captions)

    def __getitem__(self, index):
        try:
            _, modal_type, caption, labels = self.captions[index].split('&&')
        except:
            print(self.captions[index])

        caption_video = self.video_tokenizer(caption, context_length=self.max_txt_l)
        caption_image = self.image_tokenizer(caption)
        caption_audio = self.audio_tokenizer(caption)

        wb_cap = torch.tensor(self.word_based_captions[index])

        return {'video': caption_video,
                'audio': caption_audio,
                'image': caption_image,
                'label': wb_cap,
                'modal_type': modal_type
                }

class video_k400_distill_test(Dataset):
    def __init__(self, cfg, cache_file=None):

        self.clsname = k400_object_categories

        val_csv = pd.read_csv(cfg.dataset.k400.test_file, header=0,sep=',')
        labels = list(set(val_csv['label']))
        labels.sort()

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = []
            for i,row in tqdm(val_csv.iterrows()):
                label = row.iloc[0]
                vid = row.iloc[1]
                st = row.iloc[2]
                end = row.iloc[3]
                video_file = f"{vid}_{st:0>6}_{end:0>6}.mp4"
                video_path = cfg.dataset.k400.video_file_path + video_file
                if os.path.exists(video_path):
                    self.data.append([video_path, labels.index(label)])
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)

        self.data = self.data[::10]

    def __len__(self):
        return len(self.data)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break
    
    def frames2tensor(self, vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(normalize(x), axis=(0)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=0)
        vid_tube = np.transpose(vid_tube, (0, 3, 1, 2))
        vid_tube = torch.from_numpy(vid_tube)
        return vid_tube
        
    def __getitem__(self, index):

        try:
            video_path, target = self.data[index]
            video = cv2.VideoCapture(video_path)
            frames = [x for x in self._frame_from_video(video)]
            frames_tensor = self.frames2tensor(frames)
        except:
            index = index-1
            video_path, target = self.data[index]
            video = cv2.VideoCapture(video_path)
            frames = [x for x in self._frame_from_video(video)]
            frames_tensor = self.frames2tensor(frames)

        return frames_tensor, target

class video_k600_distill_test(Dataset):
    def __init__(self, cfg, cache_file=None):

        self.clsname = k600_object_categories

        val_csv = pd.read_csv(cfg.dataset.k600.test_file, header=0,sep=',')
        labels = list(set(val_csv['label']))
        labels.sort()

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = []
            for i,row in tqdm(val_csv.iterrows()):
                label = row.iloc[0]
                vid = row.iloc[1]
                st = row.iloc[2]
                end = row.iloc[3]
                video_file = f"{vid}"
                video_path = cfg.dataset.k600.video_file_path + video_file
                if os.path.exists(video_path):
                    self.data.append([video_path, labels.index(label)])
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)

        # self.data = self.data[::10]

    def __len__(self):
        return len(self.data)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break
    
    def frames2tensor(self, vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(normalize(x), axis=(0)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=0)
        vid_tube = np.transpose(vid_tube, (0, 3, 1, 2))
        vid_tube = torch.from_numpy(vid_tube)
        return vid_tube
        
    def __getitem__(self, index):

        try:
            video_path, target = self.data[index]
            video = cv2.VideoCapture(video_path)
            frames = [x for x in self._frame_from_video(video)]
            frames_tensor = self.frames2tensor(frames)
        except:
            index = index-1
            video_path, target = self.data[index]
            video = cv2.VideoCapture(video_path)
            frames = [x for x in self._frame_from_video(video)]
            frames_tensor = self.frames2tensor(frames)

        return frames_tensor, target

class video_k700_distill_test(Dataset):
    def __init__(self, cfg, cache_file=None):

        self.clsname = k700_object_categories

        val_csv = pd.read_csv(cfg.dataset.k700.test_file, header=0,sep=',')
        labels = list(set(val_csv['label']))
        labels.sort()

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = []
            for i,row in tqdm(val_csv.iterrows()):
                label = row.iloc[0]
                vid = row.iloc[1]
                st = row.iloc[2]
                end = row.iloc[3]
                video_file = f"{label}/{vid}_{st:0>6}_{end:0>6}.mp4"
                video_path = cfg.dataset.k700.video_file_path + video_file
                if os.path.exists(video_path):
                    self.data.append([video_path, labels.index(label)])
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)

        # self.data = self.data[::15]

    def __len__(self):
        return len(self.data)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break
    
    def frames2tensor(self, vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(normalize(x), axis=(0)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=0)
        vid_tube = np.transpose(vid_tube, (0, 3, 1, 2))
        vid_tube = torch.from_numpy(vid_tube)
        return vid_tube
        
    def __getitem__(self, index):

        try:
            video_path, target = self.data[index]
            video = cv2.VideoCapture(video_path)
            frames = [x for x in self._frame_from_video(video)]
            frames_tensor = self.frames2tensor(frames)
        except:
            index = index-1
            video_path, target = self.data[index]
            video = cv2.VideoCapture(video_path)
            frames = [x for x in self._frame_from_video(video)]
            frames_tensor = self.frames2tensor(frames)

        return frames_tensor, target

class image_coco_distill_test(Dataset):
    def __init__(self, cfg, cache_file):

        self.clsname = coco_object_categories

        self.dataset_root = cfg.dataset.coco.image_coco_root
        self.coco_instance_json_file = cfg.dataset.coco.coco_ann_path
        cls_num = len(coco_object_categories)
        self.transform = self.build_transform()
        coco = COCO(self.coco_instance_json_file)
        self.valset_ids = coco.getImgIds()

        instance_info = {}
        with open(self.coco_instance_json_file, 'r') as f:
            instance_info = json.load(f)

        clsid2clsidx = {}
        clsidx2clsid = {}
        clsid2clsname = {}
        for idx, cat_info in enumerate(instance_info["categories"]):
            clsid2clsidx[cat_info['id']] = idx
            clsidx2clsid[idx] = cat_info['id']
            clsid2clsname[cat_info['id']] = cat_info['name']
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.img_idx, self.word_based_caption = pickle.load(f)
        else:
            self.img_idx = []
            test_label = torch.zeros((len(self.valset_ids), cls_num), dtype=torch.long)
            for idx, imgid in tqdm(enumerate(self.valset_ids)):
                self.img_idx.append(imgid)
                annIds = coco.getAnnIds(imgIds = imgid)
                anns = coco.loadAnns(annIds)
                for ann in anns:
                    tmp_idx = clsid2clsidx[ann['category_id']]
                    test_label[idx, tmp_idx] = 1
            self.word_based_caption = test_label
            with open(cache_file, 'wb') as f:
                pickle.dump([self.img_idx, self.word_based_caption], f)
        
    def __len__(self):
        return len(self.word_based_caption)
        
    def __getitem__(self, index):
        img_idx = str(self.img_idx[index])
        for i in range(6-len(img_idx)):
            img_idx = '0' + img_idx
        if 'val' in self.coco_instance_json_file:
            img_path = f"COCO_val2014_000000{img_idx}.jpg"
        img_path = os.path.join(self.dataset_root, img_path)
        img = Image.open(img_path)
        img = img.convert('RGB')
        wb_cap = self.word_based_caption[index].clone()[None,:]
        return self.transform(img), wb_cap
    
    def build_transform(self):

        tfm_test = transforms.Compose([
            transforms.Resize((224,224),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        return tfm_test

class image_voc_distill_test(Dataset):
    def __init__(self, cfg, cache_file):

        self.clsname = voc_object_categories
        dataset_dir = 'DATA/VOCdevkit/VOC2007/'
        self.transform = self.build_transform()
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.img_idx, self.word_based_caption = pickle.load(f)
        else:

            test_data_imname2label = self.read_object_labels(dataset_dir, phase='test')
            self.im_name_list_test = self.read_im_name_list(os.path.join(dataset_dir, 'ImageSets/Main/test.txt'))

            self.img_idx = []
            test_label = torch.zeros((len(test_data_imname2label), 20), dtype=torch.long)
            for index, (k, v) in tqdm(enumerate(test_data_imname2label.items())):
                self.img_idx.append(k)
                for iid,x in enumerate(v):
                    if x>0:
                        test_label[index, iid] = 1
            self.word_based_caption = test_label
            with open(cache_file, 'wb') as f:
                pickle.dump([self.img_idx, self.word_based_caption], f)

    def read_im_name_list(self, path):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                tmp = line.strip().split(' ')
                ret.append(tmp[0])
        return ret

    def read_image_label(self, file):
        data_ = dict()
        with open(file, 'r') as f:
            for line in f:
                tmp = line.strip().split(' ')
                name = tmp[0]
                label = int(tmp[-1])
                data_[name] = label
        return data_

    def read_object_labels(self, path, phase):
        path_labels = os.path.join(path, 'ImageSets', 'Main')
        labeled_data = dict()
        num_classes = len(voc_object_categories)

        for i in range(num_classes):
            file = os.path.join(path_labels, voc_object_categories[i] + '_' + phase + '.txt')
            data_ = self.read_image_label(file)

            if i == 0:
                for (name, label) in data_.items():
                    labels = torch.zeros(num_classes).long()
                    labels[i] = label
                    labeled_data[name] = labels
            else:
                for (name, label) in data_.items():
                    labeled_data[name][i] = label
        return labeled_data
        
    def __len__(self):
        return len(self.word_based_caption)
        
    def __getitem__(self, index):
        img_idx = str(self.img_idx[index])
        img_path = f"DATA/VOCdevkit/VOC2007/JPEGImages/{img_idx}.jpg"
        img = Image.open(img_path)
        img = img.convert('RGB')
        wb_cap = self.word_based_caption[index].clone()[None,:]
        return self.transform(img), wb_cap
    
    def build_transform(self):

        tfm_test = transforms.Compose([
            transforms.Resize((224,224),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        return tfm_test

class image_voc2012_distill_test(Dataset):
    def __init__(self, cfg, cache_file):

        self.clsname = voc_object_categories
        dataset_dir = 'DATA/voc2012/PascalVOC2012/VOC2012_train_val/'
        self.transform = self.build_transform()
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.img_idx, self.word_based_caption = pickle.load(f)
        else:

            test_data_imname2label = self.read_object_labels(dataset_dir, phase='val')
            self.im_name_list_test = self.read_im_name_list(os.path.join(dataset_dir, 'ImageSets/Main/val.txt'))

            self.img_idx = []
            test_label = torch.zeros((len(test_data_imname2label), 20), dtype=torch.long)
            for index, (k, v) in tqdm(enumerate(test_data_imname2label.items())):
                self.img_idx.append(k)
                for iid,x in enumerate(v):
                    if x>0:
                        test_label[index, iid] = 1
            self.word_based_caption = test_label
            with open(cache_file, 'wb') as f:
                pickle.dump([self.img_idx, self.word_based_caption], f)

    def read_im_name_list(self, path):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                tmp = line.strip().split(' ')
                ret.append(tmp[0])
        return ret

    def read_image_label(self, file):
        data_ = dict()
        with open(file, 'r') as f:
            for line in f:
                tmp = line.strip().split(' ')
                name = tmp[0]
                label = int(tmp[-1])
                data_[name] = label
        return data_

    def read_object_labels(self, path, phase):
        path_labels = os.path.join(path, 'ImageSets', 'Main')
        labeled_data = dict()
        num_classes = len(voc_object_categories)

        for i in range(num_classes):
            file = os.path.join(path_labels, voc_object_categories[i] + '_' + phase + '.txt')
            data_ = self.read_image_label(file)

            if i == 0:
                for (name, label) in data_.items():
                    labels = torch.zeros(num_classes).long()
                    labels[i] = label
                    labeled_data[name] = labels
            else:
                for (name, label) in data_.items():
                    labeled_data[name][i] = label
        return labeled_data
        
    def __len__(self):
        return len(self.word_based_caption)
        
    def __getitem__(self, index):
        img_idx = str(self.img_idx[index])
        img_path = f"DATA/voc2012/PascalVOC2012/VOC2012_train_val/JPEGImages/{img_idx}.jpg"
        img = Image.open(img_path)
        img = img.convert('RGB')
        wb_cap = self.word_based_caption[index].clone()[None,:]
        return self.transform(img), wb_cap
    
    def build_transform(self):

        tfm_test = transforms.Compose([
            transforms.Resize((224,224),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        return tfm_test

class image_nuswide_distill_test(Dataset):
    def __init__(self, cfg, cache_file):

        dataset_dir = 'DATA/NUSWIDE/'
        im_name_list_test = self.read_name_list(os.path.join(dataset_dir, 'ImageList/TestImagelist.txt'), False)
        path_labels = os.path.join(dataset_dir, 'TrainTestLabels')
        image_dir = os.path.join(dataset_dir, "Flickr")
        num_classes = len(nuswide_object_categories)
        self.img_idx = []
        self.transform = self.build_transform()

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.img_idx, self.word_based_caption = pickle.load(f)
        else:
            test_label = []
            for i in tqdm(range(num_classes)):
                file_ = os.path.join(path_labels, 'Labels_'+nuswide_object_categories[i]+'_Test.txt')
                cls_labels = []
                with open(file_, 'r') as f:
                    for j, line in enumerate(f):
                        tmp = line.strip()
                        cls_labels.append(int(tmp))
                test_label.append(torch.tensor(cls_labels, dtype=torch.long))
            test_label = torch.stack(test_label, dim=1)

            for name in im_name_list_test:
                self.img_idx.append(image_dir + '/' + '/'.join(name.split('\\')))

            self.word_based_caption = test_label
            with open(cache_file, 'wb') as f:
                pickle.dump([self.img_idx, self.word_based_caption], f)

        # self.img_idx = self.img_idx[::10]
        # self.word_based_caption = self.word_based_caption[::10]

    def read_name_list(self, path, if_split=True):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                if if_split:
                    tmp = line.strip().split(' ')
                    ret.append(tmp[0])
                else:
                    tmp = line.strip()
                    ret.append(tmp)
        return ret
        
    def __len__(self):
        return len(self.word_based_caption)
        
    def __getitem__(self, index):
        img_idx = str(self.img_idx[index])
        img_path = img_idx
        img = Image.open(img_path)
        img = img.convert('RGB')
        wb_cap = self.word_based_caption[index].clone()[None,:]
        return self.transform(img), wb_cap
    
    def build_transform(self):

        tfm_test = transforms.Compose([
            transforms.Resize((224,224),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        return tfm_test

class image_mini_distill_test(Dataset):
    def __init__(self, cfg, cache_file):

        self.clsname = mini_object_categories

        imgs = os.listdir('DATA/ImageNet-mini/images')
        val_csv = pd.read_csv('DATA/ImageNet-mini/test.csv',header=0)
        val_set = set(list(val_csv['label']))
        val_img = set(list(val_csv['filename']))

        file2cls = open('DATA/ImageNet-mini/ImageNet2012_label.txt','r', encoding='gbk').readlines()
        file2cls = [x.split()[1:3] for x in file2cls]

        file2iid = {}
        for file_cls in file2cls:
            file,name = file_cls
            name = " ".join(name.split('_'))
            if name in mini_object_categories:
                file2iid[file] = mini_object_categories.index(name)

        self.img_idx = []
        self.transform = self.build_transform()

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.img_idx, self.word_based_caption = pickle.load(f)
        else:
            test_label = []
            for i,img_file in tqdm(enumerate(imgs)):
                img_file = img_file.rstrip('.jpg')[:9]
                if img_file in val_set:
                    test_label.append(file2iid[img_file])
            # test_label = torch.stack(test_label, dim=1)

            for name in imgs:
                if name in val_img:
                    self.img_idx.append(name)

            self.word_based_caption = test_label
            with open(cache_file, 'wb') as f:
                pickle.dump([self.img_idx, self.word_based_caption], f)

        # self.img_idx = self.img_idx[::10]
        # self.word_based_caption = self.word_based_caption[::10]
        
    def __len__(self):
        return len(self.word_based_caption)
        
    def __getitem__(self, index):
        img_idx = str(self.img_idx[index])
        img_path = "DATA/ImageNet-mini/images/" + img_idx
        img = Image.open(img_path).convert("RGB")
        wb_cap = self.word_based_caption[index]
        return self.transform(img), wb_cap
    
    def build_transform(self):

        tfm_test = transforms.Compose([
            transforms.Resize((224,224),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        return tfm_test

class image_obj_distill_test(Dataset):
    def __init__(self, cfg, cache_file):

        self.clsname = obj_object_categories

        self.dataset_root = 'DATA/object365/Images/val/val'
        self.coco_instance_json_file = 'DATA/object365/Annotations/val/val.json'
        cls_num = len(obj_object_categories)
        self.transform = self.build_transform()
        coco = COCO(self.coco_instance_json_file)
        self.valset_ids = coco.getImgIds()

        instance_info = {}
        with open(self.coco_instance_json_file, 'r') as f:
            instance_info = json.load(f)

        clsid2clsidx = {}
        clsidx2clsid = {}
        clsid2clsname = {}
        for idx, cat_info in enumerate(instance_info["categories"]):
            clsid2clsidx[cat_info['id']] = idx
            clsidx2clsid[idx] = cat_info['id']
            clsid2clsname[cat_info['id']] = cat_info['name']
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.img_idx, self.word_based_caption = pickle.load(f)
        else:
            self.img_idx = []
            test_label = torch.zeros((len(self.valset_ids), cls_num), dtype=torch.long)
            for idx, imgid in tqdm(enumerate(self.valset_ids)):
                self.img_idx.append(imgid)
                annIds = coco.getAnnIds(imgIds = imgid)
                anns = coco.loadAnns(annIds)
                for ann in anns:
                    tmp_idx = clsid2clsidx[ann['category_id']]
                    test_label[idx, tmp_idx] = 1
            self.word_based_caption = test_label
            with open(cache_file, 'wb') as f:
                pickle.dump([self.img_idx, self.word_based_caption], f)

        # self.img_idx = self.img_idx[::5]
        # self.word_based_caption = self.word_based_caption[::5]
        
    def __len__(self):
        return len(self.word_based_caption)
        
    def __getitem__(self, index):
        img_idx = str(self.img_idx[index])
        img_path = self.dataset_root + f"/obj365_val_{img_idx.zfill(12)}.jpg"
        img = Image.open(img_path)
        img = img.convert('RGB')
        wb_cap = self.word_based_caption[index].clone()[None,:]
        return self.transform(img), wb_cap
    
    def build_transform(self):

        tfm_test = transforms.Compose([
            transforms.Resize((224,224),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        return tfm_test

class audio_esc50_distill_test(Dataset):
    def __init__(self, cfg, cache_file = None):
        super().__init__()

        cls2iid = json.load(open(cfg.dataset2cls2iid,'r'))['esc50']

        self.cls2index = []
        for c in esc50_object_categories:
            self.cls2index.append(cls2iid['audio_'+c])
        # self.cls2index = torch.tensor(cls2index)

        self.clsname = esc50_object_categories

        self.ESC50_INFO = {
            'meta_dir' : "meta/esc50.csv",
            'audio_dir' : "audio"
        }

        self.dataset_dir = cfg.dataset.esc50.dataset_dir
        self.dataframe = pd.read_csv(os.path.join(self.dataset_dir, self.ESC50_INFO['meta_dir']))
        num_rows, num_cols = self.dataframe.shape

        # init one-hot encode
        clsname2idx_ = {}
        nameset_compound = set()
        nameset = set()
        for idx, synset in enumerate(esc50_classname_synonyms):
            for n in synset:
                clsname2idx_[n] = idx
                if ' ' in n:
                    nameset_compound.add(n)
                    m = n.replace(' ', '')
                    clsname2idx_[m] = idx
                    nameset.add(m)
                else:
                    nameset.add(n)

        # get one_hot encode of every datum
        self.one_hot_category = []
        for i in range(num_rows):
            category = self.dataframe['category'][i]
            if '_' in category:
                category = category.replace('_', ' ')
            one_hot_code = torch.zeros(len(esc50_classname_synonyms), dtype=torch.int32)
            one_hot_code[clsname2idx_[category]] = 1
            self.one_hot_category.append(one_hot_code)
        
    def __len__(self):
        num_rows, _ = self.dataframe.shape
        return num_rows

    def __getitem__(self, index):
        audio_file = os.path.join(self.dataset_dir, self.ESC50_INFO['audio_dir'], self.dataframe['filename'][index])
        return audio_file, self.one_hot_category[index]