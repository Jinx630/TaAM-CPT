from config_loader import load_config_from_path
from collections import OrderedDict, defaultdict
import os
import pickle
from data_helper import *
import nltk
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import json

data2cls = {
    "k400_classname_synonyms": k400_classname_synonyms,
    "k400_object_categories": k400_object_categories,
}

def synonyms_no_process():

    data = open('data_gen/train_k400_2_sys_cyrax.txt','r').readlines()
    # data_700 = open('data_gen/train_k700_2_cyrax.txt','r').readlines()
    # data = data + data_700
    
    data_27 = set(open('data_gen/train_k400_2_sys_sim_pre40_43w_cyrax.txt','r').readlines())

    captions = []
    word_based_captions = []

    cls_num = len(k400_object_categories)

    for text in tqdm(data):
        # if 'dying hair' in text:
        #     continue
        # if 'passing american football (not in game)':
        #     text = text.replace('passing american football (not in game)', 'passing American football (not in game)')
        if text in data_27:
            continue
        text = text.rstrip('\n')
        cap,label = text.split('&&')
        onehot_caption = [0] * cls_num
        labels = label.split(',')
        for l in labels:
            if l == 'capoeira dancing':
                l = 'capoeira'
            if l == 'barbecuing':
                l = 'barbequing'

            # assert l in cls2iid

            if l in k400_object_categories:
                onehot_caption[k400_object_categories.index(l)]=1
        if sum(onehot_caption)>0:
            labels = list(set(labels))
            caption = cap + '&&' +','.join(labels)
            captions.append(caption)
            word_based_captions.append(onehot_caption)
    assert len(captions) == len(word_based_captions)

    print("train datum nums:", len(word_based_captions))

    return captions, word_based_captions


def create_multimodal_dataset(config):
    dataset2train_text_paths = defaultdict(dict)

    dataset2train_text_paths['k400']['is_video_audio_image']=config.dataset.k400.is_video_audio_image
    dataset2train_text_paths['k400']['classname_synonyms']=data2cls['k400_classname_synonyms']
    dataset2train_text_paths['k400']['object_categories']=data2cls['k400_object_categories']
    dataset2train_text_paths['k400']['use_synonyms']=config.dataset.k400.is_video_audio_image
    dataset2train_text_paths['k400']['train_text_path']=config.dataset.k400.train_text_path

    return dataset2train_text_paths

cache_file='cache/train_new_label/test_01_k400_925.pkl'

config = load_config_from_path('configs/Caption_distill_double_config_video_clip.json')

dataset2train_text_paths = create_multimodal_dataset(config)

captions = []
word_based_captions = []

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        captions, word_based_captions = pickle.load(f)
else:
    for dataset_name,value in dataset2train_text_paths.items():

        caps, word_caps = synonyms_no_process()

        captions += caps
        word_based_captions += word_caps

    with open(cache_file, 'wb') as f:
        pickle.dump([captions, word_based_captions], f)

assert len(captions) == len(word_based_captions)
print(len(captions))