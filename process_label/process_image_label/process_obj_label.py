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
    "obj_classname_synonyms": obj_classname_synonyms,
    "obj_object_categories": obj_object_categories,
    # "voc_classname_synonyms": voc_classname_synonyms,
    # "voc_object_categories": voc_object_categories,
    # "nuswide_classname_synonyms": nuswide_classname_synonyms,
    # "nuswide_object_categories": nuswide_object_categories,
}

def synonyms_process(train_text_path, nameset_compound, nameset, clsname2idx_use_synonyms):

    data = open(train_text_path,'r').readlines()

    # data = data + data_nus

    cls_num = len(obj_object_categories)

    new_captions = []
    new_word_based_captions = []

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    wnl = WordNetLemmatizer()

    for i, cap in enumerate(tqdm(data)):
        try:
            cap = cap.rstrip('\n')
            caption,labels = cap.split('&&')
            labels = labels.split(',')
            caption = caption.lower()
            noum_list = word_tokenize(caption)
            
            tagged_sent = pos_tag(noum_list)  
            
            lemmas_sent = []
            for tag in tagged_sent:
                wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
                cap = ' ' + ' '.join(lemmas_sent) + ' '    
            onehot_caption = [0] * cls_num
            for l in labels:
                onehot_caption[clsname2idx_use_synonyms[l]] = 1
            # flag = 0
            # for name in nameset_compound:
            #     name_ = ' ' + name.split('#')[0] + ' '
            #     if (name_ in cap):
            #         onehot_caption[clsname2idx_use_synonyms[name]] = 1
            #         flag = 1
            #         cap = cap.replace(name_, ' ')
            #         labels.append(name)

            # for name in nameset:
            #     name_ = ' ' + name.split('#')[0] + ' '
            #     if (name_ in cap):
            #         onehot_caption[clsname2idx_use_synonyms[name]] = 1
            #         flag = 1
            #         cap = cap.replace(name_, ' ')
            #         labels.append(name)
            if 1 in onehot_caption:
                labels = list(set(labels))
                caption = caption + '&&' +','.join(labels)
                new_captions.append(caption)
                new_word_based_captions.append(onehot_caption)
        except:
            continue

    return new_captions, new_word_based_captions


def create_multimodal_dataset(config):
    dataset2train_text_paths = defaultdict(dict)

    dataset2train_text_paths['obj']['is_video_audio_image']=config.dataset.obj.is_video_audio_image
    dataset2train_text_paths['obj']['classname_synonyms']=data2cls[f'obj_classname_synonyms']
    dataset2train_text_paths['obj']['object_categories']=data2cls[f'obj_object_categories']
    dataset2train_text_paths['obj']['use_synonyms']=config.dataset.obj.is_video_audio_image
    dataset2train_text_paths['obj']['train_text_path']=config.dataset.obj.train_text_path

    # dataset2train_text_paths['voc']['is_video_audio_image']=config.dataset.voc.is_video_audio_image
    # dataset2train_text_paths['voc']['classname_synonyms']=data2cls[f'voc_classname_synonyms']
    # dataset2train_text_paths['voc']['object_categories']=data2cls[f'voc_object_categories']
    # dataset2train_text_paths['voc']['use_synonyms']=config.dataset.voc.is_video_audio_image
    # dataset2train_text_paths['voc']['train_text_path']=config.dataset.voc.train_text_path

    # dataset2train_text_paths['nuswide']['is_video_audio_image']=config.dataset.nuswide.is_video_audio_image
    # dataset2train_text_paths['nuswide']['classname_synonyms']=data2cls[f'nuswide_classname_synonyms']
    # dataset2train_text_paths['nuswide']['object_categories']=data2cls[f'nuswide_object_categories']
    # dataset2train_text_paths['nuswide']['use_synonyms']=config.dataset.nuswide.is_video_audio_image
    # dataset2train_text_paths['nuswide']['train_text_path']=config.dataset.nuswide.train_text_path

    return dataset2train_text_paths

cache_file='cache/train_new_label/test_01_obj_925.pkl'

config = load_config_from_path('configs/Caption_distill_double_config_video_clip.json')

dataset2train_text_paths = create_multimodal_dataset(config)

num = 0
clsname2idx_use_synonyms = {}
# self.clsname2idx_no_use_synonyms = {}
nameset_compound = set()
nameset = set()

for dataset_name,value in dataset2train_text_paths.items():

    classname_synonyms = value['classname_synonyms']
    use_synonyms = value['use_synonyms']

    for synset in classname_synonyms:
        # assert 'image_'+synset[0] in cls2iid
        # iid = cls2iid['image_'+synset[0]]
        for n in synset:
            clsname2idx_use_synonyms[n] = num
            if ' ' in n:
                nameset_compound.add(n)
                m = n.replace(' ', '')
                clsname2idx_use_synonyms[m] = num
                nameset.add(m)
            else:
                nameset.add(n)
        num+=1

captions = []
word_based_captions = []

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        captions, word_based_captions = pickle.load(f)
else:
    for dataset_name,value in dataset2train_text_paths.items():

        is_video_audio_image = value['is_video_audio_image']
        classname_synonyms = value['classname_synonyms']
        object_categories = value['object_categories']
        use_synonyms = value['use_synonyms']
        train_text_path = value['train_text_path']

    captions, word_based_captions = synonyms_process(train_text_path, nameset_compound, nameset, clsname2idx_use_synonyms)

    with open(cache_file, 'wb') as f:
        pickle.dump([captions, word_based_captions], f)

assert len(captions) == len(word_based_captions)
print(len(captions))