from config_loader import load_config_from_path
from collections import OrderedDict, defaultdict
import os
import pickle
from dataset.data_helper import *
import nltk
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

data2cls = {
    "k400_classname_synonyms": k400_classname_synonyms,
    "k400_object_categories": k400_object_categories,
    "coco_classname_synonyms": coco_classname_synonyms,
    "coco_object_categories": coco_object_categories,
    "esc50_classname_synonyms": esc50_classname_synonyms,
    "esc50_object_categories": esc50_object_categories,
}

def synonyms_process(captions, word_based_captions, nameset_compound, nameset, cls_num, clsname2idx_use_synonyms, select_data):

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

    for i, cap in enumerate(tqdm(captions)):
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
        onehot_caption = word_based_captions[i]
        flag = 0
        for name in nameset_compound:
            name_ = ' ' + name.split('#')[0] + ' '
            if (name_ in cap):
                onehot_caption[clsname2idx_use_synonyms[name]] = 1
                flag = 1
                cap = cap.replace(name_, ' ')
                labels.append(name)

        for name in nameset:
            name_ = ' ' + name.split('#')[0] + ' '
            if (name_ in cap):
                onehot_caption[clsname2idx_use_synonyms[name]] = 1
                flag = 1
                cap = cap.replace(name_, ' ')
                labels.append(name)
        if 1 in onehot_caption:
            labels = list(set(labels))
            caption = caption + '&&' +','.join(labels)
            new_captions.append(caption)
            new_word_based_captions.append(onehot_caption)

    return new_captions, new_word_based_captions

def synonyms_no_process(train_text_path, cls_num, clsname2idx_use_synonyms, dataset_name):

    data = open(train_text_path,'r').readlines()
    print(len(data))

    captions = []
    word_based_captions = []

    for text in data:
        text = text.rstrip('\n')
        try:
            cap,label = text.split('&&')
        except:
            print(text)
        onehot_caption = [0] * cls_num
        labels = label.split(',')
        # if set(labels) & set(K400_object_categories):
        for l in labels:
            # if l in K400_object_categories:
            if l:
                l = l+f'#{dataset_name}'
                assert l in clsname2idx_use_synonyms
                onehot_caption[clsname2idx_use_synonyms[l]]=1
        cap = cap + '&&' + ','.join(labels)
        captions.append(cap)
        word_based_captions.append(onehot_caption)

    # for tem in IMAGENET_TEMPLATES:
    #     for ind, lab in enumerate(K400_object_categories):

    #         cap = tem.format(lab)
    #         onehot_caption = [0] * cls_num
    #         onehot_caption[ind]=1

    #         self.captions.append(cap)
    #         self.word_based_captions.append(onehot_caption)

    assert len(captions) == len(word_based_captions)

    return captions, word_based_captions


def create_multimodal_dataset(config):
    dataset2train_text_paths = defaultdict(dict)

    dataset2train_text_paths['k400']['is_video_audio_image']=config.dataset.k400.is_video_audio_image
    dataset2train_text_paths['k400']['classname_synonyms']=data2cls[f'k400_classname_synonyms']
    dataset2train_text_paths['k400']['object_categories']=data2cls[f'k400_object_categories']
    dataset2train_text_paths['k400']['use_synonyms']=config.dataset.k400.is_video_audio_image
    dataset2train_text_paths['k400']['train_text_path']=config.dataset.k400.train_text_path

    dataset2train_text_paths['coco']['is_video_audio_image']=config.dataset.coco.is_video_audio_image
    dataset2train_text_paths['coco']['classname_synonyms']=data2cls[f'coco_classname_synonyms']
    dataset2train_text_paths['coco']['object_categories']=data2cls[f'coco_object_categories']
    dataset2train_text_paths['coco']['use_synonyms']=config.dataset.coco.is_video_audio_image
    dataset2train_text_paths['coco']['train_text_path']=config.dataset.coco.train_text_path

    dataset2train_text_paths['esc50']['is_video_audio_image']=config.dataset.esc50.is_video_audio_image
    dataset2train_text_paths['esc50']['classname_synonyms']=data2cls[f'esc50_classname_synonyms']
    dataset2train_text_paths['esc50']['object_categories']=data2cls[f'esc50_object_categories']
    dataset2train_text_paths['esc50']['use_synonyms']=config.dataset.esc50.is_video_audio_image
    dataset2train_text_paths['esc50']['train_text_path']=config.dataset.esc50.train_text_path

    return dataset2train_text_paths

cache_file='cache/test_01_tmp.pkl'

config = load_config_from_path('./configs/Caption_distill_double_config_video_clip.json')

dataset2train_text_paths = create_multimodal_dataset(config)

cls_num = 0
clsname2idx_use_synonyms = {}
# self.clsname2idx_no_use_synonyms = {}
nameset_compound = set()
nameset = set()

select_data = ['esc50','coco']

for dataset_name,value in dataset2train_text_paths.items():

    if dataset_name in select_data:

        classname_synonyms = value['classname_synonyms']
        use_synonyms = value['use_synonyms']

        for synset in classname_synonyms:
            for n in synset:
                clsname2idx_use_synonyms[n+f'#{dataset_name}'] = cls_num
                if ' ' in n:
                    nameset_compound.add(n+f'#{dataset_name}')
                    m = n.replace(' ', '')
                    clsname2idx_use_synonyms[m+f'#{dataset_name}'] = cls_num
                    nameset.add(m+f'#{dataset_name}')
                else:
                    nameset.add(n+f'#{dataset_name}')
            cls_num+=1

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

        if dataset_name in select_data:
            caps, word_caps = synonyms_no_process(train_text_path, cls_num, clsname2idx_use_synonyms, dataset_name)
            captions += caps
            word_based_captions += word_caps

    captions, word_based_captions = synonyms_process(captions, word_based_captions, nameset_compound, nameset, cls_num, clsname2idx_use_synonyms, select_data)

    with open(cache_file, 'wb') as f:
        pickle.dump([captions, word_based_captions], f)

assert len(captions) == len(word_based_captions)
print(len(captions))