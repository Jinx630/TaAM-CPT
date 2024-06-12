import pickle
from tqdm import tqdm

all_captions = []
all_word_based_captions = []

k700_captions, k700_word_based_captions = pickle.load(open('cache/train_925_label/test_01_k400_k600_k700_925.pkl', 'rb'))

k700_captions = k700_captions[::9]
k700_word_based_captions = k700_word_based_captions[::9]
print(len(k700_word_based_captions))

for cap in tqdm(k700_captions):
    all_captions.append('k400&&k600&&k700&&video&&' + cap)

all_word_based_captions.extend(k700_word_based_captions)

# coco
coco_captions, coco_word_based_captions = pickle.load(open('cache/train_925_label/test_01_coco_voc_nuswide_925.pkl', 'rb'))
print(len(coco_word_based_captions))

for cap in tqdm(coco_captions):
    all_captions.append('coco&&voc&&nuswide&&image&&' + cap)

all_word_based_captions.extend(coco_word_based_captions)

# esc50
esc50_captions, esc50_word_based_captions = pickle.load(open('cache/train_925_label/test_01_esc50_us8k_925.pkl', 'rb'))
print(len(esc50_word_based_captions))

for cap in tqdm(esc50_captions):
    all_captions.append('esc50&&us8k&&us8k&&audio&&' + cap)

all_word_based_captions.extend(esc50_word_based_captions)

assert len(all_captions) == len(all_word_based_captions)
print(len(all_word_based_captions))

with open('cache/train_925_label/test_01_k400_k600_k700_coco_voc_nuswide_esc50_us8k_925.pkl', 'wb') as f:
    pickle.dump([all_captions, all_word_based_captions], f)