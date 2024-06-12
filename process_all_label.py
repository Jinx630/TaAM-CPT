import pickle
from tqdm import tqdm

label_sum = [400,80,50]

all_captions = []
all_word_based_captions = []

# ==============================================================
# =========================== k400 =============================
# ==============================================================
k400_captions, k400_word_based_captions = pickle.load(open('cache/train_new_label/test_01_k400_925.pkl', 'rb'))

k400_captions = k400_captions[:20]
k400_word_based_captions = k400_word_based_captions[:20]
print(len(k400_word_based_captions))

for cap in tqdm(k400_captions):
    all_captions.append('k400&&video&&' + cap)
cls_index = label_sum.index(len(k400_word_based_captions[0]))

pre_sum = sum(label_sum[:cls_index])*[0]
pos_sum = sum(label_sum[cls_index+1:])*[0]

for word in tqdm(k400_word_based_captions):
    all_word_based_captions.append(pre_sum+word+pos_sum)

# # ==============================================================
# # =========================== k600 =============================
# # ==============================================================
# k600_captions, k600_word_based_captions = pickle.load(open('cache/train_new_label/test_01_k600_925.pkl', 'rb'))

# k600_captions = k600_captions[::25]
# k600_word_based_captions = k600_word_based_captions[::25]
# print(len(k600_word_based_captions))

# for cap in tqdm(k600_captions):
#     all_captions.append('k600&&video&&' + cap)
# cls_index = label_sum.index(len(k600_word_based_captions[0]))

# pre_sum = sum(label_sum[:cls_index])*[0]
# pos_sum = sum(label_sum[cls_index+1:])*[0]

# for word in tqdm(k600_word_based_captions):
#     all_word_based_captions.append(pre_sum+word+pos_sum)

# # ==============================================================
# # =========================== k700 =============================
# # ==============================================================
# k700_captions, k700_word_based_captions = pickle.load(open('cache/train_label/test_01_k700.pkl', 'rb'))

# k700_captions = k700_captions[::15]
# k700_word_based_captions = k700_word_based_captions[::15]
# print(len(k700_word_based_captions))

# for cap in tqdm(k700_captions):
#     all_captions.append('k700&&video&&' + cap)
# cls_index = label_sum.index(len(k700_word_based_captions[0]))

# pre_sum = sum(label_sum[:cls_index])*[0]
# pos_sum = sum(label_sum[cls_index+1:])*[0]

# for word in tqdm(k700_word_based_captions):
#     all_word_based_captions.append(pre_sum+word+pos_sum)

# ==============================================================
# =========================== coco =============================
# ==============================================================
coco_captions, coco_word_based_captions = pickle.load(open('cache/train_new_label/test_01_coco_925.pkl', 'rb'))
print(len(coco_word_based_captions))

for cap in tqdm(coco_captions):
    all_captions.append('coco&&image&&' + cap)
cls_index = label_sum.index(len(coco_word_based_captions[0]))

pre_sum = sum(label_sum[:cls_index])*[0]
pos_sum = sum(label_sum[cls_index+1:])*[0]

for word in tqdm(coco_word_based_captions):
    all_word_based_captions.append(pre_sum+word+pos_sum)

# # ==============================================================
# # ============================ voc =============================
# # ==============================================================
# voc_captions, voc_word_based_captions = pickle.load(open('cache/train_new_label/test_01_voc_925.pkl', 'rb'))
# print(len(voc_word_based_captions))

# for cap in tqdm(voc_captions):
#     all_captions.append('voc&&image&&' + cap)
# cls_index = label_sum.index(len(voc_word_based_captions[0]))

# pre_sum = sum(label_sum[:cls_index])*[0]
# pos_sum = sum(label_sum[cls_index+1:])*[0]

# for word in tqdm(voc_word_based_captions):
#     all_word_based_captions.append(pre_sum+word+pos_sum)

# # ==============================================================
# # ========================= nuswide ============================
# # ==============================================================
# nuswide_captions, nuswide_word_based_captions = pickle.load(open('cache/train_new_label/test_01_nuswide_925.pkl', 'rb'))
# print(len(nuswide_word_based_captions))

# for cap in tqdm(nuswide_captions):
#     all_captions.append('nuswide&&image&&' + cap)
# cls_index = label_sum.index(len(nuswide_word_based_captions[0]))

# pre_sum = sum(label_sum[:cls_index])*[0]
# pos_sum = sum(label_sum[cls_index+1:])*[0]

# for word in tqdm(nuswide_word_based_captions):
#     all_word_based_captions.append(pre_sum+word+pos_sum)

# # ==============================================================
# # =========================== mini =============================
# # ==============================================================
# mini_captions, mini_word_based_captions = pickle.load(open('cache/train_new_label/test_01_mini_925.pkl', 'rb'))

# mini_captions = mini_captions[::7]
# mini_word_based_captions = mini_word_based_captions[::7]
# print(len(mini_word_based_captions))

# for cap in tqdm(mini_captions):
#     all_captions.append('mini&&image&&' + cap)
# cls_index = label_sum.index(len(mini_word_based_captions[0]))

# pre_sum = sum(label_sum[:cls_index])*[0]
# pos_sum = sum(label_sum[cls_index+1:])*[0]

# for word in tqdm(mini_word_based_captions):
#     all_word_based_captions.append(pre_sum+word+pos_sum)

# # ==============================================================
# # ============================ obj =============================
# # ==============================================================
# obj_captions, obj_word_based_captions = pickle.load(open('cache/train_new_label/test_01_obj_925.pkl', 'rb'))

# obj_captions = obj_captions[::3]
# obj_word_based_captions = obj_word_based_captions[::3]
# print(len(obj_word_based_captions))

# for cap in tqdm(obj_captions):
#     all_captions.append('obj&&image&&' + cap)
# cls_index = label_sum.index(len(obj_word_based_captions[0]))

# pre_sum = sum(label_sum[:cls_index])*[0]
# pos_sum = sum(label_sum[cls_index+1:])*[0]

# for word in tqdm(obj_word_based_captions):
#     all_word_based_captions.append(pre_sum+word+pos_sum)

# ==============================================================
# =========================== esc50 ============================
# ==============================================================
esc50_captions, esc50_word_based_captions = pickle.load(open('cache/train_new_label/test_01_esc50_925.pkl', 'rb'))
esc50_captions = esc50_captions[:20]
esc50_word_based_captions = esc50_word_based_captions[:20]
print(len(esc50_word_based_captions))

for cap in tqdm(esc50_captions):
    all_captions.append('esc50&&audio&&' + cap)
cls_index = label_sum.index(len(esc50_word_based_captions[0]))

pre_sum = sum(label_sum[:cls_index])*[0]
pos_sum = sum(label_sum[cls_index+1:])*[0]

for word in tqdm(esc50_word_based_captions):
    all_word_based_captions.append(pre_sum+word+pos_sum)

assert len(all_captions) == len(all_word_based_captions)
print(len(all_word_based_captions))

# # ==============================================================
# # =========================== us8k =============================
# # ==============================================================
# us8k_captions, us8k_word_based_captions = pickle.load(open('cache/train_new_label/test_01_us8k_925.pkl', 'rb'))
# print(len(esc50_word_based_captions))

# for cap in tqdm(us8k_captions):
#     all_captions.append('us8k&&audio&&' + cap)
# cls_index = label_sum.index(len(us8k_word_based_captions[0]))

# pre_sum = sum(label_sum[:cls_index])*[0]
# pos_sum = sum(label_sum[cls_index+1:])*[0]

# for word in tqdm(us8k_word_based_captions):
#     all_word_based_captions.append(pre_sum+word+pos_sum)


# ==============================================================
# =========================== merge ============================
# ==============================================================

assert len(all_captions) == len(all_word_based_captions)
print(len(all_word_based_captions))

with open('cache/train_new_label/test_01_k400_coco_esc50_coco.pkl', 'wb') as f:
    pickle.dump([all_captions, all_word_based_captions], f)