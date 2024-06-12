import pickle
import json
from tqdm import tqdm

# annotation_file_path = "DATA/COCO/annotations/captions_train2017.json"
# sampled_idx_path = "DATA/coco_caption_text_embed_sampled_idx.pkl"

# with open(sampled_idx_path, 'rb') as f:
#     sample_capid = pickle.load(f)

# caption_info = {}
# with open(annotation_file_path, 'r') as f:
#     caption_info = json.load(f)

# anno_id2path = {}
# for i in caption_info["annotations"]:
#     anno_id2path[i["id"]] = i
# print("captions_train2017 nums:", len(anno_id2path))

# data = []
# for i, capid in enumerate(tqdm(sample_capid)):
#     caption = anno_id2path[capid]['caption'].lower()
#     caption = caption.replace('\n','')
#     data.append(caption)

# with open('training_data/coco_caption_10w.txt','w') as f:
#     for d in data:
#         f.write(d+"&&")
#         f.write('\n')


data_nus = open('DATA/finetune_nuswide_20w.txt','r').readlines()[::2]
with open('training_data/nuswide_10w.txt','w') as f:
    for d in data_nus:
        d = d.replace('\n','')
        d = d.replace('  ',' ')
        f.write(d+"&&")
        f.write('\n')