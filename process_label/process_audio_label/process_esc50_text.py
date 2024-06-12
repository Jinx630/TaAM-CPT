import os
import json

caption_file_path = [
    "training_data/Captions/audiocaps_train.jsonl",
    "training_data/Captions/audiocaps_val.jsonl",
    "training_data/Captions/audiocaps_test.jsonl",
    "training_data/Captions/WavCaps_AudioSet_SL.jsonl",
    "training_data/Captions/WavCaps_BBC_Sound_Effects.jsonl",
    "training_data/Captions/WavCaps_FreeSound.jsonl",
    "training_data/Captions/WavCaps_SoundBible.jsonl",
    "training_data/Captions/clotho_development.jsonl",
    "training_data/Captions/clotho_evaluation.jsonl"
]

caption_list = []
for train_file in caption_file_path:
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"File {train_file} does not exist")
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    caption_list += [datum['caption'] for datum in train_data]

caption_list = caption_list[::4]
print(len(caption_list))

with open('training_data/esc50_10w.txt','w') as f:
    for d in caption_list:
        d = d.replace('\n','')
        f.write(d+"&&")
        f.write('\n')

with open('data_gen/train_k400_2_cyrax.txt','r') as f:
    data = f.readlines()[::3]

print(len(data))
with open('training_data/k400_2_cyrax_10w.txt','w') as f:
    for d in data:
        f.write(d)
        # f.write('\n')