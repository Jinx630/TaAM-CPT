import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import readline
from tqdm import tqdm
from itertools import combinations
import torch
import random
import logging

logging.getLogger('transformers').setLevel(logging.ERROR)

# os.environ['CUDA_VISIBLE_DEVICES'] ='1'

model = "pretrained_weights/llama-2-7B"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model).cuda()
model = model.eval()

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'


def main():
    ct = []
    with open("synonyms_obj365.txt", "r", encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.replace('\'','').strip('\n')[1:-1].split(',')
            ct.append(ann[0])
    
    all_combinations = []
    # for clss in ct:
    #     for c in clss:
    #         all_combinations.append((c,))
    # for a in range(400):
    #     for b in range(a+1,400):
    #         a_cls = ct[a][0]
    #         b_cls = ct[b][0]
    #         all_combinations.append((a_cls,b_cls))
    # for a in range(400):
    #     for b in range(400):
    #         if a!=b:
    #             a_cls = ct[a][0]
    #             b_clss = ct[b][1:]
    #             for b_cls in b_clss:
    #                 all_combinations.append((a_cls,b_cls))

    for num_words in range(1,3):  # 遍历从一个单词到三个单词的情况
        word_combinations = combinations(ct, num_words)
        all_combinations.extend(word_combinations)

    step = len(all_combinations)//4
    split = 3
    print('split: ',split)
    all_combinations = all_combinations[split*step:(split+1)*step]
    print(all_combinations[0], ' ', all_combinations[-1])

    BATCH_SIZE = 64
    for i in tqdm(range(0,len(all_combinations),BATCH_SIZE), desc="BATCH_SIZE"):
        query_list = []
        for j in range(BATCH_SIZE):
            if i+j == len(all_combinations):
                break
            word_list=[word.strip() for word in all_combinations[i+j]]
            string = ', '.join(word_list)
            query = "Making several English sentences to describe an image as simple as possible! " + \
                    "Requirements: Generate 5 English sentences! Each sentence should less than 25 words and include:" + \
                    string + \
                    "!\n"
            query_list.append(query)
        inputs = tokenizer(query_list, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=10, top_p=0.95) 
        outputs = tokenizer.batch_decode(outputs)
        with open(f'tmp_data_obj_3_llama-2-7B/generate_sentence_{split}.txt', mode='a', encoding='utf-8') as file:
            for _1 in range(BATCH_SIZE):
                file.write(outputs[_1] + '\n')

if __name__ == "__main__":
    main()
