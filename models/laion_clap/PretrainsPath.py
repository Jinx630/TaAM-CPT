import os
from pathlib import Path
bert_base_uncased_path = os.path.join(Path(__file__).parent, "Pretrained_Models/", "bert-base-uncased")
roberta_base_path = os.path.join(Path(__file__).parent, "Pretrained_Models/", "roberta-base")
bart_base_path =  os.path.join(Path(__file__).parent, "Pretrained_Models/", "bart-base")

if not os.path.exists(bert_base_uncased_path):
    bert_base_uncased_path = "bert-base-uncased"
    
if not os.path.exists(roberta_base_path):
    roberta_base_path = "roberta-base"
    
if not os.path.exists(bart_base_path):
    bart_base_path = "facebook/bart-base"
    

PRERAINEDS_PATH = {
    "bert-base-uncased" : "/mnt/workspace/workgroup/jinmu/ckpts/bert-base-uncased",
    "roberta-base" : "/mnt/workspace/workgroup/jinmu/ckpts/roberta-base",
    "facebook/bart-base" : "/mnt/workspace/workgroup/jinmu/ckpts/bart-base"
}