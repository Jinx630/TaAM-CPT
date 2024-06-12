from transformers import BertTokenizer, BertModel
from PretrainsPath import PRERAINEDS_PATH
tokenizer = BertTokenizer.from_pretrained(PRERAINEDS_PATH['bert-base-uncased'])
model = BertModel.from_pretrained(PRERAINEDS_PATH["bert-base-uncased"])
text = "Replace me by any text you'd like."

def bert_embeddings(text):
    # text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output
    
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained(PRERAINEDS_PATH['roberta-base'])
model = RobertaModel.from_pretrained(PRERAINEDS_PATH['roberta-base'])
text = "Replace me by any text you'd like."
def Roberta_embeddings(text):
    # text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output

from transformers import BartTokenizer, BartModel

tokenizer = BartTokenizer.from_pretrained(PRERAINEDS_PATH['facebook/bart-base'])
model = BartModel.from_pretrained(PRERAINEDS_PATH['facebook/bart-base'])
text = "Replace me by any text you'd like."
def bart_embeddings(text):
    # text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output