### Script to send to the erlangen cluster:
from transformers import pipeline
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
from transformers import TextGenerationPipeline
import numpy as np
import time
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = '/agh/projects/noelia/NLP/zymCTRL/erlangen/output/'
tokenizer = AutoTokenizer.from_pretrained('/agh/projects/noelia/NLP/zymCTRL/dataset_preparation/tokenizer')
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

print("model and tokenizer loaded")

brenda_classes = ['7.1.1.9',
 '2.7.13.3',
 '3.6.4.12',
 '7.1.1.-',
 '2.7.11.1',
 '2.7.7.6',
 '3.1.-.-',
 '2.7.7.7',
 '7.1.1.2',
 '5.2.1.8']


input_ids = tokenizer.encode('7.1.1.9', return_tensors='pt').to(device)

special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

def remove_characters(str, chars_list):
    for char in chars_list:
        if char == '<sep>':
            continue
        else:
            str = str.replace(char, '')
    return str

def return_strings(sample_outputs):
    output_str = ''.join([remove_characters(tokenizer.decode(x),special_tokens).split('<sep>')[1] for x in sample_outputs])
    return output_str

print("-----------")
print(f"0: Do sample top_k 8")

output_str = ''
for i in brenda_classes:

    input_ids = tokenizer.encode(i, return_tensors='pt').to(device)
    a = model.generate(input_ids,top_k = 8, max_length=1024, do_sample=True, rep_penalty=1.3, num_return_sequences=20, eos_token_id=1, pad_token_id=0)
    output_str += return_strings(a)

counter = Counter(output_str)
print(len(output_str), counter)


print("-----------")
print(f"1: Do sample top_k 458")

output_str = ''
for i in brenda_classes:
    
    input_ids = tokenizer.encode(i, return_tensors='pt').to(device)
    a = model.generate(input_ids,top_k = 458, max_length=1024, do_sample=True, rep_penalty=1.3, num_return_sequences=20, eos_token_id=1, pad_token_id=0)
    output_str += return_strings(a)

counter = Counter(output_str)
print(len(output_str), counter)

print("-----------")
print(f"2: Do sample top_k 6")

output_str = ''
for i in brenda_classes:
    input_ids = tokenizer.encode(i, return_tensors='pt').to(device)
    a = model.generate(input_ids,top_k = 6, max_length=1024, do_sample=True, rep_penalty=1.3, num_return_sequences=20, eos_token_id=1, pad_token_id=0)
    output_str += return_strings(a)
counter = Counter(output_str)
print(len(output_str),counter)


print("-----------")
print(f"3: Do sample top_k 200")

output_str = ''
for i in brenda_classes:
    input_ids = tokenizer.encode(i, return_tensors='pt').to(device)
    a = model.generate(input_ids,top_k = 200, max_length=1024, do_sample=True, rep_penalty=1.3, num_return_sequences=20, eos_token_id=1, pad_token_id=0)
    output_str += return_strings(a)
counter = Counter(output_str)
print(len(output_str),counter)

print("-----------")
print(f"4: Do sample top_k 10")

output_str = ''
for i in brenda_classes:
    input_ids = tokenizer.encode(i, return_tensors='pt').to(device)
    a = model.generate(input_ids,top_k = 10, max_length=1024, do_sample=True, rep_penalty=1.3, num_return_sequences=20, eos_token_id=1, pad_token_id=0)
    output_str += return_strings(a)
counter = Counter(output_str)
print(len(output_str),counter)

print("-----------")
print(f"5: Do sample top_k 20")

output_str = ''
for i in brenda_classes:
    input_ids = tokenizer.encode(i, return_tensors='pt').to(device)
    a = model.generate(input_ids,top_k = 20, max_length=1024, do_sample=True, rep_penalty=1.3, num_return_sequences=20, eos_token_id=1, pad_token_id=0)
    output_str += return_strings(a)
counter = Counter(output_str)
print(len(output_str),counter)
