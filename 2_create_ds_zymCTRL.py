import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
from evaluate import logging
import pickle
import collections
import glob


def remove_characters(str, chars_list):
    for char in chars_list:
        if char == '<sep>':
            str = str.replace(char, ' ')
        else:
            str = str.replace(char, '')
    return str

def calculatePerplexity(input_ids,model,tokenizer):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def main(input_ids, model,special_tokens,device,tokenizer):
    input_ids = tokenizer.encode(input_ids,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, 
    	top_k=8, #tbd
        repetition_penalty=1.2,
        max_length=600,
        eos_token_id=1,
        pad_token_id=0,
   	    do_sample=True,
   	    num_return_sequences=100)
    
    # Check sequence sanity, sequences not-truncated
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if len(new_outputs) < 3:
        del outputs
        # generate and truncate:
        outputs = model.generate(
        input_ids, 
    	top_k=8, #tbd
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
   	    do_sample=True,
   	    num_return_sequences=100)
    
    ppls = [calculatePerplexity(output, model, tokenizer) for output in outputs ]
    
    inf_res = {}
    ppl = list(zip(ppls, [tokenizer.decode(x) for x in outputs]))
    ppl.sort(key=lambda i:i[0])
    ppl = list(set(ppl))
    first_seq,second_seq = ppl[:2]
    cond_tok = first_seq[1].split('<sep>')[0].replace(' ','')
    inf_res[cond_tok] = [(remove_characters(first_seq[1], special_tokens), first_seq[0]),(remove_characters(second_seq[1], special_tokens), second_seq[0])]
    return inf_res

if __name__=='__main__':
    device = torch.device("cuda")
    print('Reading pretrained model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('/agh/projects/noelia/NLP/zymCTRL/sequences/generated_2/')
    model = GPT2LMHeadModel.from_pretrained('/agh/projects/noelia/NLP/zymCTRL/sequences/generated_2/').to(device) # change to new one
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']
    print('Reading natural files...')
    natural_files = glob.glob('/agh/projects/noelia/NLP/zymCTRL/sequences/natural_2/*.fasta')
    labels = [i.split('/')[-1].split('_')[0] for i in natural_files]
    print("we have :," len(labels), "labels")

    for label in tqdm(labels):
        print(label)
        if os.path.exists(f"/agh/projects/noelia/NLP/zymCTRL/sequences/generated_2/{label}_1.fasta"): 
            print(f"file {label} was already printed out")
            continue
        inf_res = main(label,model,special_tokens,device,tokenizer)
        for key,value in inf_res.items():
            for index, val in enumerate(value):            
                fn = open(f"/agh/projects/noelia/NLP/zymCTRL/sequences/generated_2/{label}_{index}.fasta", "w")
                fn.write(f'>{val[1],}\n{val[0]}\n')
                fn.close()
