import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold, datasets

import umap
from sklearn.datasets import load_digits
import pickle

import torch

from transformers import pipeline

pipe = pipeline('feature-extraction', model='nferruz/ZymCTRL')

fasta_file = open('/agh/projects/noelia/NLP/zymCTRL/sequences/generated_2/finalfile.fasta','r') # the file with 10 seqs per class

sequences = {}
for line in fasta_file:
    if '>' in line:
        label = line[1:].split('.fasta')[0]
        sequences[label]=''
    else:
        sequences[label] = line.strip()

fasta_file.close()

embeddings={}
labels =  list(sequences.keys())
for label in labels:
    print(label)
    if label in embeddings: continue
    try:
        sentence = f"{label.split('_')[0]}<sep><start>{sequences[label]}<end><|endoftext|>"
        embeddings[label]=pipe(sentence)[0][0]
    except Exception as e:
        print(label,"was not printed due to",e)


import pickle

# Open a file and use dump()
with open('embeddings-generated.pkl', 'wb') as file:
    pickle.dump(embeddings, file)

