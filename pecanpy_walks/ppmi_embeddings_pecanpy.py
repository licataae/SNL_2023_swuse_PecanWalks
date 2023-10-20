#!/usr/bin/env python
# coding: utf-8

import os
import re
import psutil
import pandas as pd
import numpy as np
from multiprocessing import Pool
from datasets import load_dataset
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import pecanpy
from pecanpy import pecanpy

#load in dataset, take a subset of samples & remove the fields we don't need (url, title, id) so we just have 'text'
#ds = load_dataset("pszemraj/simple_wikipedia_LM") # EXAMPLE
ds = load_dataset("olm/wikipedia", language="en", date="20231001")
# convert the dataset to a pandas df & export...
train_dataset = ds['train']
print(train_dataset)
small_dataset = train_dataset.select(range(1000))
pd_dataset_test = small_dataset.to_pandas()
#train_dataset = dataset.train_test_split(test_size=0.1)


def load_cue_words(lang, project_dir):
    if lang == 'nl':
        df = pd.read_csv(os.path.join(project_dir, "input_files/cue_words/test_cuelist_nl.csv"))
    elif lang == 'fr':
        df = pd.read_csv(os.path.join(project_dir, "input_files/cue_words/test_cuelist_fr.csv"))
    elif lang == 'en':
        df = pd.read_csv(os.path.join(project_dir, "input_files/cue_words/test_cuelist_en.csv"))
    else:
        raise ValueError("Invalid language! 3 options for this arg: 'nl', 'fr', and 'en'")

    cue_words = df['word'].tolist()
    print(f"{len(cue_words)} cue words have been loaded for '{lang}'")
    return cue_words

def preprocess_text(text, lang):
    text = text.lower()
    #text = text.replace("-", " ")  # Replace hyphens with spaces
    #text = ''.join([c for c in text.split() if c.isalnum() or c.isspace()])  # remove non-alphanumeric characters except spaces
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    text = re.sub(r'[-,."+/!?&:;€£$0-9(){|%#°]', '', text)  # Remove hyphens, commas, ampersands, and digits
    #text = tokenizer.tokenizer(text)
    # tokenization
    tokens = text.split()  # split on whitespace instead of using word_tokenize

    # stopword removal
    if lang == 'en':
        stop_words = set(stopwords.words('english'))
    elif lang == 'fr':
        stop_words = set(stopwords.words('french'))
    elif lang == 'nl':
        stop_words = set(stopwords.words('dutch'))
    else:
        raise ValueError("Invalid language! 3 options for this arg: 'nl', 'fr', and 'en'")

    tokens = [token for token in tokens if token not in stop_words]

    return tokens


toTrainText = pd_dataset_test['text'].str.cat(sep=' ')
print(toTrainText[0:100])
print("length of sample:", len(toTrainText))

proc = preprocess_text(toTrainText, lang="en")
#print(proc[0:20000])
word_fq_thresh = 100 #frequency to consider (drop words occurring less than 20 times)

# get word occurence stat
vocab = set(proc)
vocab_counts = {w:0 for i, w in enumerate(vocab)}
for w in proc:
    vocab_counts[w] += 1
vocab = set( w for w, c in vocab_counts.items() if c > word_fq_thresh )

# index corpus
wi = {tk:i for i, tk in enumerate(vocab)}
print(str(wi)[0:500])
iw ={v:k for k, v in wi.items()}

vocab_size = len(vocab)

# save vocabulary
with open("svd_ppmi_embeddings_vocab_test.pkl", "wb") as f:
    pickle.dump(wi, f)

#-------------------- construct PPMI Matrix -----------------#
import tqdm
import math
from scipy import sparse

def cooccur_graph(proc, vocab, window=5):
    
    # build a dictionary of co-occurrences that we transform into a graph...
    D = dict()
    context = {i:0 for i, tk in enumerate(vocab)}
    target = {i:0 for i, tk in enumerate(vocab)}

    # count co-occurrences of all (target, context) pairs
    for i, tgt_w in tqdm.tqdm(enumerate(proc)):

        # record cooccurences in context window
        cntx = [ w for w in proc[(i-window):i] + proc[(i+1):(i+window+1)] ]
        for cntx_w in cntx:

            # filter out words with low fq
            if (tgt_w not in vocab) or (cntx_w not in vocab):
                continue

            pair = (wi[tgt_w], wi[cntx_w])

            # update coocurrence
            if pair in D.keys():
                D[pair] += 1
            else:
                D[pair] = 1

            # update target, context word count
            target[pair[0]] += 1
            context[pair[1]] += 1
    
    return D, context, target

D, context, target = cooccur_graph(proc, vocab, window=5)

# compute PMI value
D_size = len(D)
def pmi(tgt_w, cntx_w, ppmi=False, target=target, context=context, D=D):
    Nw = target.get(tgt_w)
    Nc = context.get(cntx_w)
    Nwc = D.get((tgt_w, cntx_w))
    
    if Nwc is None:
        return None
    if Nwc == 0:
        if ppmi:
            return 0
        else:
            return -math.inf

    val = math.log10( (Nwc * D_size) / (Nw * Nc) )
    if ppmi:
        val = max(val, 0)

    return val

# construct PPMI matrix - rows & columns = word indices & values = PPMI
row = []
col = []
data = []
for pair in D.keys():
    row.append(pair[0])
    col.append(pair[1])
    data.append(pmi(*pair, ppmi=True))
ppmi_matrix = sparse.csr_matrix((data, (row, col)), shape=(vocab_size, vocab_size))
#print("HERE IS THE PPMI MATRIX", ppmi_matrix)

# sparisfy
ppmi_matrix.eliminate_zeros()
#print("sparse ppmi matrix (sanity check)", ppmi_matrix)
#row_sums = ppmi_matrix.sum(axis=1)
#ppmi_matrix_normalized = ppmi_matrix / row_sums[:, np.newaxis]
ppmi_matrix_normalized = ppmi_matrix / ppmi_matrix.sum(axis=1)
print("ppmi matrix:\n", ppmi_matrix_normalized)

#transform the ppmi matrix to a graph, where nodes=words, edges=non-zero cells weighted by their value (PPMI)                                                                           
def get_graph_from_PPMI(matrix):
    gr = nx.Graph()
    rows, cols = np.where(matrix != 0)
    edges = zip(rows.tolist(), cols.tolist())
    for row, col in edges: #make sure the non-zero edges are weighted by their PPMI
        weight = matrix[row, col]
        gr.add_edge(row, col, weight=weight)
    all_rows = range(0, matrix.shape[0])
    for n in all_rows:
        gr.add_node(n)
    #nx.draw(gr, node_size=900, with_labels=True)                                                                                                                                        
    #plt.show()
    return gr

#graph = nx.from_numpy_matrix(ppmi_matrix_normalized)
#graph = nx.from_numpy_array(ppmi_matrix_normalized)
graph = get_graph_from_PPMI(ppmi_matrix_normalized)
edgelist = nx.write_edgelist(graph, "test.edgelist")

#not the most elegant but it will do, need to have an edgelist only with id1 id2 val
with open('test.edgelist', 'r') as input_file, open('processed.edgelist', 'w') as output_file:
    for line in input_file:
        parts = line.split()
        source_node = int(parts[0])
        target_node = int(parts[1])
        weight = float(parts[3][0:-1])  # Extract the weight value

        output_file.write(f"{source_node} {target_node} {weight}\n")

#----------------- release memory --------------------#                                                                                                                                  
#del vocab need for kNN word2id?
del proc
del D
del context
del target
del row
del col
del data

#----------------- random walks to learn embeddings --------------------#                                                                                                       

##Node2Vec Algorithm is a 2-Step representation learing algorithm...                                                                                                                     
    ##1) Use second-order random walks to generate sentences from a graph.                                                                                                               
        #A sentence is a list of node ids. The set of all sentences makes a corpus.                                                                                                      

    #2) The corpus is then used to learn an embedding vector for each node in the graph.                                                                                                 
        #Each node id is considered a unique word/token in a dictionary that has size equal to the number of nodes                                                                       
        #in the graph. The Word2Vec algorithm [2] is used for calculating the embedding vectors, like fastText                                                                           

#p=2.0 defines (unormalised) probability, 1/q, for moving away from source node                                                                                                          
#q=0.5,  # Defines (unormalised) probability, 1/q, for moving away from source node

#using the faster pecanpy implementation https://pecanpy.readthedocs.io/en/latest/index.html

import gensim
from gensim.models import Word2Vec
import typing_extensions
import numba
from numba import njit
from numba import prange
from numba_progress import ProgressBar

embedding_filename = "en_embedding_pecanpy"
embedding_model_filename = "en_embedding_model_pecanpy"
# load graph object using SparseOTF mode
g = pecanpy.SparseOTF(p=2, q=0.5, workers=1, verbose=True)
g.read_edg("./processed.edgelist", delimiter = " ", weighted=True, directed=False)

embedding_dim = 300
walk_length = 40
num_walks = 10
window = 5

cues = load_cue_words(lang="en", project_dir="./")
print(cues)
#cue_indices = [wi[cue] for cue in cues if wi[cue] is not None]


cue_indices = []
for cue in cues:
    try:
        index = wi[cue]
        cue_indices.append(index)
    except KeyError:
        pass  # Skip this cue if it doesn't exist in the dictionary

print("length of cue indices: ", len(cue_indices))
print(cue_indices)


walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length)
#walks = g.simulate_walks_from_nodes(num_walks=num_walks, walk_length=walk_length, start_nodes=valid_cue_indices)

# use random walks to train embeddings
w2v_model = Word2Vec(walks, vector_size=embedding_dim, window=window, min_count=word_fq_thresh, sg=1, workers=1, epochs=5)
emb_vocab = w2v_model.build_vocab(walks,update=True) #here
print(emb_vocab)
shape = w2v_model.wv.vectors.shape
print(f"embeddings shape: {shape}")
w2v_model.wv.save_word2vec_format(embedding_filename)
w2v_model.save(embedding_model_filename)

#----------------- find top 5 most similar --------------------#                                                                                              
with open("svd_ppmi_embeddings_vocab_test.pkl", "rb") as f:
    iw = pickle.load(f)
    
#cues = load_cue_words(lang="en", project_dir="./")
#print(cues)
#cue_indices = [iw[cue] for cue in cues]
#print("cue indices: ", cue_indices)
similar_words_dict = {}

for cue in cues:
    try:
        similar_words = w2v_model.wv.most_similar(wi[cue], topn=5)
        similar_words_dict[cue] = [(word, score) for word, score in similar_words]
    except KeyError:
        print(f"Cue word '{cue}' not found in the vocabulary.")

for cue_word, similar_words in similar_words_dict.items():
    print(f"Top 5 similar words for '{cue}':")
    for word, score in similar_words:
        print(f"Word: {word}, Similarity Score: {score}")

# Convert the results to a pandas DataFrame
results_df = pd.DataFrame([(cue, word, score) for cue, words_scores in similar_words_dict.items() for word, score in words_scores],
                          columns=['Cue', 'Similar Word', 'Similarity Score'])

# Save the results to a CSV file
results_df.to_csv('similarity_results_ppmiRWmodel.csv', index=False)
print(f"kNN search results have been saved to the current folder as similarity_results_ppmiRWmodel.csv")
