# !pip install nltk
# !pip install transformers==4.42.1
# !pip install sentencepiece
# !pip install spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy download de_core_news_sm
# !pip install torch==2.2.2 torchtext==0.17.2
# !pip install torchdata==0.7.1
# !pip install portalocker
# !pip install numpy pandas
# !pip install numpy scikit-learn

import torchtext
print(torchtext.__version__)

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper
import torchtext

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

# Dataset:
# Stores the training samples and labels.
# It is responsible for returning individual samples when requested.
#
# Example:
# dataset[0] -> ("This movie is great", "Positive")


# DataLoader:
# Loads data from the Dataset and groups samples into batches.
# It also supports shuffling, batching, and efficient memory usage.
#
# Example:
# DataLoader(dataset, batch_size=2, shuffle=True)


# batch_size:
# Determines how many samples are processed together in a single batch.
#
# Example:
# Dataset: [A, B, C, D, E, F]
# batch_size=2
#
# Batch 1 -> [A, B]
# Batch 2 -> [C, D]
# Batch 3 -> [E, F]
#
# The model processes one batch at a time instead of the entire dataset.


# shuffle=True:
# Randomly rearranges the dataset before creating batches at the start
# of each epoch.
#
# Without shuffle:
# Epoch 1 -> [A, B, C, D, E, F]
# Epoch 2 -> [A, B, C, D, E, F]
#
# With shuffle:
# Epoch 1 -> [D, A, F, B, E, C]
# Epoch 2 -> [B, E, A, F, C, D]
#
# Shuffling helps the model learn general patterns rather than
# memorizing the order of the training data.


# Iterator:
# A DataLoader is an iterator that returns one batch at a time.
#
# Example:
# for batch in dataloader:
#     model(batch)
#
# Batch 1 is loaded and processed.
# Then Batch 2 is loaded and processed.
# This avoids loading the entire dataset into memory at once.


# NLP preprocessing:
# The DataLoader can be used with custom functions to:
# - Tokenize text
# - Convert tokens to numerical IDs
# - Pad sequences to the same length
# - Convert data into tensors
#
# This ensures all samples in a batch have the same shape and can be
# processed by the neural network.

# Dataset   = Stores the data
# DataLoader = Creates batches from the data
# shuffle    = Randomizes the order of samples
# batch_size = Number of samples per batch
# Iterator   = Delivers one batch at a time

# Custom Dataset and DataLoader in PyTorch
#
# This example demonstrates how to create a custom Dataset and use a
# DataLoader to load data in batches for training a deep learning model.
#
# The dataset consists of a list of sentences. The goal is to organize
# these sentences into batches so they can be processed efficiently by
# a neural network.
#
# CustomDataset:
# - Inherits from torch.utils.data.Dataset.
# - Stores the input sentences.
# - Defines how individual samples are accessed.
#
# Key methods:
#
# __init__(self, sentences)
# - Runs when the dataset object is created.
# - Stores the list of sentences in the dataset.
#
# __getitem__(self, idx)
# - Returns a single sentence at the specified index.
# - Used by the DataLoader to retrieve samples.
#
# Example:
# dataset[0] -> "I love PyTorch"
#
# After creating the dataset, a DataLoader is used to:
# - Read data from the dataset.
# - Group samples into batches.
# - Optionally shuffle the data.
#
# batch_size:
# - Specifies the number of samples in each batch.
# - If batch_size=2, each batch contains 2 sentences.
#
# Example:
# Dataset: [A, B, C, D]
#
# Batch 1 -> [A, B]
# Batch 2 -> [C, D]
#
# shuffle=True:
# - Randomly rearranges the dataset before creating batches.
# - Helps prevent the model from learning patterns based on the
#   original order of the data.
#
# Example:
# Original order:
# [A, B, C, D]
#
# After shuffling:
# [C, A, D, B]
#
# Iterating through the DataLoader:
#
# for batch in dataloader:
#     print(batch)
#
# The DataLoader:
# 1. Retrieves samples using __getitem__().
# 2. Groups samples into batches.
# 3. Returns one batch at a time.
#
# Example with batch_size=2:
#
# Batch 1:
# ["I love PyTorch", "Deep learning is fun"]
#
# Batch 2:
# ["Python is awesome", "Neural networks learn patterns"]
#
# DataLoader acts as an iterator, meaning it provides one batch at a
# time until all samples have been processed.
#
# Benefits of using a DataLoader:
# - Automatic batching
# - Data shuffling
# - Efficient memory usage
# - Simplified training loops
#
# This Dataset + DataLoader pattern is commonly used in PyTorch-based
# machine learning and deep learning workflows.


# Custom data set and data loader in PyTorch
sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

# Create an instance of your custom dataset
custom_dataset = CustomDataset(sentences)

# Define batch size
batch_size = 2

# Create a DataLoader
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for batch in dataloader:
    print(batch)

# Creating tensors for custom data set
sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Define a custom data set

# init: The constructor takes a list of sentences, a tokenizer function, and a vocabulary (vocab) as input.
# len: This method returns the total number of samples in the data set.
# getitem: This method is responsible for processing a single sample.
# It tokenizes the sentence using the provided tokenizer and then converts the tokens into tensor indices using the vocabulary.

# Sentence
#    ↓
# Tokenizer
#    ↓
# ["i", "love", "pytorch"]
#    ↓
# Vocabulary
#    ↓
# [0, 1, 2]
#    ↓
# Tensor
#    ↓
# tensor([0, 1, 2])

class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx])
        # Convert tokens to tensor indices using vocab
        tensor_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(tensor_indices)

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# Create an instance of your custom data set
custom_dataset = CustomDataset(sentences, tokenizer, vocab)

print("Custom Dataset Length:", len(custom_dataset))
print("Sample Items:")
for i in range(6):
    sample_item = custom_dataset[i]
    print(f"Item {i + 1}: {sample_item}")

# Custom Dataset Length: 6
# Sample Items:
# Item 1: tensor([11, 19, 63, 17, 13,  2,  3, 47,  6, 16, 45,  0, 55,  3, 41, 46, 24, 10,
#         43, 61,  9, 44,  0, 14,  9, 33,  1])
# Item 2: tensor([35,  6, 16,  3, 38, 40,  0,  8,  1])
# Item 3: tensor([12,  5, 15, 31,  0,  8,  0, 57, 53,  2, 18, 62,  4,  0, 36, 49, 56, 15,
#         21,  1])
# Item 4: tensor([54, 18, 50, 23, 34, 58, 30, 27,  2,  5, 52,  7,  2,  5, 32,  1])
# Item 5: tensor([66, 29, 14, 13, 10, 22, 60,  7, 37,  1, 28, 51, 48,  4, 42, 11, 59, 39,
#          2, 12, 64, 17, 26, 65,  1])
# Item 6: tensor([19,  4, 25, 20])

"""
# Create an instance of your custom data set
custom_dataset = CustomDataset(sentences, tokenizer, vocab)

# Define batch size
batch_size = 2

# Create a data loader
#dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the data loader
for batch in dataloader:
    print(batch)
"""

# Dataset
#     ↓
# tensor([10,20,30])
#
# Dataset
#     ↓
# tensor([40,50,60,70,80])
#
# DataLoader
#     ↓
# collects both samples
#     ↓
# calls collate_fn(batch)
#     ↓
# pad_sequence()
#     ↓
# tensor([
#  [10,20,30,0,0],
#  [40,50,60,70,80]
# ])

# Create a custom collate function
def collate_fn(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch


# Create a data loader with the custom collate function with batch_first=True,
dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Iterate through the data loader
for batch in dataloader:
    for row in batch:
        for idx in row:
            words = [vocab.get_itos()[idx] for idx in row]
        print(words)

# ['if', 'you', 'want', 'to', 'know', 'what', 'a', 'man', "'", 's', 'like', ',', 'take', 'a', 'good', 'look', 'at', 'how', 'he', 'treats', 'his', 'inferiors', ',', 'not', 'his', 'equals', '.']
# ['fame', "'", 's', 'a', 'fickle', 'friend', ',', 'harry', '.', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',']
# ['it', 'is', 'our', 'choices', ',', 'harry', ',', 'that', 'show', 'what', 'we', 'truly', 'are', ',', 'far', 'more', 'than', 'our', 'abilities', '.']
# ['soon', 'we', 'must', 'all', 'face', 'the', 'choice', 'between', 'what', 'is', 'right', 'and', 'what', 'is', 'easy', '.', ',', ',', ',', ',']
# ['youth', 'can', 'not', 'know', 'how', 'age', 'thinks', 'and', 'feels', '.', 'but', 'old', 'men', 'are', 'guilty', 'if', 'they', 'forget', 'what', 'it', 'was', 'to', 'be', 'young', '.']
# ['you', 'are', 'awesome', '!', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',']

# Sentence
# ↓
# "I love pytorch"
#
# Tokenizer
# ↓
# ["i", "love", "pytorch"]
#
# Vocabulary
# ↓
# [1, 2, 3]
#
# Tensor
# ↓
# tensor([1, 2, 3])
#
# DataLoader
# ↓
# Batch:
# [
#  tensor([1,2,3]),
#  tensor([4,5,6,7,8])
# ]
#
# collate_fn
# ↓
# Padding
#
# tensor([
#  [1,2,3,0,0],
#  [4,5,6,7,8]
# ])
#
# Convert IDs back to words
# ↓
# ['i', 'love', 'pytorch', '<pad>', '<pad>']
# ['deep', 'learning', 'is', 'really', 'fun']


# Create a custom collate function
def collate_fn_bfFALSE(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, padding_value=0)
    return padded_batch



# Create a data loader with the custom collate function with batch_first=True,
dataloader_bfFALSE = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn_bfFALSE)

# Iterate through the data loader
for seq in dataloader_bfFALSE:
    for row in seq:
        #print(row)
        words = [vocab.get_itos()[idx] for idx in row]
        print(words)

# Iterate through the data loader with batch_first = TRUE
for batch in dataloader:
    print(batch)
    print("Length of sequences in the batch:",batch.shape[1])

# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

custom_dataset=CustomDataset(sentences)
custom_dataset[0]


def collate_fn(batch):
    # Tokenize each sample in the batch using the specified tokenizer
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        # Convert tokens to vocabulary indices and create a tensor for each sample
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    # Pad sequences within the batch to have equal lengths using pad_sequence
    # batch_first=True ensures that the tensors have shape (batch_size, max_sequence_length)
    padded_batch = pad_sequence(tensor_batch, batch_first=True)

    # Return the padded batch
    return padded_batch

# Create a data loader for the custom dataset
dataloader = DataLoader(
    dataset=custom_dataset,   # Custom PyTorch Dataset containing your data
    batch_size=batch_size,     # Number of samples in each mini-batch
    shuffle=True,              # Shuffle the data at the beginning of each epoch
    collate_fn=collate_fn      # Custom collate function for processing batches
)

for batch in dataloader:
    print(batch)
    print("shape of sample",len(batch))

# Custom collate function for French text preprocessing.
#
# Processing pipeline:
# 1. Sort sequences by length to reduce padding.
# 2. Tokenize each sentence.
# 3. Convert tokens to vocabulary indices.
# 4. Pad shorter sequences with <PAD> tokens.
# 5. Return batches of size 4 as uniform tensors.
#
# Sorting by sequence length minimizes the number of padding tokens
# added to each batch, improving computational efficiency and model
# performance.

# !python -m spacy download fr_core_news_sm
corpus = [
    "Ceci est une phrase.",
    "C'est un autre exemple de phrase.",
    "Voici une troisième phrase.",
    "Il fait beau aujourd'hui.",
    "J'aime beaucoup la cuisine française.",
    "Quel est ton plat préféré ?",
    "Je t'adore.",
    "Bon appétit !",
    "Je suis en train d'apprendre le français.",
    "Nous devons partir tôt demain matin.",
    "Je suis heureux.",
    "Le film était vraiment captivant !",
    "Je suis là.",
    "Je ne sais pas.",
    "Je suis fatigué après une longue journée de travail.",
    "Est-ce que tu as des projets pour le week-end ?",
    "Je vais chez le médecin cet après-midi.",
    "La musique adoucit les mœurs.",
    "Je dois acheter du pain et du lait.",
    "Il y a beaucoup de monde dans cette ville.",
    "Merci beaucoup !",
    "Au revoir !",
    "Je suis ravi de vous rencontrer enfin !",
    "Les vacances sont toujours trop courtes.",
    "Je suis en retard.",
    "Félicitations pour ton nouveau travail !",
    "Je suis désolé, je ne peux pas venir à la réunion.",
    "À quelle heure est le prochain train ?",
    "Bonjour !",
    "C'est génial !"
]


def collate_fn_fr(batch):
    # Pad sequences within the batch to have equal lengths
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    padded_batch = pad_sequence(tensor_batch, batch_first=True)
    return padded_batch


# Build tokenizer
tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, corpus))

# Sort sentences based on their length
sorted_data = sorted(corpus, key=lambda x: len(tokenizer(x)))
# print(sorted_data)
dataloader = DataLoader(sorted_data, batch_size=4, shuffle=False, collate_fn=collate_fn_fr)

for batch in dataloader:
    print(batch)


# Data preparation for German-English machine translation.
#
# Steps:
# - Configure the Multi30k dataset and define source (German) and
#   target (English) languages.
# - Create spaCy tokenizers for both languages.
# - Generate tokens from the dataset using a helper function.
# - Define special tokens such as <unk> and <pad>.
# - Build vocabularies that map tokens to unique numerical indices.
# - Set a default <unk> index for words not present in the vocabulary.
#
# The resulting vocabularies and tokenizers are used to convert
# German and English text into numerical representations that can
# be processed by a translation model.

# Translation data set

multi30k.URL["train"] = ""
multi30k.URL["valid"] = ""

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Initialize the training data iterator for the Multi30k dataset with the specified source and target languages:
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

# Create an iterator for the training data set:
data_set = iter(train_iter)

# print out the first five pairs of source and target sentences from the training data set:
for n in range(5):
    # Getting the next pair of source and target sentences from the training data set
    src, tgt = next(data_set)

    # Printing the source (German) and target (English) sentences
    print(f"sample {str(n+1)}")
    print(f"Source ({SRC_LANGUAGE}): {src}\nTarget ({TGT_LANGUAGE}): {tgt}")

# Tokenizer setup
german, english = next(data_set)
print(f"Source German ({SRC_LANGUAGE}): {german}\nTarget English  ({TGT_LANGUAGE}): { english }")

# Initialize the German and English tokenizers using spaCy's 'de_core_news_sm' model:
# Making a placeholder dict to store both tokenizers
token_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

token_transform['de'](german)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


# Tokens to indices transformation (Vocab)
#place holder dict for 'en' and 'de' vocab transforms
vocab_transform = {}

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    # Define a mapping to associate the source and target languages
    # with their respective positions in the data samples.
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    # Iterate over each data sample in the provided dataset iterator
    for data_sample in data_iter:
        # Tokenize the data sample corresponding to the specified language
        # and yield the resulting tokens.
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data iterator
    train_iterator = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    #To decrease the number of padding tokens, you sort data on the source length to batch similar-length sequences together
    sorted_dataset = sorted(train_iterator, key=lambda x: len(x[0].split()))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(sorted_dataset, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

# English/German text string, tokenize it into words or subwords,
# and then convert these tokens into their corresponding indices from the vocabulary,
# resulting in a sequence of integers seq_en that can be used for further processing in a model.
seq_en=vocab_transform['en'](token_transform['en'](english))
print(f"English text string: {english}\n English sequence: {seq_en}")

seq_de=vocab_transform['de'](token_transform['de'](german))
print(f"German text string: {german}\n German sequence: {seq_de}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tensor_transform_s function adds a beginning-of-sequence (BOS) token at the start,
# flips the sequence to reverse the order of token IDs and adds an end-of-sequence (EOS)
# token at the end of a given sequence of token IDs, then returns the concatenated result as a PyTorch tensor,
# this will be used as an input to our model:
#
# tensor_transform_t function does the similar operations except the flip operation.
# It is a good practice to reverse the order of source sentence in order for the LSTM to perform better.

# tensor_transform_s (source sentence)
# - Adds a BOS token at the beginning.
# - Reverses the order of token IDs.
# - Adds an EOS token at the end.
#
# Example:
# [10, 20, 30]
# -> [BOS, 30, 20, 10, EOS]
#
# Reversing source sequences is a common technique used in
# LSTM-based Seq2Seq models to improve learning.

# tensor_transform_t (target sentence)
# - Adds a BOS token at the beginning.
# - Keeps the original token order.
# - Adds an EOS token at the end.
#
# Example:
# [10, 20, 30]
# -> [BOS, 10, 20, 30, EOS]
#
# BOS marks the start of the sequence and EOS marks the end,
# helping the decoder know when to start and stop generating text.

seq_en=tensor_transform_s(seq_en)
seq_en

seq_de=tensor_transform_t(seq_de)
seq_de

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}

text_transform[SRC_LANGUAGE] = sequential_transforms(token_transform[SRC_LANGUAGE], #Tokenization
                                            vocab_transform[SRC_LANGUAGE], #Numericalization
                                            tensor_transform_s) # Add BOS/EOS and create tensor

text_transform[TGT_LANGUAGE] = sequential_transforms(token_transform[TGT_LANGUAGE], #Tokenization
                                            vocab_transform[TGT_LANGUAGE], #Numericalization
                                            tensor_transform_t) # Add BOS/EOS and create tensor

# Processing data in batches
# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_sequences = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
        src_sequences = torch.tensor(src_sequences, dtype=torch.int64)
        tgt_sequences = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
        tgt_sequences = torch.tensor(tgt_sequences, dtype=torch.int64)
        src_batch.append(src_sequences)
        tgt_batch.append(tgt_sequences)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)

    return src_batch.to(device), tgt_batch.to(device)

BATCH_SIZE = 4

train_iterator = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
sorted_train_iterator = sorted(train_iterator, key=lambda x: len(x[0].split()))
train_dataloader = DataLoader(sorted_train_iterator, batch_size=BATCH_SIZE, collate_fn=collate_fn,drop_last=True)

valid_iterator = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
sorted_valid_dataloader = sorted(valid_iterator, key=lambda x: len(x[0].split()))
valid_dataloader = DataLoader(sorted_valid_dataloader, batch_size=BATCH_SIZE, collate_fn=collate_fn,drop_last=True)


src, trg = next(iter(train_dataloader))
src,trg
