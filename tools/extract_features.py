import torch
import torch.nn as nn
import numpy as np

from model.bert import BERT
from dataset import BERTDataset, WordVocab


class BERTForEmbedding:
    def __init__(self, bert, vocab_path):
        # load vocab for tokenization
        self.vocab = WordVocab.load_vocab(vocab_path)
        # load pretrained model
        self.bert = bert
        
    def extract_all_layers(self, x, segment_label):
        output = self.bert.forward(x,segment_label, output_all_encoded_layers=True)
        return output

    def get_embedding(self, text, pooling_layer=-2):
        # tokenize with vocab
        token = self.vocab.to_seq(text, with_eos=True, with_sos=True)
        segment_label = [1 for _ in range(len(token))]
        # get layers
        self.bert.eval()
        with torch.no_grad():
            all_encoded_layers = self.bert(token, segment_label, masking=False, output_all_encoded_layers=True)
        # get embedding
        embedding = all_encoded_layers[pooling_layer].numpy()[0]
        # calc mean of all words to get "sentence" embedding
        embedding = np.mean(embedding, axis=0)
        return embedding
