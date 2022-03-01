import tensorflow as tf
from transformers import TFBertModel, BertTokenizer,BertConfig
import re
import numpy as np


class ProtTrans(object):

  
    """ Class containig the LM models for predicting protein function. """
    def __init__(self, output_dim, n_channels=26):
        """ Initialize the model
        :param output_dim: {int} final output dimension 
        :param n_channels: {int} number of input features per residue (26 for 1-hot encoding)
        """
        self.output_dim = output_dim
        self.n_channels = n_channels

        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        lm_model = TFBertModel.from_pretrained("Rostlab/prot_bert", from_pt=True)

    def predict(self, sequences):
        sequences = [[" ".join(list(item)) for item in sequence] for sequence in sequences]
        sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]

        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True, return_tensors="tf")

        input_ids = ids['input_ids']
        attention_mask = ids['attention_mask']

        embedding = self.lm_model(input_ids)[0]
        embedding = np.asarray(embedding)
        attention_mask = np.asarray(attention_mask)

        features = [] 
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd)
            
        return features
