import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cartwright.utils import character_tokins
from cartwright.category_helpers import return_all_category_classes_and_labels, generate_label_id
from cartwright.LSTM import LSTMClassifier, PaddedTensorDataset

class CartwrightBase:
    def __init__(self):
        self.char_dim = 128
        self.hidden_dim = 32

        self.character_tokins = character_tokins
        self.all_characters = "".join(character_tokins.keys()).split('PADUNK')[1]
        self.char_vocab_size = len(self.character_tokins)
        self.n_characters = len(self.all_characters)

        self.all_classes = return_all_category_classes_and_labels()
        self.all_labels = np.array(list(self.all_classes.keys()))
        self.label2id = generate_label_id(self.all_labels)
        self.model = LSTMClassifier(self.char_vocab_size, self.char_dim, self.hidden_dim, len(self.label2id))

    def vectorized_string(self, string):
        return [
            self.character_tokins[token] if token in self.character_tokins else self.character_tokins["UNK"]
            for token in str(string)
        ]

    def vectorized_data(self, data, item2id):
        return [[item2id[token] if token in item2id else item2id['UNK'] for token in str(seq)] for seq, _ in data]

    def vectorized_array(self, array):
        vecorized_array = []
        for stringValue in array:
            vecorized_array.append(self.vectorized_string(str(stringValue)))
        return vecorized_array

    def pad_sequences(self, vectorized_seqs, seq_lengths):
        # create a zero matrix
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()

        # fill the index
        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        return seq_tensor

    def create_dataset(self, data, batch_size=1):
        vectorized_seqs = self.vectorized_array(data)
        seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
        seq_tensor = self.pad_sequences(vectorized_seqs, seq_lengths)
        target_tensor = torch.LongTensor([0 for y in data])
        raw_data = [x for x in data]
        return DataLoader(
            PaddedTensorDataset(seq_tensor, target_tensor, seq_lengths, raw_data),
            batch_size=batch_size,
        )

    def create_training_dataset(self, data, input2id, target2id, batch_size=4):
        vectorized_seqs = self.vectorized_data(data, input2id)
        seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
        seq_tensor = self.pad_sequences(vectorized_seqs, seq_lengths)
        target_tensor = torch.LongTensor([target2id[y] for _, y in data])
        raw_data = [x for x, _ in data]
        return DataLoader(PaddedTensorDataset(seq_tensor, target_tensor, seq_lengths, raw_data), batch_size=batch_size)



    def sort_batch(self, batch, targets, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return seq_tensor.transpose(0, 1), target_tensor, seq_lengths