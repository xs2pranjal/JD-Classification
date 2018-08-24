import numpy as np
from helpers.preprocess import Preprocess
from keras.preprocessing.sequence import pad_sequences
from helpers.cnn import CNN

class Workflow_CNN():
    def __init__(self, input_df, target_col, text_col, embeddings):
        self.input_df = input_df
        self.target_col = target_col
        self.text_col =text_col
        self.embeddings = embeddings

    def initiate(self, max_words = 10000, maxlen = 200, embedding_dim = 100):

        target_encoder, target_encoded = Preprocess().LabelEncode(self.input_df, self.target_col)

        # self.input_df = Preprocess().clean_text(self.input_df, self.text_col)

        sequences, tokenizer = Preprocess().Tokenize(self.input_df, self.text_col, max_words)
        sequences = pad_sequences(sequences, maxlen)

        train_index = int(sequences.shape[0] * 0.8)

        train_x = sequences[:train_index]
        train_y = target_encoded[:train_index]

        test_x = sequences[train_index:]
        test_y = target_encoded[train_index:]
        word_index = tokenizer.word_index

        embedding_matrix = np.zeros((len(tokenizer.word_counts.keys())+1, embedding_dim))

        for word, i in word_index.items():
            if i < max_words:
                embedding_vector = self.embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        model = CNN(num_class = self.input_df[self.target_col].unique().shape[0],
                    embedding_matrix = embedding_matrix,
                    max_words = len(tokenizer.word_counts.keys())+1)
        model.train(train_x, train_y)

        return model.eval(test_x, test_y)