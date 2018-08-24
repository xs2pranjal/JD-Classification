from helpers.preprocess import Preprocess
from helpers.neural_network import NeuralNetwork


class WorkflowNN():

    def __init__(self, input_df, target_col, text_col):
        self.input_df = input_df
        self.target_col = target_col
        self.text_col = text_col

    def initiate(self):

        target_encoder, target_encoded = Preprocess().LabelEncode(self.input_df, self.target_col)
        self.input_df = Preprocess().clean_text(self.input_df, self.text_col)

        sequences, tokenizer = Preprocess().Tokenize(self.input_df, self.text_col, max_words= 5000)

        one_hot_sequences = Preprocess().vectorize_sequences(sequences, 10000)

        train_index = int(one_hot_sequences.shape[0] * 0.8)

        train_x = one_hot_sequences[:train_index]
        train_y = target_encoded[:train_index]

        test_x = one_hot_sequences[train_index:]
        test_y = target_encoded[train_index:]

        model = NeuralNetwork(num_class = self.input_df[self.target_col].unique().shape[0])
        model.train(train_x, train_y)

        return model.eval(test_x, test_y)