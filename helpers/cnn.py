from keras.layers import Embedding, Flatten, Dense, Dropout,Input, Conv1D, MaxPooling1D
from keras.models import Model
import matplotlib.pyplot as plt

class CNN():
    """This class is for Convolution Neural Network"""

    def __init__(self, num_class, embedding_matrix, max_words = 10000, embedding_dim = 100, maxlen = 200):

        embedding_layer = Embedding(max_words,
                                    embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=False)
        sequence_input = Input(shape=(maxlen,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        self.model = Conv1D(128, 5, activation='relu')(embedded_sequences)
        self.model = MaxPooling1D(5)(self.model)
        self.model = Conv1D(128, 5, activation='relu')(self.model)
        self.model = MaxPooling1D(5)(self.model)
        self.model = Conv1D(128, 5, activation='relu')(self.model)
        #x = MaxPooling1D(5)(x)  # global max pooling
        self.model = Flatten()(self.model)
        self.model = Dense(128, activation='relu')(self.model)
        preds = Dense(num_class, activation='softmax')(self.model)

        self.model = Model(sequence_input, preds)
        self.model.summary()

        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['acc'])

    def train(self,train_x, train_y):

        history = self.model.fit(train_x, train_y,
                                 epochs=50,
                                 batch_size=32,
                                 validation_data=(train_x, train_y))

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    def eval(self, test_x, test_y):
        test_metrics = self.model.evaluate(test_x, test_y)
        return ("\nCNN Test_metrics: \n  Accuracy: {} \n  Loss: {}".format(test_metrics[1], test_metrics[0]))