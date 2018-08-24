from keras import models, layers
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(self, num_class):

        self.model = models.Sequential()
        self.model.add(layers.Dense(48, activation='relu', input_shape=(10000,)))
        self.model.add(layers.Dense(48, activation='relu'))
        self.model.add(layers.Dense(num_class, activation='softmax'))

        self.model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])


    def train(self, train_x, train_y, epoch = 30, batch_size = 512):

        history = self.model.fit(train_x[100:], train_y[100:],
                                 epochs = epoch,
                                 batch_size = batch_size,
                                 validation_data = (train_x[:100], train_y[:100]))

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
        return ("\nNN Test_metrics: \n  Accuracy: {} \n  Loss: {}".format(test_metrics[1], test_metrics[0]))