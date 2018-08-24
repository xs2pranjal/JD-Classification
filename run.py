import os
import pandas as pd
from config import DATA_PATH
from sklearn.utils import shuffle
from src.cnn_service import Workflow_CNN
import numpy as np
from helpers.csvutility import CSVUtility
from src.nn_service import WorkflowNN

class Task():
    def __init__(self, input_df, target_col, text_col, embeddings):
        self.input_df = input_df
        self.target_col = target_col
        self.text_col = text_col
        self.processed_csv = self.__create_processed_csv()
        self.embeddings = embeddings
        self.result = ""

    def __create_processed_csv(self):

# input_df = pd.read_csv(os.path.join(DATA_PATH, 'document_departments.csv'))
        processed_csv =  CSVUtility(input_df=self.input_df,
                                    department_column='Department',
                                    doc_id_column= 'Document ID').generate_processed_csv()

        processed_csv = shuffle(processed_csv)
        processed_csv = processed_csv.dropna()

        return processed_csv

    def apply_nn(self):
        print ("Initiating Neural network...")
        self.result += WorkflowNN(self.processed_csv, self.target_col, self.text_col).initiate()

    def apply_cnn(self):
        print ("Initiating Convolution Neural Network with pre-trained embeddings...")
        self.result += Workflow_CNN(self.processed_csv, self.target_col, self.text_col, embeddings=self.embeddings).initiate()

    def get_result(self):
        return self.result

if __name__ == "__main__":

    print ("Loading GloVe Embeddings...")

    embeddings_index = {}
    f = open(os.path.join(DATA_PATH, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    input_df = pd.read_csv(os.path.join(DATA_PATH, 'document_departments.csv'))

    task = Task(input_df = input_df,
                target_col = 'Department',
                text_col = 'description',
                embeddings = embeddings_index)

    task.apply_nn()
    task.apply_cnn()

    print (task.get_result())