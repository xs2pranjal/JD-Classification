from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
import numpy as np
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.utils.np_utils import to_categorical

lemmatizer = WordNetLemmatizer()
stopword = list(set(stopwords.words('english')))


class Preprocess():
    """This class is for carring out pre-processing on data"""

    @staticmethod
    def LabelEncode(input_df, column_name):
        encoder = preprocessing.LabelEncoder()
        encoder.fit(input_df[column_name].values)

        return encoder, to_categorical(encoder.transform(input_df[column_name].values))

    @staticmethod
    def Tokenize(input_df, column_name, max_words = 10000):

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(input_df[column_name].tolist())
        sequences = tokenizer.texts_to_sequences(input_df[column_name].tolist())

        return sequences, tokenizer

    @staticmethod
    def clean_text(input_df, column_name):
        input_df[column_name] = input_df[column_name].apply(lambda x: Preprocess().prun(x))

        return input_df

    @staticmethod
    def acceptable_word(word):
        accepted = bool(word.lower() not in stopword)

        return accepted

    @staticmethod
    def prun(text):
        return [Preprocess().normalise(word) for word in text.split(" ") if Preprocess().acceptable_word(text)]

    @staticmethod
    def normalise(word):
        word = word.lower()
        word = lemmatizer.lemmatize(word)

        return word


    @staticmethod
    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1

        return results