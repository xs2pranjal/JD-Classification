## JD Classification

### Your Task

Make a classifier which takes in a job description and gives the department name for it.

*   Use a neural network model
*   Make use of a pre-trained Word Embeddings (example: Word2Vec, GloVe, etc.)
*   Calculate the accuracy on a test set (data not used to train the model)

### Getting Started

All the data required can be found in tha data folder in the root of the project

To run the program start with using the bash file to download the pre-trained glove embeddings using,
```bash
./download_data.sh
```

after downloading you can run the program by the following commands
```bash
python run.py
```

### Approaches

The following application consists of two approaches

*   Use of Normal Neural Network to classify
*   Use of Convolution Neural Network with a pre-trained word embedding GloVe