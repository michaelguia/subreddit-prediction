
![Reddit](https://upload.wikimedia.org/wikipedia/en/8/82/Reddit_logo_and_wordmark.svg)

# Reddit Community Detection 

## Capstone Project for Galvanize Data Science Immersive 

### Project Goals 

The goal of this project is to provide an accurate estimate of which subreddit a particular post originates. This would be used to not only understand the nature of the communities but also to provide valuable feedback to the user based on similar communities they might be interested in. 

The timeline for this project was condensed so there were several milestones to accomplish in the short amount of time. First, the data had to be acquired and loaded into memory. Second, the model had to created and tested on sample data before being productionized. Finally, a demonstration that the model answers the business question of community detection or more specifically, subreddit prediction. 

### Methodology 

#### Data Acquisition 

Finding the data was the most challenging aspect of this project. Fortunately, using Google Cloud Storage and BigQuery data pipelines I was able to copy the entire Reddit dataset to a remote cluster of cloud computing resources. From this data lake, I extracted small samples of data to perform data transformations locally, for the sake of agile development. I also experimented with Google Dataflow pipelines to wrangle terabytes of data. Installation of the Google Cloud SDK to interface with the Google Cloud Platform is required to access the data. See the documentation to learn [how to install gsutil](https://cloud.google.com/storage/docs/gsutil_install "How to install Google Storage utility").



#### Feature Embedding 

After selecting sample data to prepare for fitting the machine learning model, the next step in the process was to turn text data into features for the neural network to analyze. 

The first step was to tokenize the data. This assigns a index to each unique word in the dataset. This is how the machine learning algorithm can understand the complexities of language. Then I found the total size of the unique words in the dataset, this became assigned to the variable vocab_size. If the vocab_size exceeds the maximum number of tokens then a limit is assigned. After creating this dictionary of words and indices, the tokenizer transforms the text data int sequences of tokens.

The next step is to normalize these sequences of tokens in a way that the neural network can easiy parse through their stucture. There's a special pre-built function that I utilized called pad_sequences that assigns a standard length for each of the rows in the dataset. I call the resulting output x_train_enc and x_test_enc. These are very valuable and used later in the pipeline. 

#### Neural Network 

The final part of the pipeline was to fit and predict the machine learning model, in this case a multi-layer neural network. 

The input layer was designed to fit the shape of the output of the feature embedding matrix x_train_enc. It was then fed to an embedding layer that used the maximum number of tokens as the input dimensions and the output dimensions are defined as a ratio of the input dimensions. This embedding layer is then fed to a Flatten layer which is essentially just a collection of the vocabulary. It's called the bagofwords_layer in the code. 

Additionally, the target labels are given some treatment to transform them into a one-hot encoded matrix for the loss function to evaluate. I made the choice to limit the number of unique targets to the top 20 subreddits, due to the long tail class imbalance problem that plagued me in early stages of this project. 

Finally, the bagofwords_layer is fed to a few hidden layers with some special regularization techniques. Dropout layers reduced overfitting, however there's still room for improvement. 

The model uses Adam optimizer and top_k_categorical_accuracy to perform gradient descent and evaluate performance.


### Results 

#### Evaluation Metrics

Accuracy and Top K Accuracy 

|                | Baseline | Dense Net| LSTM | 
| -------------  | -----:| -----:| -----: |
| Top 1 accuracy | 0.311 |0.684 | 0.510 | 
| Top 5 accuracy | NaN | 0.789 |   0.857 | 


#### Predictions 

Example of the output: 


|  Subreddit Prediction    | Probability |
| -------------  | -----:|
|worldnews   | 0.250197|
|gaming      | 0.180619|
|politics    | 0.119173|
|pics        | 0.075450|
|videos      | 0.072769|
