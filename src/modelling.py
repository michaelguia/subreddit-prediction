import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from keras.models import Model, Input
from keras.layers import Flatten, Dense, Dropout, Embedding, Activation, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import keras.utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.metrics import top_k_categorical_accuracy
from src.preprocessing import Preprocessing

class Modelling(object):

    def __init__(self,x_train,x_test,y_train,y_test,x_train_enc,x_test_enc,y_matrix,input_layer,bagofwords_layer,num_subreddit_classes):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_train_enc = x_train_enc
        self.x_test_enc = x_test_enc
        self.y_matrix = y_matrix
        self.input_layer = input_layer
        self.bagofwords_layer = bagofwords_layer
        self.num_subreddit_classes = num_subreddit_classes


    def create_layers(self,x_train,x_test):
        max_num_tokens = 7000
        maxlen = 200
        wnl = WordNetLemmatizer()
        stem = SnowballStemmer('english')
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        for row in range(len(x_train)):
            x_train[row] = stem.stem(x_train[row])
            x_train[row] = wnl.lemmatize(x_train[row])
        for row in range(len(x_test)):
            x_test[row] = stem.stem(x_test[row])
            x_test[row] = wnl.lemmatize(x_test[row])
        tokenizer = Tokenizer(num_words=max_num_tokens,filters='')
        tokenizer.fit_on_texts(x_train)
        vocab_size = len(tokenizer.word_index)
        actual_num_tokens = min(max_num_tokens,vocab_size)
        print(actual_num_tokens)
        print(vocab_size)
        x_train_enc = tokenizer.texts_to_sequences(x_train)
        x_test_enc = tokenizer.texts_to_sequences(x_test)
        x_train_enc = pad_sequences(x_train_enc,maxlen=maxlen,padding='post')
        x_test_enc = pad_sequences(x_test_enc,maxlen=maxlen,padding='post')
        print(x_train_enc.shape)
        inverted_dict = dict([[v,k] for k,v in tokenizer.word_index.items()])
        def embedding_dims(num_tokens, k=2):
            return np.ceil(k * (num_tokens**0.9)).astype(int)
        num_embedding_dims = embedding_dims(actual_num_tokens,2)
        input_layer = Input(shape=(x_train_enc.shape[1],),name='input_layer')
        embedding_layer = Embedding(
            input_dim = (actual_num_tokens + 1),
            output_dim = num_embedding_dims,
            input_length = maxlen,
            )(input_layer)
        #lstm_layer = LSTM(64,return_sequences=True)(embedding_layer)
        bagofwords_layer = Flatten()(embedding_layer)
        return x_train_enc, x_test_enc, input_layer,bagofwords_layer


    def y_transform(self,y_train,y_test):
        label = LabelEncoder()
        y = y_train.append(y_test)
        #print(y.shape)
        y_matrix = pd.get_dummies(y)
        #print(y_matrix.columns)
        y_train = y_matrix[:len(y_train)]
        y_test = y_matrix[len(y_train):]
        num_subreddit_classes = y_matrix.shape[1]
        #print(num_subreddit_classes)
        return y_matrix, y_train, y_test, num_subreddit_classes


    def run_model(self,x_train_enc, x_test_enc, y_matrix, y_train, y_test, input_layer, bagofwords_layer, num_subreddit_classes):
        BATCH_SIZE = 10
        VALIDATION_SPLIT = 0.1
        current_epochs = 10

        hidden_1 = Dense(64,activation='relu')(bagofwords_layer)
        dropout_1 = Dropout(0.8)(hidden_1)
        hidden_2 = Dense(64,activation='relu')(dropout_1)
        dropout_2 = Dropout(0.8)(hidden_2)
        output_layer = Dense(num_subreddit_classes, activation='softmax', name='subreddit_output')(dropout_2)
        model = Model(inputs=input_layer, outputs=output_layer)

        metrics=['accuracy','top_k_categorical_accuracy']

        model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          # loss_weights=[1., 0.2]
                          metrics=metrics)

        history = model.fit(x_train_enc, y_train,
                    epochs=current_epochs, batch_size=BATCH_SIZE,
                    verbose=2, validation_split=VALIDATION_SPLIT
                    )

        score = model.evaluate(x_test_enc, y_test, batch_size=BATCH_SIZE, verbose=0)
        classes = model.predict(x_test_enc,verbose=1)
        classes = pd.DataFrame(classes,columns=y_matrix.columns)
        for row in range(len(classes)):
            print(classes.iloc[row].sort_values(ascending=False)[:5])
        print('Test data loss: %.3f; top 1 accuracy: %.3f; top %d accuracy: %.3f;'%(score[0],score[1], 5, score[2]))
