
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from keras.layers import Activation,Dropout,Embedding,Flatten,Input,LSTM,Dense,SpatialDropout1D
from keras.metrics import categorical_crossentropy,top_k_categorical_accuracy
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from src.preprocessing import Preprocessing


class Lstm(object):
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def run_lstm(self, x_train,x_test,y_train,y_test):

        tokenizer = Tokenizer(num_words=2500, lower=True,split=' ')
        tokenizer.fit_on_texts((x_train.append(x_test)).values)
        #print(tokenizer.word_index)  # To see the dictionary
        X = tokenizer.texts_to_sequences((x_train.append(x_test)).values)
        X = pad_sequences(X,maxlen=200,padding='post')

        y = y_train.append(y_test)
        Y = pd.get_dummies(y).values
        print(Y.shape)
        X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)
        
        embed_dim = 128
        lstm_out = 200
        batch_size = 20
        
        model = Sequential()
        model.add(Embedding(2500, embed_dim,input_length = X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(Y.shape[1],activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy','top_k_categorical_accuracy'])
        print(model.summary())

        history = model.fit(X_train, Y_train, batch_size =batch_size, epochs = 10,  verbose = 2)

        score = model.evaluate(X_valid, Y_valid, batch_size=batch_size, verbose=0)
        classes = model.predict(X_valid,verbose=1)

        print('Test data loss: %.3f; top 1 accuracy: %.3f; top %d accuracy: %.3f;'%(score[0],score[1], 5, score[2]))
