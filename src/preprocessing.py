import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
  
class Preprocessing(object):
  
    def __init__(self,file_list):
        self.file_list = file_list
  
    def copy_from_cloud(self,file_list):
        for file in file_list:
            os.system('gsutil cp gs://redditv03/raw/{}_000000000000.csv data/{}_000000000000.csv'.format(file,file))
 
    def load_dataframe(self,file_list):
        df = pd.read_csv('data/{}_000000000000.csv'.format(file_list[0]),nrows=1000)
        for file in file_list[1:]:
            df = df.append(pd.read_csv('data/{}_000000000000.csv'.format(file),nrows=1000))
        return df
  
    def subset_data(self,df):
        df = df[df.author!='[deleted]']
        df = df[df.body.notnull()]

        post_length = []
        for row in df.body:
            post_length.append(len(row.split(' ')))
        df['post_length'] = pd.Series(post_length)
        df = df[df.post_length>10]
        df = df.drop(['removal_reason','author_flair_css_class','author_flair_text','distinguished'],axis=1)
        df['top20bool'] = df.subreddit.map(df.subreddit.value_counts()[:20])
        df = df.dropna()
        #df2 = pd.DataFrame(columns=df.columns)
        #for sub in df.subreddit.unique():
        #    df2 = df2.append(df[df.subreddit==sub].sample(n=int(df.top20bool.min())))
        #print(df2.shape)
        df = df.sample(frac=0.5)
        print(df.shape)
        y = df.subreddit
        X = df.body
        
        x_train, x_test, y_train, y_test = train_test_split(X,y)
        return x_train, x_test, y_train, y_test
