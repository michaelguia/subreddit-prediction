import pandas as pd 
from src.preprocessing import Preprocessing
from src.baseline import Baseline
from src.modelling import Modelling
from src.lstm import Lstm

def main():
    global x_train,x_test,y_train,y_test
    p = Preprocessing([2010,2011,2012,2013,2014])
    #p.copy_from_cloud([2010,2011,2012,2013,2014])
    df = p.load_dataframe([2010,2011,2012,2013,2014])
    x_train,x_test,y_train,y_test = p.subset_data(df)

    global x_train_bl,x_test_bl,y_train_bl,y_test_bl
    b = Baseline(x_train,x_test,y_train,y_test)
    x_train_bl,x_test_bl,y_train_bl,y_test_bl = b.transform_baseline(x_train,x_test,y_train,y_test)
    b.run_baseline(x_train_bl, x_test_bl, y_train_bl, y_test_bl)

    l = Lstm(x_train,x_test,y_train,y_test)
    l.run_lstm(x_train,x_test,y_train,y_test)

    global x_train_enc,x_test_enc,y_matrix,input_layer,bagofwords_layer,num_subreddit_classes
    m = Modelling(x_train,x_test,y_train,y_test,x_train_enc=None,x_test_enc=None,y_matrix=None,input_layer=None,bagofwords_layer=None,num_subreddit_classes=None)
    x_train_enc, x_test_enc, input_layer, bagofwords_layer = m.create_layers(x_train,x_test)
    y_matrix, y_train, y_test, num_subreddit_classes = m.y_transform(y_train,y_test)
    m.run_model(x_train_enc, x_test_enc, y_matrix, y_train, y_test, input_layer, bagofwords_layer, num_subreddit_classes)

if __name__ == "__main__":
    main()
