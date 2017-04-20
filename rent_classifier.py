from scipy import sparse
import xgboost as xgb
import random, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import loading_data, preprocessing

def gradient_boost(x_train, y_train, x_test):
    gradient_boost_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
    gradient_boost_model.fit(x_train, y_train)
    # predict_proba() yeileds an array of class probabilities: (0.2 0,2 0.6)
    print (gradient_boost_model.predict_proba(x_test))
    return gradient_boost_model.predict_proba(x_test)
    
def rent_interest_classifier():
    y_train = train['interest_level']
    x_train = numerical_features
    x_test = processed_test_data
    y_train_copy=y_train.copy()
    diction = {'high':1,'medium':2,'low':3}
    y_train1 = map(lambda x: diction[x], y_train)
    y_train = pd.Series(y_train1, index=y_train.index)
    gradient_boost(x_train, y_train, x_test)

if __name__ == '__main__':
    train = pd.read_json("/Users/soyoungkim/Desktop/python_codes/two-sigma/data/train.json")
    test = pd.read_json("/Users/soyoungkim/Desktop/python_codes/two-sigma/data/test.json")

    start_time = time.time()
    train['interest'] = np.where(train['interest_level']=='high', 1,
                                np.where(train['interest_level']=='medium', 2, 3))

    important_features = ['bathrooms', 'bedrooms', 'price', 'price_room','latitude',
                          'longitude', 'nb_images','nb_features', 'sentiment',
                          'nb_description', 'description_len','b_counts', 'm_counts',
                          'b_count_log', 'm_count_log']

    numerical_features = preprocessing.pre_processing(train)
    processed_test_data = preprocessing.pre_processing(test)
    print ('A set of 15 derived features:{0}\n'.format(important_features))
    preprocessing.classification(numerical_features, processed_test_data, train['interest'])
    ans = rent_interest_classifier()
    ans_dataframe = pd.DataFrame(ans,columns=['low','medium','high'], index=processed_test_data.index)
    ans_dataframe.to_csv('result.csv', index=False) 
    print ('--- %s seconds ---' % (time.time() - start_time))
