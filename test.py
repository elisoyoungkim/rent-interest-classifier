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
    print (gradient_boost_model.predict_proba(x_test))
    return gradient_boost_model.predict_proba(x_test)

def k_nearest_neighbor(x_train, y_train, x_test):
    k_nearest_neighbor_model = KNeighborsClassifier(n_neighbors=5, n_jobs=12)
    k_nearest_neighbor_model.fit(x_train, y_train)
    return k_nearest_neighbor_model.predict_proba(x_test)

def logistic_regression(x_train,y_train,x_test):
    logistic_regression_model= LogisticRegression(C=1e4, solver='lbfgs', multi_class='multinomial', n_jobs=12)
    logistic_regression_model.fit(x_train, y_train)
    return logistic_regression_model.predict_proba(x_test)

def random_forest(x_train,y_train,x_test):
    random_forest_model = RandomForestClassifier(n_estimators=200, n_jobs=12)
    random_forest_model.fit(x_train,y_train)
    return random_forest_model.predict_proba(x_test)

def ada_boost(x_train,y_train,x_test):
    ada_model = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)
    ada_model.fit(x_train, y_train)
    return ada_model.predict_proba(x_test)

def XGBoost(x_train, y_train, x_test):
    xgb_train = xgb.DMatrix(x_train, label=y_train)
    xgv_test = xgb.DMatrix(x_test)
    param={}
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 12
    param['num_class'] = 3
    watchlist = [ (xgb_train ,'train')]
    num_round = 20
    bst = xgb.train(param, xgb_train, num_round, watchlist )
    res= {'train':bst.predict(xgb_train),'test':bst.predict(xgb_test)}
    return res

def stackmodel(x_train,y_train,x_test):
    # Partition the train set into 5 test sets
    x_train_duplicate = x_train.copy()
    k = x_train.shape[0]/5
    x_shape = [[], [], [], [], []]
    
    for i in range(4):
        sample = random.sample(x_train.index, k)
        x_shape[i] = x_train.ix[sample]
        x_train = x_train.drop(sample)
    x_shape[4] = x_train

    # Create train_meta and test_meta
    train_meta = pd.DataFrame()

    # For each fold in 1st, use other 5 folds as training set to predict the result for that fold.
    # Save them in train_meta
    for i in range(5):
        x_train = x_train_duplicate
        print ('\nStack {0}\n'.format(i))
        x_sub_test = x_shape[i]
        x_sub_train = x_train.drop(x_sub_test.index)
        y_sub_test = y_train[x_sub_test.index]
        y_sub_train = y_train[x_sub_train.index]
        gb_model = pd.DataFrame(gradient_boost(x_sub_train, y_sub_train, x_sub_test),
                        columns=['GB_low','GB_medium','GB_high'], index=x_sub_test.index)
        print ('Gradient Boosting is built.')
        lr_model = pd.DataFrame(logistic_regression(x_sub_train,y_sub_train,x_sub_test),
                        columns=['lr_low','lr_medium','lr_high'], index=x_sub_test.index)
        print ('Logistic is built.')
        knn_model = pd.DataFrame(k_nearest_neighbor(x_sub_train,y_sub_train,x_sub_test),
                        columns=['knn_low','knn_medium','knn_high'], index=x_sub_test.index)
        print ('KNN is built.')
        rf_model = pd.DataFrame(random_forest(x_sub_train,y_sub_train,x_sub_test),
                        columns=['rf_low','rf_medium','rf_high'], index=x_sub_test.index)
        print ('Random Forest is built.')
        ada_model = pd.DataFrame(ada_boost(x_sub_train,y_sub_train,x_sub_test),
                        columns=['ada_low','ada_medium','ada_high'], index=x_sub_test.index)
        print ('Adaboost is built.')
        train_meta = train_meta.append(pd.concat([gb_model, lr_model, knn_model, rf_model, ada_model], axis=1))
       
    # Fit each base model to the full training dataset 
    # Generate result for each classification model in a CSV format
    print ('Store each result in a CSV format')
    x_train = x_train_duplicate
    gb_model = pd.DataFrame(gradient_boost(x_train, y_train, x_test), 
                            columns=['high','medium','low'], index=x_test.index)
    print ('GB')
    gb_model.columns = ['high', 'medium', 'low']
    gb_model['listing_id'] = test.listing_id.values
    gb_model.to_csv('Gradient Boost-results.csv', index=False)
    
    lr_model = pd.DataFrame(logistic_regression(x_train, y_train, x_test), 
                            columns=['high','medium','low'], index=x_test.index)
    print ('LR')
    lr_model.columns = ['high', 'medium', 'low']
    lr_model['listing_id'] = test.listing_id.values
    lr_model.to_csv('Logistic Regression-results.csv', index=False)
    
    knn_model = pd.DataFrame(k_nearest_neighbor(x_train, y_train, x_test), 
                             columns=['high','medium','low'], index=x_test.index)
    print ('KNN')
    knn_model.columns = ['high', 'medium', 'low']
    knn_model['listing_id'] = test.listing_id.values
    knn_model.to_csv('K Neareast Neighbor-results.csv', index=False)
    
    rf_model = pd.DataFrame(random_forest(x_train, y_train, x_test), 
                            columns=['high','medium','low'], index=x_test.index)
    print ('RF')
    rf_model.columns = ['high', 'medium', 'low']
    rf_model['listing_id'] = test.listing_id.values
    rf_model.to_csv('Random Forest-results.csv', index=False)
    
    ada_model = pd.DataFrame(ada_boost(x_train, y_train, x_test), 
                             columns=['high','medium','low'], index=x_test.index)
    print ('ADA')
    ada_model.columns = ['high', 'medium', 'low']
    ada_model['listing_id'] = test.listing_id.values
    ada_model.to_csv('AdaBoost-results.csv', index=False)
    
def rent_interest_classifier():
    y_train = train['interest_level']
    x_train = numerical_features
    x_test = processed_test_data
    y_train_copy=y_train.copy()
    diction = {'high':1,'medium':2,'low':3}
    y_train1 = map(lambda x: diction[x], y_train)
    y_train = pd.Series(y_train1, index=y_train.index)
    stackmodel(x_train, y_train, x_test)

if __name__ == '__main__':
    train = pd.read_json("/Users/soyoungkim/Desktop/python_codes/two-sigma/data/train.json")
    test = pd.read_json("/Users/soyoungkim/Desktop/python_codes/two-sigma/data/test.json")

    start_time = time.time()
    train['interest'] = np.where(train['interest_level']=='high', 1,
                                np.where(train['interest_level']=='medium', 2, 3))

    global important_features
    important_features = ['bathrooms', 'bedrooms', 'price', 'price_room','latitude',
                          'longitude', 'nb_images','nb_features', 'sentiment',
                          'nb_description', 'description_len','b_counts', 'm_counts',
                          'b_count_log', 'm_count_log']
    numerical_features = preprocessing.pre_processing(train)
    processed_test_data = preprocessing.pre_processing(test)
    print ('A set of 15 derived features:{0}\n'.format(important_features))
    preprocessing.classification(numerical_features, processed_test_data, train['interest'])
    rent_interest_classifier()
    print ('--- %s seconds ---' % (time.time() - start_time))
