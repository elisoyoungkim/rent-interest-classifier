from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from sklearn.metrics import accuracy_score
import time
import EDA

def pre_processing(data):
    global important_features
    important_features = ['bathrooms', 'bedrooms', 'price', 'price_room',
                            'latitude','longitude', 'nb_images','nb_features', 
                            'nb_description', 'description_len','b_counts', 'm_counts',
                            'b_count_log', 'm_count_log']
    
    data['nb_images'] = data['photos'].apply(len)
    data['nb_features'] = data['features'].apply(len)
    data['nb_description'] = data['description'].apply(lambda x: len(x.split(' ')))
    data['description_len'] = data['description'].apply(len)
    
    def room_price(x, y):
        if y == 0:
            return 0
        return x/y
    
    def sentiment_analysis(x):
        if len(x) == 0:
            return 0
        return TextBlob(x[0]).sentiment.polarity
    
    data = data.join(data['description'].apply(
                         lambda x: TextBlob(x).sentiment.polarity).rename('sentiment'))
    data['price_room'] = data.apply(lambda row: 
                                    room_price(row['price'],row['bedrooms']), axis=1)
    
    build_counts = pd.DataFrame(data.building_id.value_counts())
    build_counts['b_counts'] = build_counts['building_id']
    build_counts['building_id'] = build_counts.index
    build_counts['b_count_log'] = np.log2(build_counts['b_counts'])
    data = pd.merge(data, build_counts, on='building_id')
    
    man_counts = pd.DataFrame(data.manager_id.value_counts())
    man_counts['m_counts'] = man_counts['manager_id']
    man_counts['manager_id'] = man_counts.index
    man_counts['m_count_log'] = np.log10(man_counts['m_counts'])
    data = pd.merge(data, man_counts, on='manager_id')
    
    return data[important_features]

def print_scores(test_name, train, test):
    print ('{0} train score: {1}\n{0} test score: {2}\n'.format(test_name,
                                                               train,
                                                               test))

def classification(train_data, test_data, target, test_size=0.2, random_state=42):    
    # Split data into X and y
    X = numerical_features
    Y = train['interest_level']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                        random_state=random_state)
    
    
    # XGBoost 
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    print_scores("XGBoost Classifier",
                 xgb_model.score(X_train, y_train),
                 accuracy_score(y_test, xgb_model.predict(X_test)))

    # Support vector machine
    svm_model = svm.SVC(decision_function_shape='ovo', tol=0.00000001)
    svm_model = svm_model.fit(X_train, y_train)
    print_scores("Support Vector Machine",
                 svm_model.score(X_train, y_train),
                 accuracy_score(y_test, svm_model.predict(X_test)))

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=10)
    random_forest = random_forest.fit(X_train, y_train)
    print_scores("Random Forest",
                 random_forest.score(X_train, y_train),
                 accuracy_score(y_test, random_forest.predict(X_test)))

    # GradientBoostingClassifier
    gradientB_model = GradientBoostingClassifier(n_estimators=20,
                                      learning_rate=1.0,
                                      max_depth=1,
                                      random_state=0).fit(X_train, y_train)
    gradientB_model = gradientB_model.fit(X_train, y_train)
    print_scores("Gradient Boosting Classifier",
                 gradientB_model.score(X_train, y_train),
                 accuracy_score(y_test, gradientB_model.predict(X_test)))

start_time = time.time()
processed_test_data = pre_processing(test)
print ('A set of 15 derived features:{0}\n'.format(important_features))
classification(numerical_features, processed_test_data, train['interest_level'])
print ('--- %s seconds ---' % (time.time() - start_time))