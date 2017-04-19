# coding: utf-8

# Rent Interest Classifier 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "/Users/soyoungkim/Desktop/python_codes/two-sigma"]).decode("utf8"))

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode(connected=True)
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from textblob import TextBlob
import loading_data

# Distributions of interest level: Low (69.5%) Medium (22.8%) High (7.8%)
plt.subplots(figsize=(10, 8))
sizes = train['interest_level'].value_counts().values
patches, texts, autotexts= plt.pie(sizes, labels=['Low', 'Medium', 'High'],
                                  colors=['mediumaquamarine','lightcoral', 'steelblue'],
                                  explode=[0.1, 0, 0], autopct="%1.1f%%", 
                                  startangle=90)

texts[0].set_fontsize(13)
texts[1].set_fontsize(13)
texts[2].set_fontsize(13)
plt.title('Interest level', fontsize=18)
plt.show()

# Feature Importance**
train['nb_images'] = train['photos'].apply(len)
train['nb_features'] = train['features'].apply(len)
train['nb_description'] = train['description'].apply(lambda x: len(x.split(' ')))
train['description_len'] = train['description'].apply(len)

def room_price(x, y):
    if y == 0:
        return 0
    return x/y

train['nb_images'] = train['photos'].apply(len)
train['nb_features'] = train['features'].apply(len)
train['nb_description'] = train['description'].apply(lambda x: len(x.split(' ')))
train['description_len'] = train['description'].apply(len)
train = train.join(
                   train['description'].apply(
                       lambda x: TextBlob(x).sentiment.polarity).rename('sentiment'))
train['price_room'] = train.apply(lambda row: room_price(row['price'], 
                                                         row['bedrooms']), axis=1)

# Attribute: Building ID
# Number of listings based on building ID
top_buildings = train['building_id'].value_counts().nlargest(10)
print (top_buildings)
print (len(train['building_id'].unique()))

grouped_building = train.groupby(
    ['building_id', 'interest_level'])['building_id'].count().unstack('interest_level').fillna(0)

grouped_building['sum'] = grouped_building.sum(axis=1)
x = grouped_building[(grouped_building['sum'] > 50) & (grouped_building['high'] > 10)]

# x = x[x.index != '0'] # Ignore N/A value
fig = plt.figure(figsize=(10, 6))
plt.title('Hight-interest buildings', fontsize=13)
plt.xlabel('High interest level', fontsize=13)
plt.ylabel('Building ID', fontsize=13)
x['high'].plot.barh(color="palevioletred");

build_counts = pd.DataFrame(train.building_id.value_counts())
build_counts['b_counts'] = build_counts['building_id']
build_counts['building_id'] = build_counts.index
build_counts['b_count_log'] = np.log2(build_counts['b_counts'])
train = pd.merge(train, build_counts, on="building_id")

# Hight-interest managers
top_managers = train['manager_id'].value_counts().nlargest(10)
print (top_managers)
print (len(train['manager_id'].unique()))

grouped_manager = train.groupby(
    ['manager_id', 'interest_level'])['manager_id'].count().unstack('interest_level').fillna(0)

grouped_manager['sum'] = grouped_manager.sum(axis=1)
print (grouped_manager.head())

x = grouped_manager.loc[(grouped_manager['high'] > 20 ) & (grouped_manager['sum'] > 50)]

plt.title('High-interest managers', fontsize=13)
plt.xlabel('High interest level', fontsize=13)
plt.ylabel('Manager ID', fontsize=13)
x['high'].plot.barh(figsize=(10, 9), color="teal");

man_counts = pd.DataFrame(train.manager_id.value_counts())
man_counts['m_counts'] = man_counts['manager_id']
man_counts['manager_id'] = man_counts.index
man_counts['m_count_log'] = np.log10(man_counts['m_counts'])
train = pd.merge(train, man_counts, on="manager_id")

# Feature Importance Ranking
numerical_features = train[['bathrooms', 'bedrooms', 'price', 'price_room',
                            'latitude','longitude', 'nb_images','nb_features', 
                            'nb_description', 'description_len',
                            'b_count_log', 'm_count_log']]

# Fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(numerical_features, train['interest_level'])

# Display the relative importance of each attribute
print (model.feature_importances_)

# Plot feature importance
plt.subplots(figsize=(17, 6))
plt.title('Feature ranking', fontsize = 18)
plt.ylabel('Importance degree', fontsize = 13)

feature_names = numerical_features.columns
plt.xticks(range(numerical_features.shape[1]), feature_names, fontsize = 10)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

# Use feature importance for feature selection
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# Converting categorical values for Interest Level to numeric values
# Low: 1, Medium: 2, High: 3
train['interest'] = np.where(train['interest_level']=='low', 1,
                             np.where(train['interest_level']=='medium', 2, 3))

X = numerical_features
Y = train['interest']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, 
                                                    random_state=7)

# Fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
  # Select features using threshold
  selection = SelectFromModel(model, threshold=thresh, prefit=True)
  select_X_train = selection.transform(X_train)
    
  # Train model
  selection_model = XGBClassifier()
  selection_model.fit(select_X_train, y_train)
    
  # Evalation model
  select_X_test = selection.transform(X_test)
  y_pred = selection_model.predict(select_X_test)
  predictions = [round(value) for value in y_pred]
  accuracy = accuracy_score(y_test, predictions)
  print ("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], 
                                                    accuracy*100.0))

# Correlation Graph
f, ax = plt.subplots(figsize=(13, 13))
corr = numerical_features.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# Correlation Matrix
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "10pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "11pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '11pt')])]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '8pt'})\
    .set_caption('Correlation Matrix')\
    .set_precision(2)\
    .set_table_styles(magnify())

# Histogram
numerical_features[['bathrooms', 'bedrooms', 'price', 'price_room',
                    'latitude','longitude', 'nb_images','nb_features', 
                    'nb_description', 'description_len','sentiment',
                    'b_counts', 'm_counts',
                    'b_count_log', 'm_count_log']].hist(figsize=(12, 12))
plt.show()

# **Attribute:  Bathrooms, Bedrooms**
fig = plt.figure(figsize=(11, 6))

# Number of listings
sns.countplot(train['bathrooms'], ax = plt.subplot(121));
plt.xlabel('NB of bathrooms', fontsize=13);
plt.ylabel('NB of listings', fontsize=13);

sns.countplot(train['bedrooms'], ax = plt.subplot(122));
plt.xlabel('NB of bedrooms', fontsize=13);
plt.ylabel('NB of listings', fontsize=13);

# Number of rooms based on Interest level
grouped_bathroom = train.groupby(
    ['bathrooms', 'interest_level'])['bathrooms'].count().unstack('interest_level').fillna(0)
grouped_bathroom[['low', 'medium', 'high']].plot.barh(stacked=True, figsize=(12, 4));

grouped_bedroom = train.groupby(
    ['bedrooms', 'interest_level'])['bedrooms'].count().unstack('interest_level').fillna(0)
grouped_bedroom[['low', 'medium', 'high']].plot.barh(stacked=True, figsize=(12.25, 4));

# Rent interest graph of New-York
sns.lmplot(x='longitude', y='latitude', fit_reg=False, hue='interest_level',
           hue_order=['low', 'medium', 'high'], size=10, scatter_kws={'alpha':0.4,'s':30},
           data=train[(train['longitude']>train['longitude'].quantile(0.1))
                      &(train['longitude']<train['longitude'].quantile(0.9))
                      &(train['latitude']>train['latitude'].quantile(0.1))                           
                      &(train['latitude']<train['latitude'].quantile(0.9))]);
plt.xlabel('Longitude');
plt.ylabel('Latitude');