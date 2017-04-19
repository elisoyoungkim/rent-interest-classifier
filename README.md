# rent-interest-classifier
{
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.6.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0,
  "cells": [
    {
      "cell_type": "markdown",
      "source": "Rent Interest Classifier \n===\n---\n\n - This classification model predicts the degree of popularity for a rental listing judged by its profiles such as the number of rooms, location, price, etc.  \n - It predicts whether a given listing would receive \"low,\" \"medium,\" or\n   \"high\" interest with its corresponding probability to a particular listing.\n\n---\n**Multiclass Classifier with Probability Estimates**\n---\nThe problem of classification is considered as learning a model that maps instances to class labels. While useful for many purposes, there are numerous applications in which the estimation of the probabilities of the different classes is more desirable than just selecting one of them, in that probabilities are useful as a measure of the reliability of a classification.\n\n**Datasets**\n---\nNYC rent listing data from the rental website RentHop which is used to find the desired home.\nDatasets include \n\n 1. ***train*** and ***test*** databases, both provided in a JavaScript Object Notation format,\n 2. ***sample submission*** listing_id with interest level probabilities for each class i.e., high, medium, and low, \n 3. ***image sample*** of selective 100 listings, and\n 4. ***kagle-renthop*** zipfile that contains all listing images where the file size is 78.5GB. \n\nThe JSON dataset is a structured database that contains the listing information as the number of bathrooms and bedrooms, building_id, created, description, display_address, features, latitude, listing_id, longitude, manager_id, photos links, price, street_address,  and interest_level.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load in \n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the \"../input/\" directory.\n# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory\n\nfrom subprocess import check_output\nprint(check_output([\"ls\", \"../input\"]).decode(\"utf8\"))\n\n# Any results you write to the current directory are saved as output.\n\nimport matplotlib.pyplot as plt\n% matplotlib inline\nimport seaborn as sns\nsns.set(style=\"whitegrid\", color_codes=True)\nsns.set(font_scale=1)\n\nimport plotly.plotly as py\nimport plotly.graph_objs as go\nfrom plotly import tools\n\nfrom plotly.offline import download_plotlyjs, init_notebook_mode, iplot\ninit_notebook_mode(connected=True)\n\ntrain = pd.read_json(\"../input/train.json\")\ntest = pd.read_json(\"../input/test.json\")",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "print ('There are {0} rows and {1} attributes.'.format(train.shape[0], train.shape[1]))\nprint (len(train['listing_id'].unique()))\ntrain = train.set_index('listing_id')\ntrain.head()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "print ('There are {0} rows and {1} attributes.'.format(test.shape[0], test.shape[1]))\ntest.tail()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "**Pre-processing and feature extraction**\n---\n**Feature Selection in Python with Scikit-Learn**\n\nFeature selection is a process where you automatically select affective features in your data that contribute most to the prediction variable or target output. In order to maximize the performance of machine learning techniques,  important attributes are selected before creating a machine learning model using the scikit-learn library - feature importance ranking.\n\nGiven an importance score for each attribute where the larger score the more important the attribute. The scores show price, the number of features/photos/words, and date as the importance attributes.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "train.info()\ntrain.describe()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "----------\n**Interest Level Distribution**\n----------\nDistribution of interest level: \n - **Low (69.5%)**\n - Medium (22.8%)\n - Hight (7.8%)",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "plt.subplots(figsize=(10, 8))\nsizes = train['interest_level'].value_counts().values\npatches, texts, autotexts= plt.pie(sizes, labels=['Low', 'Medium', 'High'],\n                                  colors=['mediumaquamarine','lightcoral', 'steelblue'],\n                                  explode=[0.1, 0, 0], autopct=\"%1.1f%%\", \n                                  startangle=90)\n\ntexts[0].set_fontsize(13)\ntexts[1].set_fontsize(13)\ntexts[2].set_fontsize(13)\nplt.title('Interest level', fontsize=18)\nplt.show()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "----------\n**Feature Importance**\n----------\nEnsemble methods are a promising solution to highly imbalanced nonlinear classification tasks with mixed variable types and noisy patterns with high variance. Methods compute the relative importance of each attribute. \nThese importance values can be used to inform a feature selection process. This shows the construction of an Extra Trees ensemble of the dataset and the display of the relative feature importance.\n\nAs can be seen in the *train.info()* table, data types are mixed.\n\n 1. **Categorical**: description, display_address, features, manager_id, building_id, street_address\n 2. **Numeric**: bathrooms, bedrooms, latitude, longitude, price\n 3. Other: created, photos \n\nIn order to generate the feature importance matrix, non-numeric data types attributes can be good measures when converted to numerical values.\n\n - **description**: The more words and well-described listings might be spotted. \n - **features**: Some features are more preferred over others.\n - **photos**: The more images might get more views with having interest.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "from wordcloud import WordCloud,STOPWORDS\nfrom nltk.corpus import stopwords\nfrom textblob import TextBlob\n\ndef room_price(x, y):\n    if y == 0:\n        return 0\n    return x/y\n\ntrain['nb_images'] = train['photos'].apply(len)\ntrain['nb_features'] = train['features'].apply(len)\ntrain['nb_description'] = train['description'].apply(lambda x: len(x.split(' ')))\ntrain['description_len'] = train['description'].apply(len)\ntrain = train.join(\n                   train['description'].apply(\n                       lambda x: TextBlob(x).sentiment.polarity).rename('sentiment'))\n\ntrain['price_room'] = train.apply(lambda row: room_price(row['price'], \n                                                         row['bedrooms']), axis=1)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "----------\nAttribute: Building ID\n---",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "# Number of listings based on building ID\ntop_buildings = train['building_id'].value_counts().nlargest(10)\nprint (top_buildings)\nprint (len(train['building_id'].unique()))\n\ngrouped_building = train.groupby(\n                           ['building_id', 'interest_level']\n                          )['building_id'].count().unstack('interest_level').fillna(0)\n\ngrouped_building['sum'] = grouped_building.sum(axis=1)\nx = grouped_building[(grouped_building['sum'] > 50) & (grouped_building['high'] > 10)]\n\n# x = x[x.index != '0'] # Ignore N/A value\n\nfig = plt.figure(figsize=(10, 6))\n\nplt.title('Hight-interest buildings', fontsize=13)\nplt.xlabel('High interest level', fontsize=13)\nplt.ylabel('Building ID', fontsize=13)\nx['high'].plot.barh(color=\"palevioletred\");\n\nbuild_counts = pd.DataFrame(train.building_id.value_counts())\nbuild_counts['b_counts'] = build_counts['building_id']\nbuild_counts['building_id'] = build_counts.index\nbuild_counts['b_count_log'] = np.log2(build_counts['b_counts'])\ntrain = pd.merge(train, build_counts, on=\"building_id\")",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "----------\nAttribute: Manager ID\n---------",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "# Hight-interest managers\ntop_managers = train['manager_id'].value_counts().nlargest(10)\nprint (top_managers)\nprint (len(train['manager_id'].unique()))\n\ngrouped_manager = train.groupby(\n    ['manager_id', 'interest_level'])['manager_id'].count().unstack('interest_level').fillna(0)\n\ngrouped_manager['sum'] = grouped_manager.sum(axis=1)\nprint (grouped_manager.head())\n\nx = grouped_manager.loc[(grouped_manager['high'] > 20 ) & (grouped_manager['sum'] > 50)]\n\nplt.title('High-interest managers', fontsize=13)\nplt.xlabel('High interest level', fontsize=13)\nplt.ylabel('Manager ID', fontsize=13)\nx['high'].plot.barh(figsize=(10, 9), color=\"teal\");\n\nman_counts = pd.DataFrame(train.manager_id.value_counts())\nman_counts['m_counts'] = man_counts['manager_id']\nman_counts['manager_id'] = man_counts.index\nman_counts['m_count_log'] = np.log10(man_counts['m_counts'])\ntrain = pd.merge(train, man_counts, on=\"manager_id\")",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "----------\nFeature Importance Ranking\n---------",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "from sklearn.ensemble import ExtraTreesClassifier\nfrom xgboost import XGBClassifier\nfrom xgboost import plot_importance\nfrom matplotlib import pyplot\n\nnumerical_features = train[['bathrooms', 'bedrooms', 'price', 'price_room',\n                            'latitude','longitude', 'nb_images','nb_features', \n                            'nb_description', 'description_len','sentiment',\n                            'b_counts', 'm_counts',\n                            'b_count_log', 'm_count_log']]\n\n# Fit an Extra Trees model to the data\nmodel = ExtraTreesClassifier()\nmodel.fit(numerical_features, train['interest_level'])\n\n# Display the relative importance of each attribute\nprint (model.feature_importances_)\n\n# Plot feature importance\nplt.subplots(figsize=(12, 6))\nplt.title('Feature ranking', fontsize = 18)\nplt.ylabel('Importance degree', fontsize = 13)\n# plt.xlabel(\"Features\", fontsize = 14)\n\nfeature_names = numerical_features.columns\nplt.xticks(range(numerical_features.shape[1]), feature_names, fontsize = 8)\npyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)\npyplot.show()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "# Use feature importance for feature selection\nfrom numpy import sort\nfrom xgboost import XGBClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.feature_selection import SelectFromModel\n\n# Converting categorical values for Interest Level to numeric values\n# Low: 1, Medium: 2, High: 3\ntrain['interest'] = np.where(train['interest_level']=='low', 1,\n                             np.where(train['interest_level']=='medium', 2, 3))\n\nX = numerical_features\nY = train['interest']\n\n# Split data into train and test sets\nX_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, \n                                                    random_state=7)\n\n# Fit model on all training data\nmodel = XGBClassifier()\nmodel.fit(X_train, y_train)\n\n# Make predictions for test data and evaluate\ny_pred = model.predict(X_test)\npredictions = [round(value) for value in y_pred]\naccuracy = accuracy_score(y_test, predictions)\nprint(\"Accuracy: %.2f%%\" % (accuracy * 100.0))\n\n# Fit model using each importance as a threshold\nthresholds = sort(model.feature_importances_)\nfor thresh in thresholds:\n\t# Select features using threshold\n\tselection = SelectFromModel(model, threshold=thresh, prefit=True)\n\tselect_X_train = selection.transform(X_train)\n    \n\t# Train model\n\tselection_model = XGBClassifier()\n\tselection_model.fit(select_X_train, y_train)\n    \n\t# Evalation model\n\tselect_X_test = selection.transform(X_test)\n\ty_pred = selection_model.predict(select_X_test)\n\tpredictions = [round(value) for value in y_pred]\n\taccuracy = accuracy_score(y_test, predictions)\n\tprint (\"Thresh=%.3f, n=%d, Accuracy: %.2f%%\" % (thresh, select_X_train.shape[1], \n                                                    accuracy*100.0))",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "----------\nCorrelation Graph\n---------",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "f, ax = plt.subplots(figsize=(13, 13))\ncorr = numerical_features.corr()\nsns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), \n            cmap=sns.diverging_palette(220, 10, as_cmap=True),\n            square=True, ax=ax)",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "----------\nCorrelation Matrix\n---------",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)\n\ndef magnify():\n    return [dict(selector=\"th\",\n                 props=[(\"font-size\", \"10pt\")]),\n            dict(selector=\"td\",\n                 props=[('padding', \"0em 0em\")]),\n            dict(selector=\"th:hover\",\n                 props=[(\"font-size\", \"11pt\")]),\n            dict(selector=\"tr:hover td:hover\",\n                 props=[('max-width', '200px'),\n                        ('font-size', '11pt')])]\n\ncorr.style.background_gradient(cmap, axis=1)\\\n    .set_properties(**{'max-width': '80px', 'font-size': '8pt'})\\\n    .set_caption('Correlation Matrix')\\\n    .set_precision(2)\\\n    .set_table_styles(magnify())",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "numerical_features[['bathrooms', 'bedrooms', 'price', 'price_room',\n                    'latitude','longitude', 'nb_images','nb_features', \n                    'nb_description', 'description_len','sentiment',\n                    'b_counts', 'm_counts',\n                    'b_count_log', 'm_count_log']].hist(figsize=(12, 12))\nplt.show()",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "----------\n**Attribute:  Bathrooms, Bedrooms**\n----------",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "'''\nsubplot grid parameters encoded as a single integer.\nijk means i x j grid, k-th subplot\nsubplot(221) #top left\nsubplot(222) #top right\nsubplot(223) #bottom left\nsubplot(224) #bottom right \n'''\nfig = plt.figure(figsize=(12, 6))\n\n# Number of listings\nsns.countplot(train['bathrooms'], ax = plt.subplot(121));\nplt.xlabel('NB of bathrooms', fontsize=13);\nplt.ylabel('NB of listings', fontsize=13);\n\nsns.countplot(train['bedrooms'], ax = plt.subplot(122));\nplt.xlabel('NB of bedrooms', fontsize=13);\nplt.ylabel('NB of listings', fontsize=13);",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "# Number of rooms based on Interest level\ngrouped_bathroom = train.groupby(\n    ['bathrooms', 'interest_level'])['bathrooms'].count().unstack('interest_level').fillna(0)\ngrouped_bathroom[['low', 'medium', 'high']].plot.barh(stacked=True, figsize=(12, 4));\n\ngrouped_bedroom = train.groupby(\n    ['bedrooms', 'interest_level'])['bedrooms'].count().unstack('interest_level').fillna(0)\ngrouped_bedroom[['low', 'medium', 'high']].plot.barh(stacked=True, figsize=(12.25, 4));",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "----------\n**Attribute:  Geographical information - latitude, longitude**\n----------",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "'''\nseaborn.lmplot(x, y, data, hue=None, col=None, row=None, palette=None, \ncol_wrap=None, size=5, aspect=1, markers='o', sharex=True, sharey=True, \nhue_order=None, col_order=None, row_order=None, legend=True, legend_out=True, \nx_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, \nn_boot=1000, units=None, order=1, logistic=False, lowess=False, robust=False, \nlogx=False, x_partial=None, y_partial=None, truncate=False, x_jitter=None, \ny_jitter=None, scatter_kws=None, line_kws=None)\n'''\n\n# Rent interest based on geographical information\nsns.lmplot(x='longitude', y='latitude', fit_reg=False, hue='interest_level',\n           hue_order=['low', 'medium', 'high'], size=9, aspect=1, scatter_kws={'alpha':0.4,'s':30},\n           data=train[(train['longitude']>train['longitude'].quantile(0.1))\n                      &(train['longitude']<train['longitude'].quantile(0.9))\n                      &(train['latitude']>train['latitude'].quantile(0.1))                           \n                      &(train['latitude']<train['latitude'].quantile(0.9))]);\nplt.xlabel('Longitude', fontsize=13);\nplt.ylabel('Latitude', fontsize=13);",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "**Method**\n---\n**Building the Model**\n\nA decision tree is a tree structure, where the classification process starts from a root node and is split on every subsequent step based on the features and their values. The exact structure of a given decision tree is determined by a tree induction algorithm; there are a number of different induction algorithms which are based on different splitting criteria such as information gain. Ensemble learning method constructs a collection of individual classifiers that are diverse yet accurate. One of the most popular techniques for constructing ensembles is boostrap aggregation called ‘bagging’. In bagging, each training set is constructed by forming a bootstrap replicate of the original training set. So this bagging algorithm is promising ensemble learner that improves the results of any decision tree based learning algorithm.\n\nGradient boosting is also powerful techniques for building predictive models. While bagging considers candidate models equally, boosting techinique is based on whether a weak learner can be modified to become better. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. XGBoost stands for eXtreme Gradient Boosting.\n\nI generated a set of new features derived from the datasets as a preprocessing. A table with a new set of 15 features is generated in a CSV format instead of original mixed data types instances and it is mapped into inputs in XGBoost classification model. Other classification models - Support Vector Machine, Rnadom Forest, and Gradient Random Boosting were used to compare its performances.",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "from sklearn import svm\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom sklearn import model_selection\nfrom sklearn.model_selection import train_test_split\nfrom textblob import TextBlob\nfrom sklearn.metrics import accuracy_score\nimport time\n\ndef pre_processing(data):\n    \n    global important_features\n    important_features = ['bathrooms', 'bedrooms', 'price', 'price_room',\n                            'latitude','longitude', 'nb_images','nb_features', \n                            'nb_description', 'description_len','b_counts', 'm_counts',\n                            'b_count_log', 'm_count_log']\n    \n    data['nb_images'] = data['photos'].apply(len)\n    data['nb_features'] = data['features'].apply(len)\n    data['nb_description'] = data['description'].apply(lambda x: len(x.split(' ')))\n    data['description_len'] = data['description'].apply(len)\n    \n    def room_price(x, y):\n        if y == 0:\n            return 0\n        return x/y\n    \n    def sentiment_analysis(x):\n        if len(x) == 0:\n            return 0\n        return TextBlob(x[0]).sentiment.polarity\n    \n    data = data.join(data['description'].apply(\n                         lambda x: TextBlob(x).sentiment.polarity).rename('sentiment'))\n    data['price_room'] = data.apply(lambda row: \n                                    room_price(row['price'],row['bedrooms']), axis=1)\n    \n    build_counts = pd.DataFrame(data.building_id.value_counts())\n    build_counts['b_counts'] = build_counts['building_id']\n    build_counts['building_id'] = build_counts.index\n    build_counts['b_count_log'] = np.log2(build_counts['b_counts'])\n    data = pd.merge(data, build_counts, on='building_id')\n    \n    man_counts = pd.DataFrame(data.manager_id.value_counts())\n    man_counts['m_counts'] = man_counts['manager_id']\n    man_counts['manager_id'] = man_counts.index\n    man_counts['m_count_log'] = np.log10(man_counts['m_counts'])\n    data = pd.merge(data, man_counts, on='manager_id')\n    \n    return data[important_features]\n\ndef print_scores(test_name, train, test):\n    print ('{0} train score: {1}\\n{0} test score: {2}\\n'.format(test_name,\n                                                               train,\n                                                               test))\n\ndef classification(train_data, test_data, target, test_size=0.2, random_state=42):    \n    # Split data into X and y\n    X = numerical_features\n    Y = train['interest_level']\n\n    # Split data into train and test sets\n    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,\n                                                        random_state=random_state)\n    \n    # Support vector machine\n    svm_model = svm.SVC(decision_function_shape='ovo', tol=0.00000001)\n    svm_model = svm_model.fit(X_train, y_train)\n    print_scores(\"Support Vector Machine\",\n                 svm_model.score(X_train, y_train),\n                 accuracy_score(y_test, svm_model.predict(X_test)))\n\n    # Random Forest\n    random_forest = RandomForestClassifier(n_estimators=10)\n    random_forest = random_forest.fit(X_train, y_train)\n    print_scores(\"Random Forest\",\n                 random_forest.score(X_train, y_train),\n                 accuracy_score(y_test, random_forest.predict(X_test)))\n\n    # GradientBoostingClassifier\n    gradientB_model = GradientBoostingClassifier(n_estimators=20,\n                                      learning_rate=1.0,\n                                      max_depth=1,\n                                      random_state=0).fit(X_train, y_train)\n    gradientB_model = gradientB_model.fit(X_train, y_train)\n    print_scores(\"Gradient Boosting Classifier\",\n                 gradientB_model.score(X_train, y_train),\n                 accuracy_score(y_test, gradientB_model.predict(X_test)))\n\nprocessed_test_data = pre_processing(test)\nprint ('A set of 15 derived features:{0}'.format(important_features))\n'''\nstart_time = time.time()\nclassification(numerical_features, processed_test_data, train['interest_level'])\nprint ('--- %s seconds ---' % (time.time() - start_time))\n'''",
      "execution_count": null,
      "outputs": [],
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Reference\n----------\n\n - Classification models:\n\n1. https://blog.nycdatascience.com/student-works/renthop-kaggle-competition-team-null/\n2. http://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/\n\n - EDA:\n   https://www.kaggle.com/poonaml/two-sigma-connect-rental-listing-inquiries/two-sigma-renthop-eda",
      "metadata": {}
    }
  ]
}
