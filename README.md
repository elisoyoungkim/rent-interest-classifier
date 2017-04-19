Rent Interest Classifier 
===
---

 - This classification model predicts the degree of popularity for a rental listing judged by its profiles such as the number of rooms, location, price, etc.  
 - It predicts whether a given listing would receive "low," "medium," or
   "high" interest with its corresponding probability to a particular listing.

---
**Multiclass Classifier with Probability Estimates**
---
The problem of classification is considered as learning a model that maps instances to class labels. While useful for many purposes, there are numerous applications in which the estimation of the probabilities of the different classes is more desirable than just selecting one of them, in that probabilities are useful as a measure of the reliability of a classification.

**Datasets**
---
NYC rent listing data from the rental website RentHop which is used to find the desired home.
Datasets include 

 1. ***train*** and ***test*** databases, both provided in a JavaScript Object Notation format,
 2. ***sample submission*** listing_id with interest level probabilities for each class i.e., high, medium, and low, 
 3. ***image sample*** of selective 100 listings, and
 4. ***kaggle-renthop*** zipfile that contains all listing images where the file size is 78.5GB. 

The JSON dataset is a structured database that contains the listing information as the number of bathrooms and bedrooms, building_id, created, description, display_address, features, latitude, listing_id, longitude, manager_id, photos links, price, street_address,  and interest_level.

**Pre-processing and feature extraction**
---
**Feature Selection in Python with Scikit-Learn**

Feature selection is a process where you automatically select affective features in your data that contribute most to the prediction variable or target output. In order to maximize the performance of machine learning techniques,  important attributes are selected before creating a machine learning model using the Scikit-learn library having the feature_importances_ member variable of the trained model. 

Given an importance score for each attribute where the larger score the more important the attribute. The scores show price, the number of features/photos/words, and date as the importance attributes.

----------
**Interest Level Distribution**
----------
Distribution of interest level: 
 - **Low (69.5%)**
 - Medium (22.8%)
 - High (7.8%)

![alt tag](https://cloud.githubusercontent.com/assets/22326212/25195695/6c24c80a-250c-11e7-88cd-13559a06e505.png)

 ----------
**Feature Importance**
----------
Ensemble methods are a promising solution to highly imbalanced nonlinear classification tasks with mixed variable types and noisy patterns with high variance. Methods compute the relative importance of each attribute. These importance values can be used to inform a feature selection process. This shows the construction of an Extra Trees ensemble of the dataset and the display of the relative feature importance.

In this datasets, data types are mixed.

 1. **Categorical**: description, display_address, features, manager_id, building_id, street_address
 2. **Numeric**: bathrooms, bedrooms, latitude, longitude, price
 3. Other: created, photos 

In order to generate the feature importance matrix, non-numeric data types attributes should be converted to numerical values. Following assumptions are considered.

 - **description**: The more words and well-described listings might be spotted. 
 - **features**: Some features are more preferred over others.
 - **photos**: The more images might get more views with having interest.

![alt tag](https://cloud.githubusercontent.com/assets/22326212/25195708/7792a400-250c-11e7-91be-087185f442d6.png)


----------
**Correlation Graph**
----------
![alt tag](https://cloud.githubusercontent.com/assets/22326212/25195723/8238d0b4-250c-11e7-80ff-329af136213f.png)

----------
**Correlation Graph**
----------
![alt tag](https://cloud.githubusercontent.com/assets/22326212/25195736/8a397002-250c-11e7-9cb1-a05a2a857572.png)

----------
**Geographical Graph**
----------
![alt tag](https://cloud.githubusercontent.com/assets/22326212/25195751/92e869d8-250c-11e7-9ecc-bc157bed2ab4.png)

 **Methods**
---
**Building the classification Model**

Two main techniques are considered to build the classification model: Decision Tree and Ensemble Method. Let us start with the definitions. 

 - A decision tree is a tree structure, where the classification process starts from a root node and is split on every subsequent step based on the features and their values. The exact structure of a given decision tree is determined by a tree induction algorithm; there are a number of different induction algorithms which are based on different splitting criteria such as information gain.
 - Ensemble learning method constructs a collection of individual classifiers that are diverse yet accurate. 
    1. Bagging
   - One of the most popular techniques for constructing ensembles is boostrap aggregation called
   ‘bagging’. In bagging, each training set is constructed by forming a bootstrap replicate of the original training set. So this bagging algorithm is promising ensemble learner that improves the results of any decision tree based learning algorithm.
    2. Boosting
   - Gradient boosting is also powerful techniques for building predictive models. While bagging considers candidate models equally, boosting technique is based on whether a weak learner can be modified to become better. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. XGBoost stands for eXtreme Gradient Boosting.

I generated a set of new features derived from the datasets as a preprocessing. A table with a new set of 15 features is generated in a CSV format instead of original mixed data types instances and it is mapped into inputs in XGBoost classification model. Other classification models - Support Vector Machine, Rnadom Forest, and Gradient Random Boosting were used to compare its performances.
![alt tag](https://cloud.githubusercontent.com/assets/22326212/25195768/a1eed5b6-250c-11e7-82fa-d407b0f6f146.png)

Conclusion
---

Instead of running a time consuming cross validation, you can pull a fold out from the full training set and use it as your subtest in XGBoost. By adding it to the watchlist, you can assess the accuracy of your model on a smaller test set while training it.  Of course, since this doesn’t compare all folds against each other like actual cross validation, it doesn’t account for variance between folds and is prone to some degree of error. Often, the test score could be lowered to 0.55 in script, but our submitted model received a Kaggle score closer to 0.6.

Our final model, then, was an XGBoost that included a) basic features like price, b) mutated features, like price-per-room and number of photos, and c) engineered features, including neighborhood designations and principal-component stand-ins for various features.  This netted us a final score of 0.5625.  It's a solid figure, though one much higher than the contest leaders.  We came up with some good ideas, then, but we still have a ways to go before we're machine-learning gurus.

The conclusions we can draw from this project are many. It is extremely important to fully understand and define the value that is being modeled, as well as keep an objective view in regards to the analysis. It was quite interesting and humbling to find so many seemingly contrarian indicators throughout this analysis - such as the surprisingly unimportant Latitude/Longitude features.
