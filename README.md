## Data Incubator Challenge - 2016

One of the most common uses of big data is to predict ‘what users want’, which is well used by many companies such as Netflix and Amazon. Here I propose a prediction project that predicts movie ratings. The average user ratings for newly launched movies can be predicted from understanding the relationship between some attributes of a movie and its rating using data mining techniques. There are such as kernel regression, neural network, model trees, to name a few. The ability of handle large data sets is expected of every data scientist these days. Companies no longer prefer to work on samples, they are using full data. So we need to work on real and dirty datasets. The movie rating prediction problem could sound easy, but data management is a non-trivial challenge. Here I want to use few ensemble methods including random forest, bagging decision trees, or adaboost for regression techniques.

Data collection:
The Internet Movie Database (IMDb) is one of the biggest movie database on the web including rating by users. 
IMDB makes their raw data available (http://www.imdb.com/interfaces/).

Data pre-processing and feature selection:
Unfortunately, the data is divided into many text files and the format of each file differs slightly. The fact I had limited amount of working time, not to mention this is more 'proposal', I used cleaned dataset from here (https://blog.nycdatascience.com/student-works/machine-learning/movie-rating-prediction/). I used this datset not to use it for my project in Data Incubator Fellowship, but to show ensemble mothods are good methods in movie rating predicion problem. 
 - steps to follow to use IMDbPY API in SQL: http://imdbpy.sourceforge.net/docs/README.sqldb.txt

For my real project, to create one data file containing all the desired information, I will use a Python-based package called IMDbPY, which requires some development libraries installation including SQLObject. Through this pre-processing phase, IMDb plain text data files can be loaded into the MySQL database server as one table with a schema, in that I extract the relevant information and store in a database. Finally, this data is exported to csv to make it easier to import into data analysis packages. Some statistical approaches will be used to select the affecting features to prediction.

Data mining:
Experiments will be conducted using MySQL and Matlab for prediction model, Python will be used if necessary as this is my favorite language!! MySQL is used since Matlab has a limitation to loading a large dataset.

Correlation plot of IMDb movie dataset shows the correlations among a set of selective features using Matlab.
More anyalysis apporaches are needed. More specifically, based on real full IMDb datasets by scraping all relevant data into MySQL, a set of new number K predictive features are derived from existing features with understanding of rating tendencies. I start with simple statistical analysis based on fine-grained time series and delve into deeper analytical viewpoints to model prediction system. As for data mining techniques, mainly ensemble methods would be used, which is good for highly imbalanced nonlinear classification/regression tasks with mixed variable types and noisy patterns with high variance. 
