# coding: utf-8

# Rent Interest Classifier 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "/Users/soyoungkim/Desktop/python_codes/two-sigma"]).decode("utf8"))

import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

#from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
#init_notebook_mode(connected=True)

# Load datasets
train = pd.read_json("/Users/soyoungkim/Desktop/python_codes/two-sigma/data/train.json")
test = pd.read_json("/Users/soyoungkim/Desktop/python_codes/two-sigma/data/test.json")
global train, test

print ('There are {0} rows and {1} attributes.'.format(train.shape[0], train.shape[1]))
print (len(train['listing_id'].unique()))

# Set 'listing_id' as the primary key
train = train.set_index('listing_id')

print ('There are {0} rows and {1} attributes.'.format(test.shape[0], test.shape[1]))

# Get loaded dataset information: data types, data size, etc
train.info()
train.describe()
