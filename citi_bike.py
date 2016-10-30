# -*- coding: utf-8 -*-
'''
Created on Oct 29, 2016
@author: soyoungkim
'''
from __future__ import division

import pandas as pd
import numpy as np
import time
from datetime import datetime as dt
from datetime import timedelta
from functools import partial
from math import radians, sin, cos, atan2, sqrt
from geopy.distance import great_circle
from dateutil import parser

''' datasets: citi bike riders information in 2015 (12 months), that is, 12 datasets.
    contains: (tripduration, starttime, stoptime, start station id, start station name
            start station latitude, start station longitude, end station id, end station name, 
            end station latitude, end station longitude, bikeid, usertype, birth year, gender)
            tripduration in secodns
            usertype: Subscirber, Customer
            gender: 1, 2

    1. median trip duration fraction in seconds
    2. rides being start and end staions equal
    3. stardard deviation of nb of stations visited by a bike (group by bike)
    4. average length in kilometers of a trip following Great Cicle Arcs
       (remove tuples that satisfy question 2)
    5. among average duration of trips for each month in 2015 (12 avg montly duration values)
       difference shortest and longest avg durations, in seconds
    6. hourly usage fracton = all rides starting at station that leave during a specfic hour
    7. customer: 30 mins ride (1800 seconds)
       subscriber: 45 mins ride (2700 secoonds)
       fraction of rides exceed time limit
       (group by usertype; count exceed rides)
    8. average number of times that bike is moved in 2015
       (graph traversal)
    '''  
'''
Question 1: 623.0
Question 2: 0.0223583913373
Question 3: 405880.880994
Question 4: 1.29517545468
Question 5: 430.57029597
Question 7: 0.0381067801681
Question 8: 320.790020054
--- 3733.87149382 seconds ---'''

def great_circle_distance(geo_cordinate):
  def mile(x): return 1.609 * x
  start = (geo_cordinate[0], geo_cordinate[1])
  end = (geo_cordinate[2], geo_cordinate[3])
  return mile(great_circle(start, end).miles)

def parse(x):
  date, time= x.split(' ')
  yyyy, mo, dd = date.split('-')
  hh, mm, ss = time.split(':')
  return parser.parse("%s %s %s %s:%s:%s" % (yyyy,mo,dd,hh,mm,ss))

def flatten(listinlist):
    out = []
    for item in listinlist:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out

if __name__ == '__main__':
    pass

start_time = time.time()

# load data (12 months; csv format), only duration, starttime, endtime, startstation info, endstation info, bikeid, usertype)
fields = ['tripduration', 'starttime', 'stoptime', 'start station id', 'start station longitude','start station latitude',
          'end station id', 'end station longitude','end station latitude', 'bikeid', 'usertype']

data = []
for i in range(12):
  if i < 9:
    file_name = '20150' + str(i+1) + '-citibike-tripdata.csv'
  else: 
    file_name = '2015' + str(i+1) + '-citibike-tripdata.csv'
  data.append(pd.read_csv(file_name, skipinitialspace=True, usecols=fields))

montly_duration_median_list = []
montly_duration_mean_list = []
distances = []
graph = {}

number_of_rides = 0
count_all = 0
exceed_count_all = 0
for month in data:
  print 'Number of tuples in file {0}: '.format(i+1), len(month)
  number_of_rides += len(month)
  montly_duration_median_list.append(month.tripduration.median())
  montly_duration_mean_list.append(month.tripduration.mean())
  count = 0
  exceed_count = 0
  for i in range(len(month)):
    if month['start station id'][i] == month['end station id'][i]:
      count += 1
    else:
      distances.append((month['start station longitude'][i], month['start station latitude'][i],
                       month['end station longitude'][i], month['end station latitude'][i]))
    # print month['usertype'][i], month['tripduration'][i]
    if month['usertype'][i] == 'Customer' and int(month['tripduration'][i]) > 1800: exceed_count += 1
    if month['usertype'][i] == 'Subscriber' and int(month['tripduration'][i]) > 2700: exceed_count += 1
    nodes = [month['start station id'][i], month['end station id'][i]]
    if month['bikeid'][i] in graph:
      graph[month['bikeid'][i]].append(nodes)
    else:
      graph.update({month['bikeid'][i]:[nodes]})

  count_all += count 
  exceed_count_all += exceed_count

nb_of_moves = []
[nb_of_moves.append(len(set(flatten(graph[key])))) for key in graph]

distance_list = []
for distance in distances:
  distance_list.append(great_circle_distance(distance))

# join 
citibike_data = pd.concat(data)
longest = max(montly_duration_mean_list)
shortest = min(montly_duration_mean_list)

# citibike_data['starttime'] = citibike_data['starttime'].apply(strptime_with_offset)
# citibike_data['starttime'] = citibike_data['starttime'].apply(parse)
# citibike_data['starttime']  = pd.to_datetime(citibike_data['starttime'])
# citibike_data['starttime'] = pd.to_datetime(citibike_data['starttime'], format='%m/%d/%Y %H:%M') 
# hourly = citibike_data.groupby(citibike_data.index.hour).sum() 
# print dfcsv.groupby(pd.TimeGrouper('60Min'))['tripduration'].sum()

print 'Question 1: ', np.median(montly_duration_median_list)
print 'Question 2: ', count_all/number_of_rides
print 'Question 3: ', citibike_data.groupby('bikeid')['tripduration'].sum().std()
print 'Question 4: ', np.mean(distance_list)
print 'Question 5: ', longest - shortest
# print 'Question 6: ', citibike_data.groupby([times.hour])['tripduration'].sum()/len(citibike_data)
print 'Question 7: ', exceed_count_all/len(citibike_data)
print 'Question 8: ', np.mean(nb_of_moves)
print '--- %s minutes ---' % (time.time() - start_time)/60


