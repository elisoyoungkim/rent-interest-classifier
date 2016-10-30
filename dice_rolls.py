'''
Created on Oct 30, 2016
@author: soyoungkim
'''
from __future__ import division
import operator as op

''' Find Standard Deviation Using Range Rule of Thumb
    Range is the simplest measure of variation, and it is the difference between high and low values. 
    Standard deviation is the standard measure of variation.'''

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numerator = reduce(op.mul, xrange(n, n-r, -1))
    denominator = reduce(op.mul, xrange(1, r+1))
    return numerator//denominator

def favorable_cases(summation, rolling_times, dice):
    k_max = (summation - rolling_times)//dice
    nb_favorable_cases = 0
    for k in range(0, k_max+1):
        val = (-1)**k * ncr(rolling_times, k) * ncr(summation - dice*k - 1, summation - dice*k - rolling_times)
        nb_favorable_cases += val
    return nb_favorable_cases

def possible_outcomes(space, dice):
    return pow(6, space)

def product_max(summation, rolling_times):
    return pow(summation//rolling_times, rolling_times) 

def digit_precision(value):
    roundup = 10 - len(str(value).split('.')[0])
    return round(value, roundup)

def get_probability(likeleness, space):
    return likeleness/space

def faces_list(summation, rolling_times, point):
    faces = []
    for i in range(1, rolling_times):
        if (rolling_times-i)*6 >= summation-i*point:
            faces.append(point)
        else:
            break
    return faces, summation-sum(faces)

def product_min(summation, rolling_times):
    faces_array = []
    for point in range(1,6):
        (new_faces, new_remaining) = faces_list(summation, rolling_times, point)
        rolling_times = rolling_times - sum(new_faces)
        summation = new_remaining
        if len(new_faces) is not 0: faces_array = faces_array + new_faces
    if new_remaining % 6 is 0: 
        for j in range(new_remaining//6): faces_array.append(6)
    print '\n\tSum: {0}, Rolling times: {1}'.format(sum(faces_array),len(faces_array))
    return reduce(op.mul, faces_array)

def std_rule_of_thumb(max_value, min_value, nb_samples):
    return (max_value-min_value)/nb_samples

if __name__ == '__main__':
    p_24_8 = get_probability(favorable_cases(24, 8, 6), possible_outcomes(8, 6))
    p_150_50 = get_probability(favorable_cases(150, 50, 6), possible_outcomes(50, 6))
    p_24_8_max = product_max(24,8)
    p_24_8_min = product_min(24,8)
    p_150_50_max = product_max(150,50)
    p_150_50_min = product_min(150,50)
    p_24_8_std = std_rule_of_thumb(p_24_8_max, p_24_8_min, favorable_cases(24, 8, 6))
    p_150_50_std = std_rule_of_thumb(p_150_50_max, p_150_50_min, favorable_cases(150, 50, 6))

    print '\n\tThe probabilities of P(24,8) and P(150,50) are {0} and {1}, repectively'.format(p_24_8, p_150_50)
    print '''\n\tThe maximum and minimum values of product P(24,8) are 
          \n\t\t\t\t\t {0} and {1}, repectively.'''.format(p_24_8_max , p_24_8_min)
    print '''\n\tThe maximum and minimum values of product P(150,50) are 
         \n\t\t\t\t\t {0} and {1}, repectively.'''.format(p_150_50_max, p_150_50_min)
    print '''\n\n\tFinal answers:
                   1. The expected values of product P(24,8) and P(150,50): {0}, {1}
                   2. The starndard deviation of poduct P(24,8) and P(150,50): {2}, {3}'''.format((p_24_8_max+p_24_8_min)/2, (p_150_50_max+p_150_50_min)/2, p_24_8_std, p_150_50_std)
