import sys
from math import cos, sin, acos
import numpy as np
import time as t
import random as r
import math as m

from scipy import stats


'''
    Argument read & check
'''
def arg_setting (argv) :
    if len(argv) != 3 :
        print ("\nArgument # error\n\n\tUsage  :  python  poi_array_test.py  [target POI]  [POI Map]\n")
        sys.exit()

    return argv[1], readFile(argv[2])


'''
    Calculate two geo-points' distance
'''
pi = 3.14159265358979323846

def deg2rad (deg) :
    return deg * pi / 180

def rad2deg (rad) :
    return rad * 180 / pi

def distance (lat1, lon1, lat2, lon2) :
    theta = lon1 - lon2

    dist = sin (deg2rad(lat1)) * sin(deg2rad(lat2)) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * cos(deg2rad(theta))

    if dist > 1.0 :
        dist = 1.0
    elif dist < -1.0 :
        dist = -1.0

    dist = acos(dist)
    dist = rad2deg(dist)
    dist = dist * 60 * 1.1515

    # transform dist from mile to km
    dist = dist*1.609344
    return dist


'''
    Read '\t' split data (mainly checkin & poi data)
'''
def readFile (fileName) :
    data = []
    with open (fileName, 'r') as f:
        for line in f:
            data.append([row for row in line.strip().split('\t')])
    f.close()
    return data


'''
    Get specific type of poi positions
'''
def get_spec (poi_map, target) :
    spec = []

    for row in poi_map :
        if row[3] == target :
            spec.append (row)
    return spec


'''
    Return NDCG 10
'''
def NDCG (test, ground, ndcg_num) :
    ndcg = .0
    ideal_ndcg = .0
    score = {}
    for cnt in range(len(ground)) :
        score[ground[cnt]] = len(ground) - cnt
    for cnt in range(ndcg_num) :
        ideal_ndcg += (len(ground)-cnt)/m.log2(cnt+2)
        ndcg += score[test[cnt]] / m.log2(cnt+2)
    return ndcg/ideal_ndcg


'''
    Accumulate target surrounding check-ins
'''
def lr_x_checkins (lati, longi, radius, poimap) :
    count = 0
    for cnt in range(len(poimap)) :
        if distance(lati, longi, float(poimap[cnt][1]), float(poimap[cnt][2])) < radius :
            count += int(poimap[cnt][4])
    return count

'''
    Accumulate target surrounding pois
'''

def lr_x_pois (lati, longi, radius, poimap) :
    count = 0
    for cnt in range(len(poimap)) :
        if distance(lati, longi, float(poimap[cnt][1]), float(poimap[cnt][2])) < radius :
            count += 1
    return count



if __name__ == "__main__" :

    #ts = t.time()
    tarPOI, poiMAP = arg_setting (sys.argv)
    specPOI = get_spec (poiMAP, tarPOI)
    #print (tarPOI)


    # Do only once, calculate accumulated feature in assigned radius
    '''
    lr_x_ = []
    cnt = 1
    for row in specPOI :
        print ("%d / %d"%(cnt, len(specPOI)), end='\r')
        lr_x_.append(lr_x_checkins (float(row[1]), float(row[2]), .25, poiMAP))
        cnt += 1
    print ("%d / %d" % (cnt-1, len(specPOI)))

    fileName = "LR_checkins_radius250m_" + tarPOI
    fw = open (fileName, 'w')
    for cnt in range (len(specPOI)) :
        for cntIn in range(len(specPOI[cnt])) :
            fw.write (specPOI[cnt][cntIn])
            fw.write ('\t')
        fw.write (str(lr_x_[cnt]))
        fw.write ('\n')
    fw.close()
    sys.exit()
    '''
    # Use read file since second times

    lrFileName = "LR_checkins_radius250m_" + tarPOI
    lr_read = readFile (lrFileName)
    #print (len(lr_read))



    lr_data = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for cnt in range(len(lr_read)) :
        if int(lr_read[cnt][4]) > 1 :
            lr_data.append ([int(lr_read[cnt][5]), int(lr_read[cnt][4])])
    #print (len(lr_data))

    r.shuffle (lr_data)

    test_num = m.floor(len(lr_data) / 10)
    for cnt in range(len(lr_data)) :
        if cnt < test_num :
            x_test.append (lr_data[cnt][0])
            y_test.append (lr_data[cnt][1])
        else :
            x_train.append (lr_data[cnt][0])
            y_train.append (lr_data[cnt][1])
    x_test = np.array (x_test)
    x_train = np.array (x_train)
    y_test = np.array (y_test)
    y_train = np.array (y_train)


    # Train
    slope, intercept, r_value, p_value, std_err = stats.linregress (x_train, y_train)
    #print ("Loss  :  ", std_err)


    # Test
    y_pred = []
    for cnt in range(len(x_test)) :
        y_pred.append (slope*x_test[cnt] + intercept)

    '''
    print ("Test number  :  ", len(y_test))
    print ("Ground Truth  :\n", y_test)
    print ("Test Result  :\n", y_pred)
    '''

    # Get ordered position
    yp_sort_pos = sorted (range(len(y_pred)), key=lambda k : y_pred[k])
    yp_sort_pos.reverse()
    yg_sort_pos = sorted (range(len(y_test)), key=lambda k : y_test[k])
    yg_sort_pos.reverse()

    ndcg5 = NDCG (yp_sort_pos, yg_sort_pos, 5)
    ndcg10 = NDCG (yp_sort_pos, yg_sort_pos, 10)
    '''
    print ("NDCG5  :  {}".format(ndcg5))
    print ("NDCG10  :  {}".format(ndcg10))
    '''
    #te = t.time()
    #print ("Processing time  :  {}".format(te-ts))


    # Write result to file
    fileName = "LR_checkins_results_radius250m_" + tarPOI
    fw = open (fileName, 'a+')
    fw.write (str(ndcg5))
    fw.write ('\t')
    fw.write (str(ndcg10))
    fw.write ('\n')
    fw.close()

