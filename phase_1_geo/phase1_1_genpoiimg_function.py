import sys
from math import cos, sin, acos
import numpy as np
import matplotlib.pyplot as plt
import random as r
import math as m


'''
    Argument Setting and Excecuting Hint
'''
def arg_setting (argv) :
    if len(argv) != 5 :
        print ("\nArgument # error\n\n\tUsage  :  python  poi_array_test.py  [target POI]  [POI Map]  [Side Length]  [Border patition #]\n")
        sys.exit()
    return argv[1], readFile(argv[2]), float(argv[3]), int(argv[4])


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
    Read ',' split data (mainly checkin & poi data)

    Input Data format :
        (lati),(longi),(venue type),(unique user),(total checkins),(store name)
'''
def readFile (fileName) :
    data = []
    with open (fileName, 'r') as f:
        for line in f:
            data.append([row for row in line.strip().split(',')])
    f.close()
    return data


'''
    find ... LAT [_minus, _plus] :
        input target latitude & longitude, find setting distance's latitude

    find ... LON [_minus, _plus] :
        input target latitude & longitude, find setting distance's longitude

    dis : distance (km)
'''
def findLAT_minus (lat, lon, dis) :
    lati = lat

    while abs(distance(lat, lon, lati, lon) - dis) / dis > .005 :
        #print (lati, '\t', distance(lat,lon,lati,lon), '\t', ((distance(lat,lon,lati,lon)-dis)/dis))
        lati -= .00001

    return float("{0:.7f}".format(lati))


def findLAT_plus (lat, lon, dis) :
    lati = lat

    while abs(distance(lat, lon, lati, lon) - dis) / dis > .005 :
        #print (lati, '\t', distance(lat,lon,lati,lon), '\t', ((distance(lat,lon,lati,lon)-dis)/dis))
        lati += .00001

    return float("{0:.7f}".format(lati))


def findLON_minus (lat, lon, dis) :
    longi = lon

    while abs(distance(lat, lon, lat, longi) - dis) / dis > .005 :
        #print (longi, '\t', distance(lat,lon,lat,longi), '\t', ((distance(lat,lon,lat,longi)-dis)/dis))
        longi -= .00001

    return float("{0:.7f}".format(longi))


def findLON_plus (lat, lon, dis) :
    longi = lon

    while abs(distance(lat, lon, lat, longi) - dis) / dis > .005 :
        #print (longi, '\t', distance(lat,lon,lat,longi), '\t', ((distance(lat,lon,lat,longi)-dis)/dis))
        longi += .00001

    return float("{0:.7f}".format(longi))


'''
    input target geo-point, calculate poi-map in square range
        inter_num when in binary cut mode means binary cut times
'''
def find_poi_array_interval (tar_lati, tar_longi, radius, inter_num) :
    lat_down = findLAT_minus (tar_lati, tar_longi, radius)
    lat_up = findLAT_plus (tar_lati, tar_longi, radius)
    lon_left = findLON_minus (tar_lati, tar_longi, radius)
    lon_right = findLON_plus (tar_lati, tar_longi, radius)

    # Spilt grid from latitude and longitude range
    lat_range = abs(lat_up - lat_down)
    lon_range = abs(lon_right - lon_left)

    # every interval point record in np.array
    '''
    # if array range=500m, each interval about 45m (split in 11 parts)
    #lat_interval = np.linspace (lat_down, lat_up, 11)
    lat_interval = np.linspace (lat_down, lat_up, inter_num)
    #lon_interval = np.linspace (lon_right, lon_left, 11)
    lon_interval = np.linspace (lon_right, lon_left, inter_num)
    '''

    # Using binary cut mainly (quad-tree style)
    lat_interval = np.linspace (lat_down, lat_up, 2**inter_num+1)
    lon_interval = np.linspace (lon_left, lon_right, 2**inter_num+1)


    #print ("Latitude Range  :  %f" % lat_range)
    #print ("Longitude Range  :  %f" % lon_range)

    return lat_interval, lon_interval


'''
    accumulate checkins surround with assigned position

    Argu : assigned position (latitude, longitude)
    Retrurn : 2D np.array with accumulate checkins
'''
def accu_checkin (lat_inte, lon_inte, inter_degree, poi_map, skip_lat, skip_lon) :
    inter_num = 2**inter_degree

    # Initialize poi array
    accu_arr = np.zeros ([inter_num, inter_num], dtype=np.float32)

    '''
        x axis => longitude
        y axis => latitude
    '''
    for cnt in range(len(poi_map)) :

        # skip center poi (self)
        if float(poi_map[cnt][0]) == float(skip_lat) and float(poi_map[cnt][1]) == float(skip_lon) :
            continue

        row = -1
        col = -1
        lat = float (poi_map[cnt][0])
        lon = float (poi_map[cnt][1])
        if lat >= lat_inte[0] :
            for cntIn in range(len(lat_inte)) :
                if lat < lat_inte[cntIn] :
                    col = cntIn-1
                    break

        if lon >= lon_inte[0] :
            for cntIn in range(len(lon_inte)) :
                if lon < lon_inte[cntIn] :
                    row = cntIn-1
                    break

        if row != -1 and col != -1 :
            accu_arr[row, col] += int(poi_map[cnt][4])


    return accu_arr


'''
    Input :
        [comp_type] : (list) venue type of competitiveness to the target
                      e.g. "Coffee Shop" to the "Starbucks"
'''
def accu_competitive (lat_inte, lon_inte, inter_degree, poi_map, comp_type, skip_lat, skip_lon) :
    inter_num = 2**inter_degree

    # Initialize poi array
    accu_arr = np.zeros ([inter_num, inter_num], dtype=np.float32)

    '''
        x axis => longitude
        y axis => latitude
    '''

    for cnt in range(len(poi_map)) :

        #print (skip_lat, '\t', skip_lon)
        #print (poi_map[cnt][0])

        # skip center poi (self)
        if float(poi_map[cnt][0]) == float(skip_lat) and float(poi_map[cnt][1]) == float(skip_lon) :
            continue

        row = -1
        col = -1
        lat = float (poi_map[cnt][0])
        lon = float (poi_map[cnt][1])
        if lat >= lat_inte[0] :
            for cntIn in range(len(lat_inte)) :
                if lat < lat_inte[cntIn] :
                    col = cntIn-1
                    break

        if lon >= lon_inte[0] :
            for cntIn in range(len(lon_inte)) :
                if lon < lon_inte[cntIn] :
                    row = cntIn-1
                    break

        if row != -1 and col != -1 :
            if poi_map[cnt][2] in comp_type :
                accu_arr [row, col] += int(poi_map[cnt][4])

    return accu_arr


'''
    Get specific type of poi positions
'''
def get_spec (poi_map, target) :    
    spec = []

    for row in poi_map :
        if row[5] == target :
            spec.append (row)

    return spec


'''
    Write POI image to file

    [row 0] : total venue num
    [row 1] : poi img side partition num
    [row 2] : '\n'
    [row 3] -----> poi img info

    [poi img info] :
        label (e.g. total counts)
        center lati, longi
        [
            poi map content
        ]
'''
def writePOIIMG (poiIMG, fileName, part) :

    # file status
    fw = open (fileName, 'w')
    fw.write (str(len(poiIMG)))
    fw.write ('\n')
    fw.write (str(part))
    fw.write ('\n\n')

    for cnt in range(len(poiIMG)) :
        # label and (lati, longi)
        fw.write (str(poiIMG[cnt][0]))
        fw.write ('\n')
        fw.write (str(poiIMG[cnt][2]))
        fw.write ('\t')
        fw.write (str(poiIMG[cnt][3]))
        fw.write ('\n')

        # poi img
        for row in poiIMG[cnt][1] :
            for ele in row :
                fw.write (str(ele))
                fw.write ('\t')
            fw.write ('\n')

    fw.close()

