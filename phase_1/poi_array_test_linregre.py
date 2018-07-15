import sys
from math import cos, sin, acos
import numpy as np
import matplotlib.pyplot as plt
import time as t
import random as r
import math as m

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, MaxPooling2D, GlobalMaxPool2D, GlobalMaxPool1D
from keras.optimizers import Adam
from keras import backend as kb

from scipy import stats

'''
    Argument read & check
'''
def arg_setting (argv) :

    '''
    if len(argv) != 5 :
        print ("\nArgument # error\n\nUsage  :  python  poi_array_test.py  [target lati]  [target longi]  [distance range]  [poi data]\n")
        sys.exit()

    return argv[1], argv[2], argv[3], readFile(argv[4])
    '''

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
def accu_checkin (lat_inte, lon_inte, inter_degree, poi_map) :
    
    inter_num = 2**inter_degree

    # Initialize poi array
    accu_arr = np.zeros ([inter_num, inter_num], dtype=np.float32)

    '''
        x axis => longitude
        y axis => latitude
    '''
    for cnt in range(len(poi_map)) :
        row = -1
        col = -1
        lat = float (poi_map[cnt][1])
        lon = float (poi_map[cnt][2])
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
            accu_arr[row, col] += 1


    return accu_arr


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


def lr_x (lati, longi, radius, poimap) :

    count = 0

    for cnt in range(len(poimap)) :
        if distance(lati, longi, float(poimap[cnt][1]), float(poimap[cnt][2])) < radius :
            count += int(poimap[cnt][4])

    return count



if __name__ == "__main__" :

    #lat, lon, dis, poi = arg_setting (sys.argv) 
    tarPOI, poiMAP = arg_setting (sys.argv)

    # get all target POI data row
    specPOI = get_spec (poiMAP, tarPOI)


    print (tarPOI)


    partDEG = 4
    poiIMG = []
    cnt = 1
    ts = t.time()
    lr_x_ = []
    for row in specPOI :
        print ("%d / %d"%(cnt, len(specPOI)), end='\r')



        ### Get linear regression x ###
        lr_x_.append(lr_x (float(row[1]), float(row[2]), .5, poiMAP))
        cnt += 1
        continue
        ###############################


        # return target poi map range
        lat_interval, lon_interval = find_poi_array_interval (float(row[1]), float(row[2]), .38, partDEG)

        # return target poi img, simply accumulate checkins
        # [0] : # of checkins
        # [1] : accu poi MAP
        # [2][3] : center latitude & longitude
        if int(row[4]) > 1 :
            poiIMG.append ([int(row[4]), accu_checkin (lat_interval, lon_interval, partDEG, poiMAP), row[1], row[2]])

        cnt += 1
    te = t.time()

    print ("%d / %d" % (cnt-1, len(specPOI)))
    print ("Processing Time  :  %f  (secs)" % (te-ts))


    ############################
    ### Do linear regression ###
    lr_data = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for cnt in range(len(specPOI)) :
        if int(specPOI[cnt][4]) > 1 :
            lr_data.append ([int(lr_x_[cnt]), int(specPOI[cnt][4])])
    print (len(lr_data))

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
    '''
    model = Sequential()

    model.add (Dense(output_dim=1, input_dim=1))
    model.compile (loss='mse', optimizer='sgd')

    print ("LR Training ...")
    train_times = 301
    for step in range(train_times) :
        cost = model.train_on_batch(x_train, y_train)
        if step % 100 == 0:
            print ("train cost  :  {}".format(cost))
    '''
    slope, intercept, r_value, p_value, std_err = stats.linregress (x_train, y_train)
    print ("Loss  :  ", std_err)

    # Test
    '''
    y_pred = model.predict(x_test)
    '''
    y_pred = []
    for cnt in range(len(x_test)) :
        y_pred.append (slope*x_test[cnt] + intercept)


    print ("Test number  :  ", len(y_test))
    print ("Ground Truth  :\n", y_test)
    print ("Test Result  :\n", y_pred)


    yp_sort_pos = sorted (range(len(y_pred)), key=lambda k : y_pred[k])
    yp_sort_pos.reverse()
    yg_sort_pos = sorted (range(len(y_test)), key=lambda k : y_test[k])
    yg_sort_pos.reverse()

    ndcg5 = NDCG (yp_sort_pos, yg_sort_pos, 5)
    ndcg10 = NDCG (yp_sort_pos, yg_sort_pos, 10)

    print ("NDCG5  :  {}".format(ndcg5))
    print ("NDCG10  :  {}".format(ndcg10))

    '''
    x = np.linspace (0, 30000, 2)
    y = slope*x + intercept
    plt.scatter (x_train, y_train)
    plt.plot (x, y, color='red')
    plt.show()
    '''
    fileName = "LinearRegression_test_" + sys.argv[1]
    fw = open (fileName, 'a+')
    fw.write (str(ndcg5))
    fw.write ('\t')
    fw.write (str(ndcg10))
    fw.write ('\n')
    fw.close()

    sys.exit()
    ############################
    ############################


    print ("Whole map distance  :  %f  (km)" % (distance(lat_interval[0], lon_interval[0], lat_interval[len(lat_interval)-1], lon_interval[len(lat_interval)-1])))
    print ("Map interval #  :  %d" % (len(lat_interval)-1))
    print ("Map interval distance  :  %f (m)" % (1000*distance(lat_interval[0], lon_interval[0], lat_interval[1], lon_interval[1])))


    #print (poiIMG[0][1].shape)
    #print (poiIMG[0][0])
    #print (poiIMG[0][1])


    '''
    titleWords = tarPOI + '\n' + '(' + str(poiIMG[0][2]) + ', ' +str(poiIMG[0][3]) + ")    checkins = " + str(poiIMG[0][0])
    plt.title (titleWords)
    plt.imshow(poiIMG[0][1], cmap='gray')
    plt.show()
    '''
    print ("-------------------------------------")

    #-----------------------#

    # Depart Testing and Training Data (poiIMG)
    test = []
    train = []

    num_test = m.floor(len(poiIMG) / 10)
    num_train = len(poiIMG) - num_test

    ran = sorted(r.sample(range(0, len(poiIMG)), int(num_test)))

    counter=0
    for cnt in range(len(poiIMG)) :
        if cnt == ran[counter] :
            test.append (poiIMG[cnt])
            if counter < len(ran)-1 :
                counter += 1
        else :
            train.append (poiIMG[cnt])


    #########################
    ### Do CNN regression ###
    #########################

    ### Extract training and testing data ###
    # x_ : poi map
    # y_ : check-ins

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    side = int(2**partDEG)

    # Without doing shuffle
    for cnt in range(len(train)) :
        x_train.append (train[cnt][1])
        y_train.append (train[cnt][0])
    x_train = np.array(x_train)
    x_train = x_train.reshape (-1, 1, side, side)
    y_train = np.array(y_train)

    for cnt in range(len(test)) :
        x_test.append (test[cnt][1])
        y_test.append (test[cnt][0])
    x_test = np.array(x_test)
    x_test = x_test.reshape (-1, 1, side, side)
    y_test = np.array(y_test)

    #y_train = np_utils.to_categorical (y_train)
    #y_test = np_utils.to_categorical(y_test)


    ### Build CNN model ###
    model = Sequential()

    # Conv layer 1 (32, side, side)
    model.add (Convolution2D(
        nb_filter=32,
        nb_row=5,
        nb_col=5,
        border_mode='same', #padding
        dim_ordering='th',
        input_shape=(1, side, side)
        ))
    model.add (Activation('relu'))

    # Pooling layer 1 (max pooling, 32, side/2, side/2)
    # Need Pooling in this case ???

    model.add (MaxPooling2D(
        pool_size=(2, 2),
        strides=(2,2),
        border_mode='same'
        ))


    # Convolution layer 2 (64, side/2, side/2)
    model.add (Convolution2D(32, 7, 7, border_mode='same'))
    model.add (Activation('relu'))


    model.add (Convolution2D(64, 5, 5, border_mode='same'))
    model.add (Activation('relu'))



    # Pooling layer 2 (64, side/4, side/4)
    '''
    model.add (MaxPooling2D(pool_size=(2,2), border_mode='same'))
    '''

    # Fully connected layer 1 (64*side/4*side/4) => (1024)
    model.add(Flatten())
    #model.add(GlobalMaxPool2D())
    model.add(Dense(512))
    #model.add(Activation('relu'))

    # Fully connected layer 2 (1024) => (256)
    model.add(Dense(128))
    model.add(Activation('relu'))


    # Fully connected layer 3
    # For regression, no softmax, only one neuron output
    model.add(Dense(1))
    model.add(Activation('relu'))

    # Optimizer
    adam = Adam(lr=5e-5)    #lr = Learning rate


    # Metrics method
    model.compile(optimizer=adam,
            loss='mse', # do regression, mse instead of cross entropy
            metrics=['accuracy'])

    print ("Training--------------------------")

    # Train model
    model.fit (x_train, y_train, nb_epoch=64, batch_size=8)

    print ("Testing---------------------------")

    # Evalution
    loss, accuracy = model.evaluate (x_test, y_test)

    print ("----------------------------------")
    print ("\nTest loss  :  {}".format(loss))
    print ("\nTest accuracy  :  {}".format(accuracy))


    print ("----------------------------------")
    print ("Layers  :  {}".format(len(model.layers)))
    print ("Test Output  :")
    #print (model.layers[7].output[0])

    y_pred = model.predict (x_test)
    print(len(y_pred))
    #print (y_pred)
    yp = []
    for cnt in range(len(y_pred)) :
        yp.append (y_pred[cnt][0])

    yp_sort_pos = sorted (range(len(yp)), key=lambda k : yp[k])
    yp_sort_pos.reverse()

    print (yp)

    '''
    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [kb.function([inp], [out]) for out in outputs]
    test_ = np.random.random([1, side,side])[np.newaxis,...]
    layer_outs = [func([test_]) for func in functors]
    print(layer_outs[7])
    '''
    print ("=================")
    print ("Ground Truth  : {}".format(len(y_test)))
    print (y_test)

    y_test_sort_pos = sorted (range(len(y_test)), key=lambda k:y_test[k])
    y_test_sort_pos.reverse()

    # Calculate MSE
    accu = 0
    for cnt in range(len(y_test)) :
        accu += (abs(y_test[cnt]-yp[cnt]))**2

    print ("MSE  :  {}".format(accu/len(y_test)))
    print ("RMSE  :  {}".format((accu/len(y_test))**.5))

    print ("Ground truth pos  :  {}".format(y_test_sort_pos))
    print ("Test output pos  :  {}".format(yp_sort_pos))

    print ("NDCG10  :  {}".format(NDCG (yp_sort_pos, y_test_sort_pos, 10)))



    # Test Random NDCG
    ran = list(range(len(y_test)))
    r.shuffle(ran)
    print ("Random order pos  :  {}".format(ran))
    print ("Random NDCG10  :  {}".format(NDCG(ran, y_test_sort_pos, 10)))

    wor = []
    for cnt in range(len(y_test_sort_pos)) :
        wor.append (y_test_sort_pos[len(y_test_sort_pos)-cnt-1])
    print ("Worst NDCG10  :  {}".format(NDCG(wor, y_test_sort_pos, 10)))


