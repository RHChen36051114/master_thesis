import sys
import numpy as np
import random as r
import math as m


'''
    Argument read & check
'''
def arg_setting (argv) :
    if len(argv) != 2 :
        print ("\nArgument # error\n\n\tUsage  :  python  phase1_2_cnn_function.py  [POI Map File]\n")
    poiTYPE = argv[1].split('_')[1]
    side, poiIMG = readPOIIMG(argv[1])

    return poiTYPE, side, poiIMG


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
    Read poi img file that generated from phase1_1
'''
def readPOIIMG (fileName) :
    rawdata = readFile (fileName)

    data = []

    loop_cnt = int(rawdata[0][0])
    partition = int(rawdata[1][0])

    temp = []
    for cnt in range(3, len(rawdata), partition+2) :
        # [0] : label (e.g.  center check-ins)
        temp.append (int(rawdata[cnt][0]))

        # [1] : poi map(img)
        arr = []
        for cntIn in range(2, partition+2) :
            arr.append ([float(t) for t in rawdata[cnt+cntIn]])
        arr = np.array(arr)
        temp.append (arr)

        # [2][3] : latitude, longitude
        temp.append (float(rawdata[cnt+1][0]))
        temp.append (float(rawdata[cnt+1][1]))

        data.append (temp)
        temp = []

    return partition, data


'''
    Input  :  poi img list
    Output :  training and testing data for CNN input
'''
def departTrainTest (poiIMG, side) :
    
    num_test = int(len(poiIMG)/10)
    num_train = len(poiIMG) - num_test
    ran = sorted (r.sample(range(0, len(poiIMG)), num_test))

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    counter=0
    for cnt in range(len(poiIMG)) :
        if cnt == ran[counter] :
            x_test.append (poiIMG[cnt][1])
            y_test.append (poiIMG[cnt][0])
            if counter < len(ran)-1 :
                counter += 1
        else :
            x_train.append (poiIMG[cnt][1])
            y_train.append (poiIMG[cnt][0])

    x_train = np.array(x_train)
    x_train = x_train.reshape (-1, 1, side, side)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_test = x_test.reshape(-1, 1, side, side)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


'''
    Return NDCG
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

