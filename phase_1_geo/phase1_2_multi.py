'''
    phase1_2_multi  :  read plural POI IMAGE files and do multi-channel CNN processing
'''
import sys
import random as r
import time as t
import phase1_2_cnn_function as p1_2
import combine_multiFeature as mu

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D, BatchNormalization

from keras.optimizers import Adam
from keras import backend as kb


if __name__ == "__main__" :

    ts = t.time()

    #################
    # Preprocessing #
    #################
    # Read plural poimap, depart train, test
    dataList = mu.arg_setting (sys.argv)
    data, part = mu.read_img (dataList)
    label, poimap = mu.combine_imgDATA (data, part)
    x_train, y_train, x_test, y_test = p1_2.departTrainTest_multi (label, poimap)


    ###################
    # Build CNN model #
    ###################
    model = Sequential()

    # Conv layer 1
    model.add (Conv2D(32, (5, 5), padding='same',
                input_shape=x_train.shape[1:]))
    model.add (Activation('relu'))

    # Batch Normalization layer 1
    #model.add (BatchNormalization())

    # Pooling layer 1
    #model.add (MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same'))
    #model.add (Dropout(0.2))
    model.add (MaxPooling2D(2,2))

    # Convolution layer 2
    model.add (Conv2D(48, (3, 3), padding='same'))
    model.add (Activation('relu'))

    # Batch Normalization layer 2
    #model.add (BatchNormalization())

    # Convolution layer 3
    model.add (Conv2D(48, (3, 3), padding='same'))
    model.add (Activation('relu'))

    # Batch Normalization layer 3
    #model.add (BatchNormalization())

    # Fully Connected layer 1
    model.add (Flatten())

    # Fully connected layer 2
    model.add (Dense(1024))
    model.add (Activation('relu'))

    # Fully connected layer 3
    model.add (Dense(512))
    model.add (Activation('relu'))

    # Fully connected layer 4
    model.add (Dense(128))
    model.add (Activation('relu'))

    # Output layer
    model.add (Dense(1))
    model.add (Activation('relu'))

    model.summary()

    #################################
    # Results and Optimizer Setting #
    #################################
    # Optimizer
    adam = Adam(lr=5e-5)

    # Metrics method
    model.compile (optimizer=adam, loss='mse', metrics=['accuracy'])


    print ("------------- Training ---------------")

    # Train model
    model.fit (x_train, y_train, epochs=64, batch_size=8)


    print ("-------------- Testing ---------------")

    # Evalution
    loss, accuracy = model.evaluate (x_test, y_test)


    print ("--------------------------------------")
    print ("Layers  :  {}".format(len(model.layers)))
    print ("Tess loss  :  {}".format(loss))
    print ("Test accuracy  :  {}".format(accuracy))


    # Regression Prediction
    y_pred = model.predict (x_test)


    print (len(y_pred))
    yp = []
    for cnt in range(len(y_pred)) :
        yp.append (y_pred[cnt][0])


    print ("Ground Truth  :  {}".format(len(y_test)))
    print (y_test)
    print ("Predict  :")
    print (yp)


    # if training failed, don't record the result
    zero = 0
    for cnt in range(len(yp)) :
        if yp[cnt] == 0 :
            zero += 1
    if zero >= len(yp)*0.7 :
        print ("\nTraining Failed\n")
        sys.exit()


    # Get result index (ordered)
    yp_sort_pos = sorted (range(len(yp)), key=lambda k: yp[k])
    yp_sort_pos.reverse()

    y_test_sort_pos = sorted (range(len(y_test)), key=lambda k:y_test[k])
    y_test_sort_pos.reverse()

    ran = list(range(len(y_test)))
    r.shuffle (ran)

    
    print ("Ground Truth position  :  {}".format(y_test_sort_pos))
    print ("Test output position  :  {}".format(yp_sort_pos))
    print ("Random position  :  {}".format(ran))
    print ("------------------------------------------------------")
    ndcg5 = p1_2.NDCG(yp_sort_pos, y_test_sort_pos, 5)
    ndcg10 = p1_2.NDCG(yp_sort_pos, y_test_sort_pos, 10)
    ndcg15 = p1_2.NDCG(yp_sort_pos, y_test_sort_pos, 15)
    ndcg20 = p1_2.NDCG(yp_sort_pos, y_test_sort_pos, 20)
    print ("NDCG5  :  {}".format(ndcg5))
    print ("NDCG10  :  {}".format(ndcg10))
    print ("NDCG15  :  {}".format(ndcg15))
    print ("NDCG20  :  {}".format(ndcg20))
    print ("------------------------------------------------------")
    ran_ndcg5 = p1_2.NDCG(ran, y_test_sort_pos, 5)
    ran_ndcg10 = p1_2.NDCG(ran, y_test_sort_pos, 10)
    print ("Random NDCG5  :  {}".format(ran_ndcg5))
    print ("Random NDCG10  :  {}".format(ran_ndcg10))
    print ("------------------------------------------------------")

    wor = []
    for cnt in range(len(y_test_sort_pos)) :
        wor.append (y_test_sort_pos[len(y_test_sort_pos)-cnt-1])
    print ("Worst NDCG5  :  {}".format(p1_2.NDCG(wor, y_test_sort_pos, 5)))
    print ("Worst NDCG10  :  {}".format(p1_2.NDCG(wor, y_test_sort_pos, 10)))


    # Write result to file
    fileName = "testResult_phase1_test1_" + poiTYPE
    fw = open (fileName, 'a+')
    fw.write (str(ndcg5))
    fw.write ('\t')
    fw.write (str(ndcg10))
    fw.write ('\t')
    fw.write (str(ndcg15))
    fw.write ('\t')
    fw.write (str(ndcg20))
    fw.write ('\n')
    fw.close()


    te = t.time()
    print ("Program excecuting time  :  {}  (secs)".format(te-ts))

