'''
    phase1_2  :  read POI IMAGE file and do CNN processing
'''
import sys
import random as r
import time as t
import phase1_2_cnn_function as p1_2

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, MaxPooling2D, GlobalMaxPool2D
from keras.optimizers import Adam
from keras import backend as kb


if __name__ == "__main__" :

    ts = t.time()
    poiTYPE, side, poiIMG = p1_2.arg_setting (sys.argv)
    x_train, y_train, x_test, y_test = p1_2.departTrainTest (poiIMG, side)


    # Build CNN model
    model = Sequential()

    # Conv layer 1
    model.add (Convolution2D(32, 5, 5, border_mode='same', dim_ordering='th', input_shape=(1, side, side)))
    model.add (Activation('relu'))

    # Pooling layer 1
    model.add (MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same'))

    # Convolution layer 2
    model.add (Convolution2D(32, 7, 7, border_mode='same'))
    model.add (Activation('relu'))

    # Convolution layer 3
    model.add (Convolution2D(64, 5, 5, border_mode='same'))
    model.add (Activation('relu'))

    # Fully Connected layer 1
    model.add (Flatten())

    # Fully connected layer 2
    model.add (Dense(512))

    # Fully connected layer 3
    model.add (Dense(128))
    model.add (Activation('relu'))

    # Output layer
    model.add (Dense(1))
    model.add (Activation('relu'))


    # Optimizer
    adam = Adam(lr=5e-5)


    # Metrics method
    model.compile (optimizer=adam, loss='mse', metrics=['accuracy'])


    print ("------------- Training ---------------")

    # Train model
    model.fit (x_train, y_train, nb_epoch=64, batch_size=8)


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
    print ("NDCG5  :  {}".format(ndcg5))
    print ("NDCG10  :  {}".format(ndcg10))
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
    fw.write (str(ran_ndcg5))
    fw.write ('\t')
    fw.write (str(ran_ndcg10))
    fw.write ('\n')
    fw.close()


    te = t.time()
    print ("Program excecuting time  :  {}  (secs)".format(te-ts))

