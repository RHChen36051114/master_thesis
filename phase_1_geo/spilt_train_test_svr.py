import sys
import random as r


def readFile (fileName) :
    data = []

    with open (fileName, 'r') as f:
        for line in f:
            data.append([row for row in line.strip().split(' ')])
    f.close()
    return data



if __name__ == "__main__" :

    data = readFile (sys.argv[1])

    num_test = int(len(data)/10)
    num_train = len(data) - num_test
    ran = sorted (r.sample(range(0, len(data)), num_test))

    train = []
    test = []
    counter = 0
    for cnt in range (len(data)) :
        if cnt == ran[counter] :
            test.append (data[cnt])
            if counter < len(ran)-1 :
                counter += 1
        else :
            train.append (data[cnt])


    fileName_tr = sys.argv[1] + "_train"
    fw_tr = open (fileName_tr, 'w')
    for cnt in range (len(train)) :
        for cntIn in range(len(train[cnt])) :
            fw_tr.write (train[cnt][cntIn])
            fw_tr.write (' ')
        fw_tr.write ('\n')
    fw_tr.close()


    fileName_te = sys.argv[1] + "_test"
    fw_te = open (fileName_te, 'w')
    for cnt in range (len(test)) :
        for cntIn in range(len(test[cnt])) :
            fw_te.write (test[cnt][cntIn])
            fw_te.write (' ')
        fw_te.write ('\n')
    fw_te.close()

