import sys
import numpy as np


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

        data.append (temp)
        temp = []
    return partition, data



if __name__ == "__main__" :

    #part, data = readPOIIMG ("poiMAP_Park_500m_16part")
    part, data = readPOIIMG (sys.argv[1])


    fileName = sys.argv[1] + "_svrInput"
    fw = open (fileName, 'w')

    for cnt in range(len(data)) :
        fw.write (str(data[cnt][0]))
        fw.write (' ')

        temp = data[cnt][1].reshape (part**2)
        for cntIn in range (len(temp)) :
            fw.write (str(cntIn+1))
            fw.write (':')
            fw.write (str(temp[cntIn]))
            fw.write (' ')
        fw.write ('\n')

    fw.close()

