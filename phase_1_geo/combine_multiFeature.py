import sys
import numpy as np
import phase1_1_genpoiimg_function as p1_1
import phase1_2_cnn_function as p1_2


def arg_setting (argv) :

    if len(argv) <= 2 :
        print ("\n\t[Argument error]  :  need data amounts and data path\n")
        sys.exit()

    if len(argv) != (int(argv[1])+2) :
        print ("\n\tUsage  :  python combine_multiFeature.py [combine data #] [combine data] ...\n")
        sys.exit()

    data_list = []
    for cnt in range(2, int(argv[1])+2) :
        data_list.append (argv[cnt])

    return data_list


'''
    return [list], include the content in data list
'''
def read_img (dList) :
    data = []
    for cnt in range(len(dList)) :
        part, data_ = p1_2.readPOIIMG(dList[cnt])
        data.append (data_)

    return data, int(part)


'''
    input  :  [ [POI IMG DATA 1],
                [POI IMG DATA 2],
                        .
                        .
                        .
                [POI IMG DATA N] ]

    output :  [ Combined IMG DATA ]
        Combined POI IMG  :  [ (feature 1, feature 2, ...) (tab) (feature 1, feature 2, ...) (tab) ... ]
'''
def combine_imgDATA (data, part) :

    cdata = []
    poimap = []
    temp = []

    # invarience : label, lati, longi
    # difference : POI IMG

    # initialize multi-dimension POI MAP
    for cnt in range (part**2) :
        poimap.append ([])


    # loop 1 : same feature, whole targets
    for cnt in range (len(data[0])) :


        # temp[0]:label,  temp[1]:lati,  temp[2]:longi
        temp.append (data[0][cnt][0])
        temp.append (data[0][cnt][2])
        temp.append (data[0][cnt][3])


        # loop 2 : add POI MAP feature from different file
        for cntIn in range (len(data)) :
            for cntIMG in range(part) :
                for cntIMGin in range (part) :
                    poimap[cntIMG*part + cntIMGin].append (data[cntIn][cnt][1][cntIMG][cntIMGin])

        cdata.append (temp)
        cdata.append (poimap)
        temp = []

        # refresh poimap
        poimap = []
        for cnt in range(part**2) :
            poimap.append ([])

    #return cdata

    '''
        If just want to write out multi-poimap file, can stop from here
        [return cdata]

        If want to do cnn process, do following
        [return label, poimap]
            [label]  :  list
            [poimap] :  ndarray (poi#, side, side, channels)
    '''
    label = []
    poimap = []

    for cnt in range (len(cdata)) :
        # label
        if cnt % 2 == 0 :
            label.append (cdata[cnt][0])
        # poimap
        else :
            poimap.append (cdata[cnt])

    poimap = np.array (poimap)
    poimap = poimap.reshape (len(poimap), part, part, len(data))

    return label, poimap



if __name__ == "__main__" :

    dataList = arg_setting (sys.argv)
    data, part = read_img (dataList)
    label, poimap = combine_imgDATA (data, part)


    print (len(label))
    print (len (poimap))
    print (poimap.shape)
    print (poimap[0][0])


