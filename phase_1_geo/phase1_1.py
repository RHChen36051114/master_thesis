'''
    phase 1_1  :  generate POI IMAGE and write it to file
'''
import sys
import time as t
import phase1_1_genpoiimg_function as p1_1


if __name__ == "__main__" :

    ts = t.time()
    tarPOI, poiMAP, radius, partDEG = p1_1.arg_setting (sys.argv)
    specPOI = p1_1.get_spec (poiMAP, tarPOI)
    print (tarPOI)


    # Setting Competitiveness venue type of experiment targets
    comp_venu_st = ["Coffee Shop"]
    comp_venu_mc = ["Fast Food", "Burgers"]
    comp_venu_dd = ["Coffee Shop", "Donuts", "Breakfast", "CaFA", "Bakery", "Ice Cream"]


    # find fitness parameter for radius (cause by error of distance calculation)
    r_act = .1
    sideLEN = 0
    while (abs(radius*2 - sideLEN)) > 0.001 :
        r_act += 0.001
        # correctness distance arg
        lat_inte, lon_inte = p1_1.find_poi_array_interval (float(specPOI[0][0]), float(specPOI[0][1]), r_act, partDEG)
        sideLEN = p1_1.distance (lat_inte[0], lon_inte[0], lat_inte[len(lat_inte)-1], lon_inte[len(lon_inte)-1])
    r_act = float("{0:.4f}".format(r_act))


    # generate poi IMG
    poiIMG = []
    cnt = 1
    for row in specPOI :
        print ("%d / %d" % (cnt, len(specPOI)), end='\r')

        # return target poi map range
        # .38 means radius = 500m, side length = 1km
        lat_interval, lon_interval = p1_1.find_poi_array_interval (float(row[0]), float(row[1]), r_act, partDEG)

        # return target poi img
        # [0]: # of check-ins,  [1]: accu poi MAP (seen using function),  [2][3]: center lati & longi
        # filter check-ins = 1
        if int(row[4]) > 1 :
            poiIMG.append ([int(row[4]), p1_1.accu_checkin(lat_interval, lon_interval, partDEG, poiMAP, row[0], row[1]), row[0], row[1]])
            #poiIMG.append ([int(row[4]), p1_1.accu_competitive(lat_interval, lon_interval, partDEG, poiMAP, comp_venu_st, row[0], row[1]), row[0], row[1]])
        cnt += 1


    # print out poi IMG message
    print ("%d / %d" % (cnt-1, len(specPOI)))
    print ("Map border distance  :  %f  (km)" % (p1_1.distance(lat_interval[0], lon_interval[0], lat_interval[len(lat_interval)-1], lon_interval[len(lon_interval)-1])))
    print ("Map interval #  :  %d" % (len(lat_interval)-1))
    print ("Map interval distance  :  %f (m)" % (1000*p1_1.distance(lat_interval[0], lon_interval[0], lat_interval[1], lon_interval[1])))


    # write poi img to file
    fileName = "poiMAP_" + tarPOI + '_' + str(int(radius*2*1000)) + "m_" + str(2**partDEG) + "part"
    p1_1.writePOIIMG (poiIMG, fileName, 2**partDEG)

    te = t.time()
    print ("Processing Time  :  %f  (secs)" % (te-ts))

