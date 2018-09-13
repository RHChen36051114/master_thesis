from svmutil import *
import math as m
import sys


def svr_buildModel (inputFile) :
    y, x = svm_read_problem (inputFile)
    model = svm_train (y, x, '-s 3 -c 5 -t 0 -e 0.001')
    return model


def svr_testResult (testFile, model) :
    y, x = svm_read_problem (testFile)
    result = svm_predict (y, x, model)
    #return [(float(result[1][1]))**0.5, result[0]]
    return result[0], y


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


if __name__ == "__main__" :
    
    model = svr_buildModel (sys.argv[1])
    testResult, answer = svr_testResult (sys.argv[2], model)
    #model = svr_buildModel ("poiMAP_Park_500m_16part_svrInput_train")
    #testResult, answer = svr_testResult ("poiMAP_Park_500m_16part_svrInput_test", model)
    
    #print ('\n', len(testResult), '\n\n', testResult)
    #print ('\n\n', answer)
    
    
    anspos = sorted (range(len(answer)), key=lambda k: answer[k])
    anspos.reverse()
    
    predpos = sorted (range(len(testResult)), key=lambda k: testResult[k])
    predpos.reverse()
    
    #print ("\n\nSVR NDCG5  :  {}".format(NDCG(predpos, anspos, 5)))
    #print ("\n\nSVR NDCG10  :  {}".format(NDCG(predpos, anspos, 10)))



    # Write to file
    fileName = "svrResult_" + sys.argv[1].split('_')[1] + '_'+ sys.argv[1].split('_')[2] + '_' + sys.argv[1].split('_')[3]
    fw = open (fileName, 'a+')
    fw.write (str(NDCG(predpos, anspos, 5)))
    fw.write ('\t')
    fw.write (str(NDCG(predpos, anspos, 10)))
    fw.write ('\n')
    fw.close()

