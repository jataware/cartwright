import geotime_classify
import os
def testoncsv():
    t= geotime_classify.GeoTimeClassify(20)
    assert t.columns_classified(os.getcwd()+'/geotime_classify/datasets/Test_1.csv')