############################################################################
# LOGISTIC REGRESSION                                                      #
# Note: NJUST Machine Learning Assignment.                                 #
# Task: Logistic Classification                                            #
############################################################################

def selectDB(nDataBase):
    if nDataBase == "iris":
        data = "data/iris_x.dat"
        lable = "data/iris_y.dat"
        nClass = 3
    elif nDataBase == "exam":
        data = "data/exam_x.dat"
        lable = "data/exam_y.dat"
        nClass = 2
    else:
        data = "data/iris_x.dat"
        lable = "data/iris_y.dat"
        nClass = 3

    return data, lable, nClass
