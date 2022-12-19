from sklearn import svm 

def SVM():
    return svm.SVC(gamma=0.001, C=100., probability=True,decision_function_shape='ovo',kernel='rbf')