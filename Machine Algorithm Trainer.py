from sklearn import tree #Needed for machine learning model
import pandas as pd #Needed to easily read csv file
import numpy as np #Used for loading nested array with pixel values
from joblib import dump, load #Save trained machine learning algorithm

csvpath=r"C:\Users\Akin\Desktop\AllSky_classified\index_training.csv" #Put csv file path as shown
category=3 #1 for cloud only training , 2 for cirrus only training , 3 for both
start=0 #First image you want to start training from
end=1000 #Last image you want to finish training on
def MachineCreator(csvpath,category,start,end):
    if category==1:
        df = pd.read_csv (csvpath, header=0) #Read csv file
        col_a = list(df.eval("cloud")) #Load target values for selected header
        samples=np.load('imagedata.npy') #Load pixel data from every image
        clf=tree.DecisionTreeClassifier() #Decision tree type algorithm
        clf=clf.fit(samples[start:end],col_a[start:end]) #Fit data to target
        dump(clf, 'cloud.joblib') #Save training algorithm
    elif category==2:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cirrus")) 
        samples=np.load('imagedata.npy')
        clf=tree.DecisionTreeClassifier()
        clf=clf.fit(samples[start:end],col_a[start:end])
        dump(clf, 'cirrus.joblib')
    elif category==3:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cloud"))
        col_b = list(df.eval("cirrus"))
        samples=np.load('imagedata.npy')
        clf1=tree.DecisionTreeClassifier()
        clf1=clf1.fit(samples[start:end],col_a[start:end])
        dump(clf1, 'cloud.joblib')
        clf2=tree.DecisionTreeClassifier()
        clf2=clf2.fit(samples[start:end],col_b[start:end])
        dump(clf2, 'cirrus.joblib')
                    
MachineCreator(csvpath,category,start,end)
