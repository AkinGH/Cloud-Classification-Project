from joblib import dump, load #Used for loading trained algorithm
import pandas as pd #Needed to easily read csv file
import numpy as np #Used for loading nested array with pixel values


csvpath=r"C:\Users\Akin\Desktop\AllSky_classified\index_training.csv" #Put csv file path as shown
category=2 #1 for cloud only training , 2 for cirrus only training , 3 for both
start=1001 #first image you want to start predict from
end=1500 #last image you want to predict

def Predictor(csvpath,category,start,end):
    if category==1:
        df = pd.read_csv (csvpath, header=0) #Read csv file
        col_a = list(df.eval("cloud"))  #Load target values for selected header
        right=0 #Keeps a count for correct guesses
        wrong=0 #Keeps a count for wrong guesses
        samples=np.load('imagedata.npy')#Load pixel data from every image
        clf = load('cloud.joblib') #Load trained algorithm
        predictions=clf.predict(samples[start:end]) #Predicts images in selected range
        for i in predictions: #Checks and counts right and wrong predictions
            if predictions[i]==col_a[i+start]:
                right=right+1
            elif predictions[i]!=col_a[i+start]:
                wrong=wrong+1
        print((right/(end-start))*100) #Displays right and wrong decisions
        print((wrong/(end-start))*100)                        
    elif category==2:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cirrus"))
        right=0
        wrong=0
        samples=np.load('imagedata.npy')
        clf = load('cirrus.joblib')
        predictions=clf.predict(samples[start:end])
        for i in predictions:
            if predictions[i]==col_a[i+start]:
                right=right+1
            elif predictions[i]!=col_a[i+start]:
                wrong=wrong+1
        print((right/(end-start))*100)
        print((wrong/(end-start))*100)
    elif category==3:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cloud"))
        col_b = list(df.eval("cirrus"))
        right=0
        wrong=0
        samples=np.load('imagedata.npy')
        clf1 = load('cloud.joblib')
        predictions=clf1.predict(samples[start:end])

        for i in predictions:
            if predictions[i]==col_a[i+start]:
                right=right+1
            elif predictions[i]!=col_a[i+start]:
                wrong=wrong+1        
        print((right/(end-start))*100,"cloud")
        print((wrong/(end-start))*100,"cirrus")

        clf2 = load('cirrus.joblib')
        predictions=clf2.predict(samples[start:end])

        for i in predictions:
            if predictions[i]==col_b[i+start]:
                right=right+1
            elif predictions[i]!=col_b[i+start]:
                wrong=wrong+1        
        print((right/(end-start))*100,"cloud")
        print((wrong/(end-start))*100,"cirrus")

Predictor(csvpath,category,start,end)
