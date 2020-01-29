from sklearn import tree #Needed for machine learning model
import pandas as pd #Needed to easily read csv file
import numpy as np #Used for loading nested array with pixel values
from joblib import dump, load #Save trained machine learning algorithm

csvpath=r"C:\Users\Akin\Desktop\AllSky_classified\index_training.csv" #Put csv file path as shown
category=3 #1 for cloud only training , 2 for cirrus only training , 3 for both


modstart=2000 #First image you want to start training on
modend=8000 #Last image you want to finish training on
steps=200 #The size of the steps you want to take
size=100 #The actual number of images you want to use everytime a step is taken , if steps and size are equal you wont be skipping out on images

def modfunc(modstart,modend,steps,size) #Creates an array used for picking out the indicies of the images you want to use
    mod=[] #Need a modified array for choosing cloud trainers, they are all bunched to together
    modified=np.arange(modstart,modend,steps) #Creates the array used for selecting images
    for i in modified:
        function=np.arange(i,i+size,1) #Picks out the image range you selected for every step taken
        for n in function:
            mod.append(n) #Adds to the mod, so it stores all image indicies
    return mod

def MachineCreator(csvpath,category,mod):
    if category==1:
        df = pd.read_csv (csvpath, header=0) #Read csv file
        col_a = list(df.eval("cloud")) #Load target values for selected header
        samples=np.load('imagedata.npy') #Load pixel data from every image
        clf=tree.DecisionTreeClassifier() #Decision tree type algorithm
        func=slice(mod[0],mod[len(mod)-1]) #Creates slicing function used for picking images using the array created above
        samples=samples[func] #Slices the samples array
        col_a=col_a[func] #slices the result array to fit the samples array
        clf=clf.fit(samples,col_a) #Fit data to target
        dump(clf, 'cloud.joblib') #Save training algorithm
    elif category==2:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cirrus")) 
        samples=np.load('imagedata.npy')
        clf=tree.DecisionTreeClassifier()
        func=slice(mod[0],mod[len(mod)-1])
        samples=samples[func]
        col_a=col_a[func]
        clf=clf.fit(samples,col_a)
        dump(clf, 'cirrus.joblib')
    elif category==3:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cloud"))
        col_b = list(df.eval("cirrus"))
        samples=np.load('imagedata.npy')
        clf1=tree.DecisionTreeClassifier()
        func=slice(mod[0],mod[len(mod)-1])
        samples=samples[func]
        col_a=col_a[func]
        col_b=col_b[func]
        clf1=clf1.fit(samples,col_a)
        dump(clf1, 'cloud.joblib')
        clf2=tree.DecisionTreeClassifier()
        clf2=clf2.fit(samples,col_b)
        dump(clf2, 'cirrus.joblib')



MachineCreator(csvpath,category,modfunc(modstart,modend,steps,size))


    
