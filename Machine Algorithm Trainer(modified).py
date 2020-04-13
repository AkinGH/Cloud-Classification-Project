from sklearn import tree #Needed for machine learning model
import pandas as pd #Needed to easily read csv file
import numpy as np #Used for loading nested array with pixel values
from joblib import dump, load #Save trained machine learning algorithm

csvpath=r"C:\Users\Akin\Desktop\Simple.csv" #Put csv file path as shown
category=2 #1 for cloud only training , 2 for cirrus only training 

name='simple1000' #Name which represents the dataset you are training , it is added to the trained algorithms name as an identifier
modstart=0 #First image you want to start training on
modend=4400 #Last image you want to finish training on
steps=200 #The size of the steps you want to take
size=100 #The actual number of images you want to use everytime a step is taken , if steps and size are equal you wont be skipping out on images
depth=3
def modfunc(modstart,modend,steps,size): #Creates an array used for picking out the indicies of the images you want to use
    mod=[] #Need a modified array for choosing cloud trainers, they are all bunched to together
    modified=np.arange(modstart,modend,steps) #Creates the array used for selecting images
    for i in modified:
        function=np.arange(i,i+size,1) #Picks out the image range you selected for every step taken
        for n in function:
            mod.append(n) #Adds to the mod, so it stores all image indicies
    return mod

def MachineCreator(csvpath,category,mod,name,depth):
    if category==1:
        df = pd.read_csv (csvpath, header=0) #Read csv file
        col_a = list(df.eval("cloud")) #Load target values for selected header
        samples=np.load('imagedata2.npz') #Load pixel data from every image
        samples=samples["arr_0"] #Npz is for storing mutliple array so this takes out the first array , the only array
        clf=tree.DecisionTreeClassifier(max_depth=depth) #Decision tree type algorithm
        samples_m=[] #Array that is the sorted version of samples
        col_m=[] #Array that is the sorted version of col_a
        for i in mod: #Sorts samples and col_a with the indicies values from the mod function
            samples_m.append(samples[i])
            col_m.append(col_a[i])
        clf=clf.fit(samples_m,col_m) #Fit data to target
        dump(clf, ('cloud'+str(modstart)+'-'+str(modend)+','+str(steps)+','+str(size)+name+'.joblib')) #Save training algorithm with customised name
    elif category==2:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cirrus")) 
        samples=np.load('imagedata2.npz')
        samples=samples["arr_0"]
        clf=tree.DecisionTreeClassifier(max_depth=depth)
        samples_m=[]
        col_m=[]
        for i in mod:
            samples_m.append(samples[i])
            col_m.append(col_a[i])
        clf=clf.fit(samples_m,col_m)
        dump(clf, ('cirrus'+str(modstart)+'-'+str(modend)+','+str(steps)+','+str(size)+name+'.joblib'))
 

MachineCreator(csvpath,category,modfunc(modstart,modend,steps,size),name,depth)


