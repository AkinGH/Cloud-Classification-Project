from joblib import dump, load #Used for loading trained algorithm
import pandas as pd #Needed to easily read csv file
import numpy as np #Used for loading nested array with pixel values


csvpath=r"C:\Users\Akin\Desktop\AllSky_classified\index_training.csv" #Put csv file path as shown
category=3 #1 for cloud only training , 2 for cirrus only training , 3 for both


modstart=2000 #First image you want to start training on
modend=5000 #Last image you want to finish training on
steps=200 #The size of the steps you want to take
size=100 #The actual number of images you want to use everytime a step is taken , if steps and size are equal you wont be skipping out on images

def modfunc(modstart,modend,steps,size): #Creates an array used for picking out the indicies of the images you want to use
    mod=[] #Need a modified array for choosing cloud trainers, they are all bunched to together
    modified=np.arange(modstart,modend,steps) #Creates the array used for selecting images
    for i in modified:
        function=np.arange(i,i+size,1) #Picks out the image range you selected for every step taken
        for n in function:
            mod.append(n) #Adds to the mod, so it stores all image indicies
    return mod

def Predictor(csvpath,category,mod):
    if category==1:
        df = pd.read_csv (csvpath, header=0) #Read csv file
        col_a = list(df.eval("cloud"))  #Load target values for selected header
        right=0 #Keeps a count for correct guesses
        wrong=0 #Keeps a count for wrong guesses
        samples=np.load('imagedata.npy')#Load pixel data from every image
        clf = load('cloud.joblib') #Load trained algorithm
        func=slice(mod[0],mod[len(mod)-1]) #Creates slicing function used for picking images using the array created above
        samples=samples[func] #Slices the samples array
        col_a=col_a[func] #slices the result array to fit the samples array
        predictions=clf.predict(samples) #Predicts images in selected range
        for i in predictions: #Checks and counts right and wrong predictions
            if predictions[i]==col_a[i]:
                right=right+1
            elif predictions[i]!=col_a[i]:
                wrong=wrong+1
        print((right/(len(mod)))*100) #Displays right and wrong decisions
        print((wrong/(len(mod)))*100)                      
    elif category==2:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cirrus"))
        right=0
        wrong=0
        samples=np.load('imagedata.npy')
        clf = load('cirrus.joblib')
        func=slice(mod[0],mod[len(mod)-1]) 
        samples=samples[func] 
        col_a=col_a[func] 
        predictions=clf.predict(samples)
        for i in predictions:
            if predictions[i]==col_a[i]:
                right=right+1
            elif predictions[i]!=col_a[i]:
                wrong=wrong+1
        print((right/(len(mod)))*100)
        print((wrong/(len(mod)))*100)
    elif category==3:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cloud"))
        col_b = list(df.eval("cirrus"))
        right=0
        wrong=0
        samples=np.load('imagedata.npy')
        clf1 = load('cloud.joblib')
        func=slice(mod[0],mod[len(mod)-1]) 
        samples=samples[func] 
        col_a=col_a[func] 
        predictions=clf1.predict(samples)

        for i in predictions:
            if predictions[i]==col_a[i]:
                right=right+1
            elif predictions[i]!=col_a[i]:
                wrong=wrong+1        
        print((right/(len(mod)))*100,"cloud")
        print((wrong/(len(mod)))*100,"cloud")

        clf2 = load('cirrus.joblib')
        col_b=col_b[func] 
        predictions=clf2.predict(samples)
        right=0
        wrong=0
        for i in predictions:
            if predictions[i]==col_b[i]:
                right=right+1
            elif predictions[i]!=col_b[i]:
                wrong=wrong+1        
        print((right/(len(mod)))*100,"cloud")
        print((wrong/(len(mod)))*100,"cloud")

Predictor(csvpath,category,modfunc(modstart,modend,steps,size))
