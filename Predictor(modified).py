from joblib import dump, load #Used for loading trained algorithm
import pandas as pd #Needed to easily read csv file
import numpy as np #Used for loading nested array with pixel values


csvpath=r"C:\Users\Akin\Desktop\AllSky_classified\index_training.csv" #Put csv file path as shown
category=2 #1 for cloud only training , 2 for cirrus only training , 3 for both
save=r"C:\Users\Akin\Desktop\Converted"
trained='0-13000,200,100..'
cname='cloud.joblib'
sname='cirrus.joblib'
modstart=300 #First image you want to start training on
modend=4300 #Last image you want to finish training on
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

def Predictor(csvpath,category,mod,save,trained,cname,sname):
    if category==1:
        df = pd.read_csv (csvpath, header=0) #Reads csv file
        col_a = list(df.eval("cloud")) #Loads column data into an array , these are the target values
        right=0 #Right counter
        wrong=0 #Wrong counter
        mright=0 #Modified Right counter 
        mwrong=0 #Modified Wrong counter
        samples=np.load('imagedata.npz') #Load pixel data
        samples=samples["arr_0"] #Take out the first and only array, npz is for multiple arrays
        clf = load(cname) #Load trained algorithm
        samples_m=[] #Array that is the sorted version of samples
        col_m=[] #Array that is the sorted version of col_a
        results={'IMG No.':[],'Prediction':[],'Actual Value':[],'Right':[],'Wrong':[],'%':[],'MRight':[],'MWrong':[],'M%':[]} #Dictionary that is going to be used to create csv with result data
        for i in mod: #Sorts samples and col_a with the indicies values from the mod function
            samples_m.append(samples[i])
            col_m.append(col_a[i])
        predictions=clf.predict(samples_m) #Predicts values from the sorted array
        for i in range(0,len(predictions)): #Checks if predicted value is right or wrong and if it is wrong 
            if predictions[i]==col_m[i]:
                right=right+1
            elif predictions[i]!=col_m[i]: 
                if ((predictions[i]-col_m[i])^2)>1: #Checks if the difference between prediction and target is more than 1
                    mwrong=mwrong+1 #Counts wrong values only if the difference is more than 1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i]) #Store img no for each image into our dictionary
                    results['Prediction'].append(predictions[i]) #Store predicted value for each image into our dictionary
                    results['Actual Value'].append(col_m[i]) #Store the target value for each image into our dictionary
                else:
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i])
                    results['Prediction'].append(predictions[i])
                    results['Actual Value'].append(col_m[i])
        mright=right+(wrong-mwrong) #Calculates modified right value which ignores wrong values with a difference of 1
        results['Right'].append(right) #Store our Right counter into the dictionary
        results['Wrong'].append(wrong) #Store our Wrong counter into the dictionary
        results['%'].append((right/len(predictions)*100)) #Calculate percentage right
        results['MRight'].append(mright) #Store our modified Right value
        results['MWrong'].append(mwrong) #Store our modified Wrong value
        results['M%'].append((mright/len(predictions)*100)) #Calculate percentage for modified right
        dcsv= pd.DataFrame({ key:pd.Series(value) for key, value in results.items() }) #Turn dictionary into a pandas dataframe 
        export_csv=dcsv.to_csv (save+'\\'+'cloud'+trained+str(modstart)+'-'+str(modend)+','+str(steps)+','+str(size)+".csv") #Turn dataframe into a csv file with customised name
        print(right)
        print(wrong)                    
    elif category==2:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cirrus"))
        right=0
        wrong=0
        mright=0
        mwrong=0
        samples=np.load('imagedata.npz')
        samples=samples["arr_0"]
        clf = load(sname)
        samples_m=[]
        col_m=[]
        results={'IMG No.':[],'Prediction':[],'Actual Value':[],'Right':[],'Wrong':[],'%':[],'MRight':[],'MWrong':[],'M%':[]}
        for i in mod:
            samples_m.append(samples[i])
            col_m.append(col_a[i])
        predictions=clf.predict(samples_m)
        for i in range(0,len(predictions)):
            if predictions[i]==col_m[i]:
                right=right+1
            elif predictions[i]!=col_m[i]:
                if ((predictions[i]-col_m[i])^2)>1:
                    mwrong=mwrong+1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i])
                    results['Prediction'].append(predictions[i])
                    results['Actual Value'].append(col_m[i])
                else:
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i])
                    results['Prediction'].append(predictions[i])
                    results['Actual Value'].append(col_m[i])
        mright=right+(wrong-mwrong)
        results['Right'].append(right)
        results['Wrong'].append(wrong)
        results['%'].append((right/len(predictions)*100))
        results['MRight'].append(mright)
        results['MWrong'].append(mwrong)
        results['M%'].append((mright/len(predictions)*100))
        dcsv= pd.DataFrame({ key:pd.Series(value) for key, value in results.items() })
        export_csv=dcsv.to_csv (save+'\\'+'cirrus'+trained+str(modstart)+'-'+str(modend)+','+str(steps)+','+str(size)+".csv")
        print(right)
        print(wrong)
    elif category==3:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cloud"))
        col_b = list(df.eval("cirrus"))
        right=0
        wrong=0
        mright=0
        mwrong=0
        samples=np.load('imagedata.npz')
        samples=samples["arr_0"]
        clf = load(cname)
        samples_m=[]
        col_m=[]
        col_n=[]
        results={'IMG No.':[],'Prediction':[],'Actual Value':[],'Right':[],'Wrong':[],'%':[],'MRight':[],'MWrong':[],'M%':[]}
        for i in mod:
            samples_m.append(samples[i])
            col_m.append(col_a[i])
            col_n.append(col_a[i])
        predictions=clf.predict(samples_m)
        for i in range(0,len(predictions)):
            if predictions[i]==col_m[i]:
                right=right+1
            elif predictions[i]!=col_m[i]:
                if ((predictions[i]-col_m[i])^2)>1:
                    mwrong=mwrong+1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i])
                    results['Prediction'].append(predictions[i])
                    results['Actual Value'].append(col_m[i])
                else:
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i])
                    results['Prediction'].append(predictions[i])
                    results['Actual Value'].append(col_m[i])
        mright=right+(wrong-mwrong)
        results['Right'].append(right)
        results['Wrong'].append(wrong)
        results['%'].append((right/len(predictions)*100))
        results['MRight'].append(mright)
        results['MWrong'].append(mwrong)
        results['M%'].append((mright/len(predictions)*100))
        dcsv= pd.DataFrame({ key:pd.Series(value) for key, value in results.items() })
        export_csv=dcsv.to_csv (save+'\\'+'cloud'+trained+str(modstart)+'-'+str(modend)+','+str(steps)+','+str(size)+".csv")
        print(right)
        print(wrong)
        clf = load(sname)
        predictions=clf.predict(samples_m)
        right=0
        wrong=0
        mright=0
        mwrong=0
        for i in range(0,len(predictions)):
            if predictions[i]==col_n[i]:
                right=right+1
            elif predictions[i]!=col_n[i]:
                if ((predictions[i]-col_n[i])^2)>1:
                    mwrong=mwrong+1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i])
                    results['Prediction'].append(predictions[i])
                    results['Actual Value'].append(col_n[i])
                else:
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i])
                    results['Prediction'].append(predictions[i])
                    results['Actual Value'].append(col_n[i])
        mright=right+(wrong-mwrong)
        results['Right'].append(right)
        results['Wrong'].append(wrong)
        results['%'].append((right/len(predictions)*100))
        results['MRight'].append(mright)
        results['MWrong'].append(mwrong)
        results['M%'].append((mright/len(predictions)*100))
        dcsv= pd.DataFrame({ key:pd.Series(value) for key, value in results.items() })
        export_csv=dcsv.to_csv (save+'\\'+'cirrus'+trained+str(modstart)+'-'+str(modend)+','+str(steps)+','+str(size)+".csv")
        print(right)
        print(wrong)
        
Predictor(csvpath,category,modfunc(modstart,modend,steps,size),save,trained,cname,sname)

