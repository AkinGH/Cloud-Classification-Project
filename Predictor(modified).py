from joblib import dump, load #Used for loading trained algorithm
import pandas as pd #Needed to easily read csv file
import numpy as np #Used for loading nested array with pixel values


csvpath=r"C:\Users\Akin\Desktop\AllSky_classified\index_training.csv" #Put csv file path as shown
category=1 #1 for cloud only training , 2 for cirrus only training , 3 for both
save=r"C:\Users\Akin\Desktop\Converted"
trained='0-13000,200,100..'
cname='cloud0-13000,200,100.joblib'
sname='cirrus2.joblib'
modstart=100 #First image you want to start training on
modend=12900 #Last image you want to finish training on
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
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cloud"))
        right=0
        wrong=0
        
        m1wrong=0 #Modified Wrong counter
        
        m2wrong=0 #Modified Wrong counter
        
        m3wrong=0 #Modified Wrong counter
        samples=np.load('imagedata.npz')
        samples=samples["arr_0"]
        clf = load(cname)
        samples_m=[]
        col_m=[] #Array that is the sorted version of col_a
        results={'IMG No.':[],'Prediction':[],'Actual Value':[],'Right':[],'Wrong':[],'%':[],'M1Right':[],'M1Wrong':[],'M1%':[],'M2Right':[],'M2Wrong':[],'M2%':[],'M3Right':[],'M3Wrong':[],'M3%':[],'M1-3Right':[],'M1-3Wrong':[],'M1-3%':[],'M4-7Right':[],'M4-7Wrong':[],'M4-7%':[]} #Dictionary that is going to be used to create csv with result data
        for i in mod:
            samples_m.append(samples[i])
            col_m.append(col_a[i])
        predictions=clf.predict(samples_m)
        for i in range(0,len(predictions)):
            if predictions[i]==col_m[i]:
                right=right+1
            elif predictions[i]!=col_m[i]: 
                if sqrt(((predictions[i]-col_m[i])^2))==1: #Checks if the difference between prediction and target is more than 1
                    m1wrong=m1wrong+1 #Counts wrong values only if the difference is more than 1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i]) #Store img no for each image into our dictionary
                    results['Prediction'].append(predictions[i]) #Store predicted value for each image into our dictionary
                    results['Actual Value'].append(col_m[i]) #Store the target value for each image into our dictionary
                elif sqrt(((predictions[i]-col_m[i])^2))==2: #Checks if the difference between prediction and target is more than 1
                    m2wrong=m2wrong+1 #Counts wrong values only if the difference is more than 1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i]) #Store img no for each image into our dictionary
                    results['Prediction'].append(predictions[i]) #Store predicted value for each image into our dictionary
                    results['Actual Value'].append(col_m[i]) #Store the target value for each image into our dictionary
                elif sqrt(((predictions[i]-col_m[i])^2))==3: #Checks if the difference between prediction and target is more than 1
                    m3wrong=m3wrong+1 #Counts wrong values only if the difference is more than 1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i]) #Store img no for each image into our dictionary
                    results['Prediction'].append(predictions[i]) #Store predicted value for each image into our dictionary
                    results['Actual Value'].append(col_m[i]) #Store the target value for each image into our dictionary
                else:
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i])
                    results['Prediction'].append(predictions[i])
                    results['Actual Value'].append(col_m[i])
        m1right=right+(wrong-m1wrong) #Calculates modified right value which from the wrong values with a difference of 1
        m2right=right+(wrong-m2wrong)
        m3right=right+(wrong-m3wrong)
        m1_3wrong=m1wrong+m2wrong+m3wrong
        m1_3right=(len(predictions))-m1_3wrong
        m4_7wrong=wrong-(m1_3wrong)
        m4_7right=(len(predictions))-m4_7wrong
        m1p=(m1right/len(predictions))*100
        m2p=(m2right/len(predictions))*100
        m3p=(m3right/len(predictions))*100
        m1_3p=(m1_3right/len(predictions))*100
        m4_7p=(m4_7right/len(predictions))*100
        results['Right'].append(right) #Store our Right counter into the dictionary
        results['Wrong'].append(wrong) #Store our Wrong counter into the dictionary
        results['%'].append(((right/len(predictions))*100)) #Calculate percentage right
        results['M1Right'].append(m1right) #Store our modified Right value
        results['M1Wrong'].append(m1wrong) #Store our modified Wrong value
        results['M1%'].append(m1p) #Calculate percentage for modified right
        results['M2Right'].append(m2right)
        results['M2Wrong'].append(m2wrong)
        results['M2%'].append(m2p)
        results['M3Right'].append(m3right)
        results['M3Wrong'].append(m3wrong)
        results['M3%'].append(m3p)
        results['M1-3Right'].append(m1_3right)
        results['M1-3Wrong'].append(m1_3wrong)
        results['M1-3%'].append(m1_3p)
        results['M4-7Right'].append(m4_7right)
        results['M4-7Wrong'].append(m4_7wrong)
        results['M4-7%'].append(m4_7p)
        dcsv= pd.DataFrame({ key:pd.Series(value) for key, value in results.items() }) #Turn dictionary into a pandas dataframe 
        export_csv=dcsv.to_csv (save+'\\'+'cloud'+trained+str(modstart)+'-'+str(modend)+','+str(steps)+','+str(size)+".csv") #Turn dataframe into a csv file with customised name
        print(right)
        print(wrong)                       
    elif category==2:
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval("cirrus"))
        right=0
        wrong=0
        
        m1wrong=0 #Modified Wrong counter
        
        m2wrong=0 #Modified Wrong counter
        
        m3wrong=0 #Modified Wrong counter
        samples=np.load('imagedata.npz')
        samples=samples["arr_0"]
        clf = load(sname)
        samples_m=[]
        col_m=[] #Array that is the sorted version of col_a
        results={'IMG No.':[],'Prediction':[],'Actual Value':[],'Right':[],'Wrong':[],'%':[],'M1Right':[],'M1Wrong':[],'M1%':[],'M2Right':[],'M2Wrong':[],'M2%':[],'M3Right':[],'M3Wrong':[],'M3%':[],'M1-3Right':[],'M1-3Wrong':[],'M1-3%':[],'M4-7Right':[],'M4-7Wrong':[],'M4-7%':[]} #Dictionary that is going to be used to create csv with result data
        for i in mod:
            samples_m.append(samples[i])
            col_m.append(col_a[i])
        predictions=clf.predict(samples_m)
        for i in range(0,len(predictions)):
            if predictions[i]==col_m[i]:
                right=right+1
            elif predictions[i]!=col_m[i]: 
                if sqrt(((predictions[i]-col_m[i])^2))==1: #Checks if the difference between prediction and target is more than 1
                    m1wrong=m1wrong+1 #Counts wrong values only if the difference is more than 1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i]) #Store img no for each image into our dictionary
                    results['Prediction'].append(predictions[i]) #Store predicted value for each image into our dictionary
                    results['Actual Value'].append(col_m[i]) #Store the target value for each image into our dictionary
                elif sqrt(((predictions[i]-col_m[i])^2))==2: #Checks if the difference between prediction and target is more than 1
                    m2wrong=m2wrong+1 #Counts wrong values only if the difference is more than 1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i]) #Store img no for each image into our dictionary
                    results['Prediction'].append(predictions[i]) #Store predicted value for each image into our dictionary
                    results['Actual Value'].append(col_m[i]) #Store the target value for each image into our dictionary
                elif sqrt(((predictions[i]-col_m[i])^2))==3: #Checks if the difference between prediction and target is more than 1
                    m3wrong=m3wrong+1 #Counts wrong values only if the difference is more than 1
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i]) #Store img no for each image into our dictionary
                    results['Prediction'].append(predictions[i]) #Store predicted value for each image into our dictionary
                    results['Actual Value'].append(col_m[i]) #Store the target value for each image into our dictionary
                else:
                    wrong=wrong+1
                    results['IMG No.'].append(mod[i])
                    results['Prediction'].append(predictions[i])
                    results['Actual Value'].append(col_m[i])
        m1right=right+(wrong-m1wrong) #Calculates modified right value which from the wrong values with a difference of 1
        m2right=right+(wrong-m2wrong)
        m3right=right+(wrong-m3wrong)
        m1_3wrong=m1wrong+m2wrong+m3wrong
        m1_3right=(len(predictions))-m1_3wrong
        m4_7wrong=wrong-(m1_3wrong)
        m4_7right=(len(predictions))-m4_7wrong
        m1p=(m1right/len(predictions))*100
        m2p=(m2right/len(predictions))*100
        m3p=(m3right/len(predictions))*100
        m1_3p=(m1_3right/len(predictions))*100
        m4_7p=(m4_7right/len(predictions))*100
        results['Right'].append(right) #Store our Right counter into the dictionary
        results['Wrong'].append(wrong) #Store our Wrong counter into the dictionary
        results['%'].append(((right/len(predictions))*100)) #Calculate percentage right
        results['M1Right'].append(m1right) #Store our modified Right value
        results['M1Wrong'].append(m1wrong) #Store our modified Wrong value
        results['M1%'].append(m1p) #Calculate percentage for modified right
        results['M2Right'].append(m2right)
        results['M2Wrong'].append(m2wrong)
        results['M2%'].append(m2p)
        results['M3Right'].append(m3right)
        results['M3Wrong'].append(m3wrong)
        results['M3%'].append(m3p)
        results['M1-3Right'].append(m1_3right)
        results['M1-3Wrong'].append(m1_3wrong)
        results['M1-3%'].append(m1_3p)
        results['M4-7Right'].append(m4_7right)
        results['M4-7Wrong'].append(m4_7wrong)
        results['M4-7%'].append(m4_7p)
        dcsv= pd.DataFrame({ key:pd.Series(value) for key, value in results.items() }) #Turn dictionary into a pandas dataframe 
        export_csv=dcsv.to_csv (save+'\\'+'cirrus'+trained+str(modstart)+'-'+str(modend)+','+str(steps)+','+str(size)+".csv") #Turn dataframe into a csv file with customised name
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
modstart=0 #First image you want to start training on
modend=13000 #Last image you want to finish training on
trained='100-12900,200,100..'
cname='cloud100-12900,200,100.joblib'
sname='cirrus100-12900,200,100.joblib'
Predictor(csvpath,category,modfunc(modstart,modend,steps,size),save,trained,cname,sname)

