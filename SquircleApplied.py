import numpy as np  #Required for image manipulation as the image pixel values are stored in a numpy array, for easy array manipulation
import pandas as pd #Required for an easy way to read data from a csv file
from PIL import Image, ImageDraw #Required to draw circular mask on image as well as open images into arrays and save images from an array
import squircle #Required to turn circular image into a square

csvpath=r"C:\Users\Akin\Desktop\AllSky_classified\index_training.csv" #Put in the file path for csvfile containing image data in the form as is already there, if only converting one image just set this to any string
ipath=r"C:\Users\Akin\Desktop\AllSky_classified\train\50.jpg" #If you are converting one image put in the file path for that image in the form as is already there
folderimg=r"C:\Users\Akin\Desktop\AllSky_classified\train" #If you are converting multiple images , put in the path for the folder containing the images in the form as is already shown
sname="result.png" #Put the name that you want your result file to be named to , adding a .jpg at the end as is shown. If converting multiple images set this to any string as it is ignored in favour of names from the csv file
fileindex="filename_index" #Put in string form the name of the header of the column of the csv file containing the names of your images , If doing a single image just put any string here
cdiam=480 #Put in the diameter in pixels for the circle size you want
iterate=True #For deciding if you want to do multiple images using a csv file with image names (True) or a single image (False)

def Circlecrop(ipath,cdiam,sname): #This function creates and applies a circular mask to image in order to make a circular image
    img=Image.open(ipath)
    if cdiam>img.size[0] or cdiam>img.size[1]: #Checks if your circle diameter is larger than the width or height of the picture
        return ("Circle diameter bigger than picture width or height")  
    else: #Calculation for finding the values needed to create a circle with the diameter that the user inputed
        diamc1=img.size[0]-cdiam 
        diamc2=img.size[1]-cdiam
        a=int(diamc1/2)
        b=int(diamc2/2)
        c=img.size[0]-(a)
        d=img.size[1]-(b)
    #This creates the circular mask using the values calculated from above (a,b,c,d) and stacks it on top of the image and saves it
    npImage=np.array(img) 
    alpha = Image.new('L', img.size,0) #Creating the alpha , at the moment its just a black rectangle/square
    draw = ImageDraw.Draw(alpha) #Drawing it
    draw.pieslice([a,b,c,d],0,360,fill=255) #Drawing a circle into the alpha mask so its got a circle gap on it.
    npAlpha=np.array(alpha)
    npImage=np.dstack((npImage,npAlpha)) #Stack alpha onto the image
    return Image.fromarray(npImage).save(sname) #Save the image , neccessary as further image manipulation does not work otherwise

def Squarecrop(cdiam,sname): #Crops your masked image into a square with the width as the diameter of the circle
    npImage=Image.open(sname) 
    width, height = npImage.size 
    left = (width - cdiam)/2
    top = (height - cdiam)/2
    right = (width + cdiam)/2
    bottom = (height + cdiam)/2
    npImage=npImage.crop((left, top, right, bottom)) #Crops the image using the above information
    return (npImage.save(sname)) #Save the image , neccessary as further image manipulation does not work otherwise   

def Converter(sname): #Converts your circle containing the circular image into a new square
    image = Image.open(sname)
    image = np.asarray(image)
    converted = squircle.to_square(image,'fgs') #Function which converts the circular image into a square
    Image.fromarray(converted).save(sname) #Save the final image

def Master(csvpath,ipath,folderimg,sname,fileindex,cdiam,iterate): #This functions combines the functions above in order to give your converted image and iterates to do multiple images if you want it to
    if iterate==False: #For single image, uses the above function to create the picture
        Circlecrop(ipath,cdiam,sname)
        Squarecrop(cdiam,sname)
        Converter(sname)
        Squarecrop(cdiam-10,sname)    
    elif iterate==True: #For multiple images , reads csv file , iterates through it and performs the functions to get the square picture for each iteration
        df = pd.read_csv (csvpath, header=0)
        col_a = list(df.eval(fileindex))
        for i in col_a:
            sname=str(i)+".png"
            ipath=folderimg+'\\'+str(i)+".jpg"
            Circlecrop(ipath,cdiam,sname)
            Squarecrop(cdiam,sname)
            Converter(sname)
            Squarecrop(cdiam-10,sname)   

Master(csvpath,ipath,folderimg,sname,fileindex,cdiam,iterate)

    
