import numpy as np #required for array manipulation and saving
import glob #used for reading file paths , normal python caused errors because of the /
import re #used for reading file paths , normal python caused errors because of the /
from PIL import Image, ImageDraw #Used for opening images easily

length=470 #Put in the length of your images here
width=470 #Put in the width of your image here
numofimg=13087 #The number of images to store the pixel data of
a = glob.glob(r'C:\Users\Akin\Desktop\Converted\*.png') #Put file path as shown, including the *

def PixelStorer(length,width,numofimg,a):
    a = sorted(a, key=lambda x:float(re.findall("(\d+)",x)[0])) #Uses for sorting the array storing the image paths
    pixels=np.zeros(length*width) #Empty array used for storing the pixel values of one image
    samples=np.zeros((numofimg+1,length*width)) #An array of an array, number of samples is the parent array (goes up from 0 to numofimages),
    #pixel value array for each picture is the child array so its a (n samples, n pixel values of whole image) array, needed for machine learning algorithm
     
    for counter in range(0,numofimg): #Counter which will loop through each picture
        counter += 1 #Counts for the counter
        samples[counter]=pixels #Puts the pixel values (child array) into the 2 dimensional array (parent array)
        ipath=a[counter] #Cycles through the image path for each image 
        img=Image.open(ipath) #Opens the image in pandas format
        npImage=np.array(img) #Turns the format into an array with pixel values stored inside
        z=0 #Counter for storing the value of each pixel
        for x in range(0,length): #This nested for loop takes out the gamma value for each pixel in an image and stores it in an array
            for y in range(0,width):
                pixels[z]=npImage[x,y,1] #Takes out the pixel gamma value for each pixel and stores in this array
                z=z+1 #Cycles to next      

    np.savez_compressed('imagedata.npz', samples) #Saves the nested array required for the machine learning algorithm

PixelStorer(length,width,numofimg,a)
    
    
