"""
Functions for processing images taken by Leopard camera.
"""
import numpy as np
import cv2
import glob
import os
import re

def raw_greyscale_array(fileName, rows, cols, bit_depth = np.uint8):
    """
    Reads .raw greyscale image and outputs image as numpy array.
    """
    with open(fileName, 'rb') as f:
        img = np.fromfile(f, dtype = bit_depth, count = rows*cols)   
    return img.reshape((rows, cols))

def raw_2_pgm(directory, rows, cols, bit_depth = np.uint8, stereo = False):
    """
    Find all .raw images in given directory, and saves them as .pgm files in a
    new directory "PGM".
    """
    raw_imgs = glob.glob(os.path.join(directory, "*.raw"))
    
    for raw_img in raw_imgs:
        print("Processing: ", raw_img)
        
        path, fileName = os.path.split(raw_img)
        newPath = os.path.join(path, "PGM")
        
        if not os.path.isdir(newPath):
            os.mkdir(newPath)
            print("New directory created at: ", newPath)
        
        items = re.match("([a-z\-]+)([0-9]+).([a-z]+)", fileName).groups()
        img = raw_greyscale_array(raw_img, rows, cols, bit_depth)
        
        if stereo:
            fName1 = os.path.join(newPath,"left{}.pgm".format(items[1]))
            fName2 = os.path.join(newPath,"right{}.pgm".format(items[1]))
            cv2.imwrite(fName1, img[:,int(cols/2):])
            cv2.imwrite(fName2, img[:,:int(cols/2)])
        else:
            fName = os.path.join(newPath,"{0}{1}.pgm".format(items[0],items[1]))
            cv2.imwrite(fName, img)
    return
    
if __name__ == "__main__":
    directory = r"C:/Users/dhen2714/OneDrive - The University of Sydney (Students)/Phd/Experiments/VolunteerForeheadTracking/Tests/PreSetup/20171220_Calibration_intrinsic/Images"
    
    raw_2_pgm(directory, 480, 1280, stereo = True)

