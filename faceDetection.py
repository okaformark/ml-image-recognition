from os import listdir
from os.path import isdir
from numpy import asarray
import numpy as np
import cv2 as cv
from sklearn.preprocessing import LabelEncoder
from math import sqrt
import dlib


class FaceDetection(object):
    def __init__(self):
        
        self.detector = dlib.get_frontal_face_detector()
        self.new_path = 'C:\Python38\dataset\images'
        self.image_path = 'C:\Python38\images'
        self.directory = '\images'
        self.width = 128
        self.height = 128
        self.dim = (self.width, self.height)

# METHOD TO SAVE THE EXTRACTED IMAGES
    def save(self, img, name, bbox):
        self.x, self.y, self.w, self.h = bbox
        self.imgcrop = img[self.y:self.h, self.x:self.w]
        try:
            self.imgcrop = cv.resize(self.imgcrop,self.dim)
            cv.imwrite(name + ".jpg", self.imgcrop)
        except cv.error as e:
            print(e)

# LOADS THE ORIGINAL IMAGES USING OPEN CV
    def load_original_images(self,folder):
        self.images = list()

        for filename in listdir(self.image_path + '\\' + folder):
            self.path = self.image_path + '\\' + folder + '\\' + filename
            self.img = cv.imread(self.path)
            self.images.append(self.img)
        return self.images

# DETECTS AND EXTRACTS THE FACE FROM THE LOADED IMAGES
    def extract_face(self):
        self.folders = ['donnie','grace','janelle','me', 'mum', 'riri']
        for folder in self.folders:
            print(folder,'\n')
            self.frames = self.load_original_images(folder)
            
            for i, frame in enumerate(self.frames):
                self.grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                self.faces = self.detector(self.grey)
                self.fit = 5

                for _, face in enumerate(self.faces): 
                    self.x1, self.y1 = face.left(), face.top()
                    self.x2, self.y2 = face.right(), face.bottom()
                    cv.rectangle(frame, (self.x1, self.y1), (self.x2,self.y2), (220, 255, 220), 1)
                    #save(frame, new_path + '\\'+ folder + '\\'+ folder + str(i+1),(x1 - fit, y1 - fit, x2 + fit, y2 + fit))
                    self.save(frame, self.new_path + '\\'+ folder + '\\'+ folder + str(i) ,(self.x1, self.y1, self.x2, self.y2))

                frame = cv.resize(frame,(800,800))
                cv.imshow('img',frame)
                #cv.waitKey(0)
                print("saved")

