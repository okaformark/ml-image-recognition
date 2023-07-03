from os import listdir
from os.path import isdir
from numpy import asarray
import numpy as np
import cv2 as cv
from sklearn.preprocessing import LabelEncoder
from math import sqrt


class FaceRecognition(object):
    def __init__(self):
        self.image_path = 'C:\Python38\dataset\images'
        self.width = 128
        self.height = 128
        self.dim = (self.width, self.height)

# READS, RESIZE AND NORMALIZE THE EXTRACTED IMAGES
    def load_images(self,directory):
        self.images = list()

        for filename in listdir(directory):
            self.path = directory + filename
            self.img = cv.imread(self.path)
            self.resized = cv.resize(self.img, self.dim, interpolation = cv.INTER_AREA)
            self.normalized_image = cv.normalize(self.resized,None, 0, 1, cv.NORM_MINMAX, dtype = cv.CV_32F)
            self.arr = np.array(self.normalized_image)
            self.newarr = self.arr.reshape(-1)
            self.images.append(self.newarr)
        return self.images

# LABELS THE IMAGES AND STOES IN LIST
    def load_dataset(self, directory):
        self.X, self.Y = list(), list()
        for subdir in listdir(directory):
            self.path = directory  + '\\' + subdir + '\\'
            if not isdir(self.path):
                continue
            self.images = self.load_images(self.path)
            self.labels = [subdir+str(i+1) for i in range(len(self.images))]
            print('Loaded %d examples for class: %s' % (len(self.images), subdir))
            print('Label name:\n', self.labels)

            self.X.extend(self.images)
            self.Y.extend(self.labels)
        return asarray(self.X), asarray(self.Y)

# METHOD TO DERIVE THE EUCLIDEAN DISTANCE
    def euclidean_distance(self, row1, row2):
        self.distance = 0.0
        for i in range(len(row1) - 1):
            self.distance += (row1[i] - row2[i]) ** 2
        return sqrt(self.distance)

# GET NEIGHBORS
    def get_neighbors(self,train, test_row, num_neighbors):
        self.distances = list()

        for train_row in train:
            self.dist = self.euclidean_distance(test_row, train_row)
            self.distances.append((train_row, self.dist))
        self.distances.sort(key=lambda tup:tup[1])
        self.neighbors = []

        for i in range(num_neighbors):
            self.neighbors.append(self.distances[i][0])
        return self.neighbors

# PREDICT CLASSIFICATION
    def predict_classification(self,train, test_row, num_neighbors):
        self.neighbors = self.get_neighbors(train, test_row, num_neighbors)
        self.output_values = [row[-1] for row in self.neighbors]
        self.prediction = max(set(self.output_values), key = self.output_values.count)
        return self.prediction
            

# TEST UNKNOWN IMAGES AGAINST THE TRAINED DATA
    def test(self):
        self.folders = ['donnie','grace','janelle','me', 'mum', 'riri']

        for i in range(len(self.folders) - 1):
            self.trainX, self.trainY = self.load_dataset(self.image_path)
            self.le = LabelEncoder()
            self.label = self.le.fit_transform(self.trainY)
            self.labelCol = self.label[:, np.newaxis]
            self.data = np.concatenate((self.trainX, self.labelCol), axis = 1)
            print(self.data.shape)

        for i in range(len(self.folders)):

            self.img = cv.imread('C:\Python38\dataset' + '\\test' + '\\unknown'+ str(i+1) + '.jpg')
            self.resized = cv.resize(self.img, self.dim, interpolation = cv.INTER_AREA)
            self.normalized_image = cv.normalize(self.resized,None, 0, 1, cv.NORM_MINMAX, dtype = cv.CV_32F)
            self.arr = np.array(self.normalized_image)
            self.imgUnknown = self.arr.reshape(-1)

            self.expect = 0
            self.prediction = self.predict_classification(self.trainX, self.imgUnknown, 4)
            print('\n======\nExpected %d, predictedd %d.' %(self.expect, self.prediction))


            
    
