from deepface import DeepFace
import os
import cv2
from datetime import datetime
import numpy as np
import glob
import pickle
from imutils import paths

class GenerateFaceEmbeddings:

    def __init__(self, args):
        self.args = args
        self.image_size = '112,112'
        self.model = DeepFace
        self.threshold = 0.5

    def getfaceEmbedding(self):
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(self.args.dataset))
        knownEmbeddings = []
        knownNames = []
        total = 0
       

        for (i, imagePath) in enumerate(imagePaths):
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            face_embedding = self.model.represent(image, model_name='Facenet', enforce_detection=False)
            knownEmbeddings.append(face_embedding)
            knownNames.append(name)               
            total += 1
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        print(data)
        with open('embeddings.pickle', 'wb') as f:
            f.write(pickle.dumps(data))

