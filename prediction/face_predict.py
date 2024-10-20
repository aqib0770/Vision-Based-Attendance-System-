from mtcnn import MTCNN
import os
import warnings
import cv2
import numpy as np
from keras.models import load_model
import sys
import dlib
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from insightface.src.common import face_preprocess
from deepface import DeepFace
import traceback
from utils import utils

class FacePredictor:

    def __init__(self):
        self.detector = MTCNN()
        self.model = load_model('softmax_model.h5')
        self.le = pickle.loads(open('label_encoder.pickle', "rb").read())
        self.data = pickle.loads(open('embeddings.pickle', "rb").read())
        self.threshold = 0.5
        self.embeddings = self.data['embeddings']
        self.labels = self.le.fit_transform(self.data['names'])
    
    

    def predict(self):

        cosine_threshold = 0.8
        proba_threshold = 0.85
        comparing_num = 5
        frames = 0
        trackers = []
        texts = []
        print(texts, '1')

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            print(ret)
            if not ret:
                print("Failed to grab frame")
                break
            frames += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = utils.rescaleFrame(frame, 0.3)
            
            bboxes = self.detector.detect_faces(frame)

            if frames % 3 == 0:

                trackers = []
                texts = []
                
                if len(bboxes) > 0:
                    print(texts, '2')
                    for bboxe in bboxes:
                        bbox = bboxe['box']
                        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                        keypoints = bboxe['keypoints']
                        landmarks = np.array([keypoints['left_eye'][0], keypoints['right_eye'][0], keypoints['nose'][0], keypoints['mouth_left'][0], keypoints['mouth_right'][0], 
                                            keypoints['left_eye'][1], keypoints['right_eye'][1], keypoints['nose'][1], keypoints['mouth_left'][1], keypoints['mouth_right'][1]])
                        landmarks = landmarks.reshape((2, 5)).T
                        nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                        face_embedding = DeepFace.represent(nimg, model_name='Facenet', enforce_detection=False)
                        print(np.shape(face_embedding[0]['embedding']))
                        preds = self.model.predict(np.expand_dims(face_embedding[0]['embedding'], axis=0))[0]
                        preds = preds.flatten()
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = self.le.classes_[j]

                        match_class_idx = (self.labels == j)
                        
                        match_class_idx = np.where(match_class_idx)[0]
                        selected_idx  = np.random.choice(match_class_idx, comparing_num)
                        compare_emb = self.embeddings[selected_idx[0]]
                        cos_similarity = utils.CosineSimilarity(face_embedding[0]['embedding'], compare_emb[0]['embedding'])
                        if np.any(cos_similarity > cosine_threshold) and proba > proba_threshold:
                            text = name
                            utils.mark_attendance(name)
                        else:
                            text = "Unknown"
                        print(text)
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                        tracker.start_track(frame, rect)
                        trackers.append(tracker)
                        texts.append(text)
                        y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame, text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            else:
                for tracker, text in zip(trackers, texts):
                    pos = tracker.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()


# FacePredictor({}).predict()