import sys
import cv2
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from datetime import datetime
from insightface.src.common import face_preprocess
from mtcnn.mtcnn import MTCNN
from utils import utils


class TrainingDataCollector:

    def __init__(self, args):
        self.args = args
        self.detector = MTCNN()

    def collectImagesFromCamera(self):
        cap = cv2.VideoCapture(0)
        faces = 0
        frames = 0
        max_faces = int(self.args['faces'])
        max_box = np.zeros(4)
        
        if not os.path.exists(self.args['output']):
            os.makedirs(self.args['output'])

        while faces < max_faces:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames += 1

            dtString = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            bboxes = self.detector.detect_faces(frame)

            if len(bboxes) > 0:
                max_area = 0
                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    keypoints = bboxe['keypoints']
                    area = (bbox[2] * bbox[0]) * (bbox[3] * bbox[1])
                    if area > max_area:
                        max_area = area
                        max_bbox = bbox
                        max_box = max_bbox[0:4]

                        if frames %3 == 0:
                            landmarks = np.array([keypoints['left_eye'][0], keypoints['right_eye'][0], keypoints['nose'][0], keypoints['mouth_left'][0], keypoints['mouth_right'][0], 
                                                keypoints['left_eye'][1], keypoints['right_eye'][1], keypoints['nose'][1], keypoints['mouth_left'][1], keypoints['mouth_right'][1]])
                            landmarks = landmarks.reshape((2, 5)).T
                            nimg = face_preprocess.preprocess(frame, max_box, landmarks, image_size='112,112')
                            cv2.imwrite(os.path.join(self.args['output'], "{}.jpg".format(dtString)), nimg)
                            cv2.rectangle(frame, (max_box[0], max_box[1]), (max_box[2], max_box[3]), (0, 255, 0), 2)
                            print("Face {} captured".format(faces + 1))
                            faces += 1

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

