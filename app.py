import argparse
import tkinter as tk
from tkinter import StringVar

from clientApp import collectUserImageForRegistration, getFaceEmbedding, trainModel
from collect_training_data.get_faces_from_camera import TrainingDataCollector
from get_face_embeddings.face_embeddings import GenerateFaceEmbeddings
from model_training.train_softmax import TrainSoftmax
from prediction.face_predict import FacePredictor


class RegistrationModule:
    def __init__(self):
        # self.logFileName = logFileName
        self.window = tk.Tk()
        self.window.title("Face Recognition and Tracking")
        self.window.resizable(0, 0)
        window_height = 600
        window_width = 880
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        self.window.configure(background='#ffffff')
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        header = tk.Label(self.window, text="Vision Based Attendence System", width=80, height=2, fg="white", bg="#363e75",
                          font=('times', 18, 'bold', 'underline'))
        header.place(x=0, y=0)
        studentName = tk.Label(self.window, text="Student Name", width=12, fg="white", bg="#363e75", height=2, font=('times', 15))
        studentName.place(x=80, y=140)
        self.studentNameTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.studentNameTxt.place(x=205, y=140)
        self.message = tk.Label(self.window, text="", bg="white", fg="black", width=30, height=1,
                                activebackground="#e47911", font=('times', 15))
        self.message.place(x=220, y=220)
        lbl3 = tk.Label(self.window, text="Notification : ", width=15, fg="white", bg="#363e75", height=2,
                        font=('times', 15))
        lbl3.place(x=80, y=260)
        self.message = tk.Label(self.window, text="", bg="#bbc7d4", fg="black", width=58, height=2, activebackground="#bbc7d4",
                           font=('times', 15))
        self.message.place(x=205, y=260)
        takeImg = tk.Button(self.window, text="Take Images", command=self.collectUserImageForRegistration, fg="white", bg="#363e75", width=15,
                            height=2,
                            activebackground="#118ce1", font=('times', 15, ' bold '))
        takeImg.place(x=80, y=350)
        trainImg = tk.Button(self.window, text="Train Images", command=self.trainModel, fg="white", bg="#363e75", width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '))
        trainImg.place(x=350, y=350)
        predictImg = tk.Button(self.window, text="Predict", command=self.makePrediction, fg="white", bg="#363e75",
                             width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '))
        predictImg.place(x=600, y=350)
        quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="#363e75", width=10, height=2,
                               activebackground="#118ce1", font=('times', 15, 'bold'))
        quitWindow.place(x=650, y=510)
        self.window.mainloop()

    def collectUserImageForRegistration(self):
      
        name = (self.studentNameTxt.get())
        ap = argparse.ArgumentParser()

        ap.add_argument("--faces", default=50,
                        help="Number of faces that camera will get")
        ap.add_argument("--output", default="../datasets/train/" + name,
                        help="Path to faces output")

        args = vars(ap.parse_args())

        trnngDataCollctrObj = TrainingDataCollector(args)
        trnngDataCollctrObj.collectImagesFromCamera()

        notifctn = "We have collected " + str(args["faces"]) + " images for training."
        self.message.configure(text=notifctn)

    def getFaceEmbedding(self):

        ap = argparse.ArgumentParser()

        ap.add_argument("--dataset", default="../datasets/train",
                        help="Path to training dataset")
        ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings.pickle")
        # Argument of insightface
        ap.add_argument('--image-size', default='112,112', help='')
        # ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
        ap.add_argument('--ga-model', default='', help='path to load model.')
        ap.add_argument('--gpu', default=0, type=int, help='gpu id')
        ap.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
        # ap.add_argument('--output', required=True)
        args = ap.parse_args()

        notifctn = "Generating face embeddings for training images."
        self.message.configure(text=notifctn)

        genFaceEmbdng = GenerateFaceEmbeddings(args)
        genFaceEmbdng.getfaceEmbedding()

    def trainModel(self):
        # ============================================= Training Params ====================================================== #

        ap = argparse.ArgumentParser()

        # ap = argparse.ArgumentParser()
        ap.add_argument("--embeddings", default="embeddings.pickle",
                        help="path to serialized db of facial embeddings")
        ap.add_argument("--model", default="softmax_model.h5",
                        help="path to output trained model")
        ap.add_argument("--le", default="label_encoder.pickle",
                        help="path to output label encoder")

        args = vars(ap.parse_args())

        self.getFaceEmbedding()
        faceRecogModel = TrainSoftmax(args)
        faceRecogModel.train()

        notifctn = "Model training is successful. Now you can go for prediction."
        self.message.configure(text=notifctn)

    def makePrediction(self):
        faceDetector = FacePredictor()
        faceDetector.predict()

    def close_window(self):
        self.window.destroy()

    def callback(self, url):
        webbrowser.open_new(url)


regStrtnModule = RegistrationModule()
