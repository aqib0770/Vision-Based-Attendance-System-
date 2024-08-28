import argparse
import tkinter as tk
from tkinter import StringVar
import webbrowser

from collect_training_data.get_faces_from_camera import TrainingDataCollector
from get_face_embeddings.face_embeddings import GenerateFaceEmbeddings
from model_training.train_softmax import TrainSoftmax
from prediction.face_predict import FacePredictor


class RegistrationModule:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Face Recognition and Tracking")
        self.window.resizable(0, 0)
        window_height = 600
        window_width = 880
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        self.window.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
        self.window.configure(background='#ffffff')

        header = tk.Label(self.window, text="Vision Based Attendance System", fg="white", bg="#363e75",
                          font=('Arial', 30, 'bold', 'underline'))
        header.pack(pady=20)

        student_frame = tk.Frame(self.window, bg="#ffffff")
        student_frame.pack(pady=20)
        
        studentName = tk.Label(student_frame, text="Student Name:", fg="white", bg="#363e75", font=('Arial', 15))
        studentName.grid(row=0, column=0, padx=10, pady=5)
        
        self.studentNameTxt = tk.Entry(student_frame, width=20, font=('times', 15))
        self.studentNameTxt.grid(row=0, column=1, padx=10, pady=5)
        
        self.message = tk.Label(self.window, text="", bg="#bbc7d4", fg="black", width=58, height=2, 
                                font=('times', 15))
        self.message.pack(pady=10)
        
        button_frame = tk.Frame(self.window, bg="#ffffff")
        button_frame.pack(pady=20)
        
        takeImg = tk.Button(button_frame, text="Take Images", command=self.collectUserImageForRegistration, fg="white", bg="#363e75",
                            width=15, height=2, activebackground="#118ce1", font=('Arial', 15, 'bold'))
        takeImg.grid(row=0, column=0, padx=10, pady=10)
        
        trainImg = tk.Button(button_frame, text="Train Model", command=self.trainModel, fg="white", bg="#363e75",
                             width=15, height=2, activebackground="#118ce1", font=('Arial', 15, 'bold'))
        trainImg.grid(row=0, column=1, padx=10, pady=10)
        
        predictImg = tk.Button(button_frame, text="Start Attendance", command=self.makePrediction, fg="white", bg="#363e75",
                               width=15, height=2, activebackground="#118ce1", font=('Arial', 15, 'bold'))
        predictImg.grid(row=0, column=2, padx=10, pady=10)
        
        quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="#363e75",
                               width=10, height=2, activebackground="#118ce1", font=('Arial', 15, 'bold'))
        quitWindow.pack(pady=30)

        self.window.mainloop()

    def collectUserImageForRegistration(self):
        name = self.studentNameTxt.get()
        ap = argparse.ArgumentParser()

        ap.add_argument("--faces", default=50, help="Number of faces that camera will get")
        ap.add_argument("--output", default=f"./datasets/train/{name}", help="Path to faces output")

        args = vars(ap.parse_args())

        trnngDataCollctrObj = TrainingDataCollector(args)
        trnngDataCollctrObj.collectImagesFromCamera()

        notifctn = f"We have collected {args['faces']} images for training."
        self.message.configure(text=notifctn)

    def getFaceEmbedding(self):
        ap = argparse.ArgumentParser()

        ap.add_argument("--dataset", default="datasets/train", help="Path to training dataset")
        ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings.pickle")
        ap.add_argument('--image-size', default='112,112', help='')
        ap.add_argument('--ga-model', default='', help='path to load model.')
        ap.add_argument('--gpu', default=0, type=int, help='gpu id')
        ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from beginning')
        ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

        args = ap.parse_args()

        notifctn = "Generating face embeddings for training images."
        self.message.configure(text=notifctn)

        genFaceEmbdng = GenerateFaceEmbeddings(args)
        genFaceEmbdng.getfaceEmbedding()

    def trainModel(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--embeddings", default="embeddings.pickle", help="path to serialized db of facial embeddings")
        ap.add_argument("--model", default="softmax_model.h5", help="path to output trained model")
        ap.add_argument("--le", default="label_encoder.pickle", help="path to output label encoder")

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
