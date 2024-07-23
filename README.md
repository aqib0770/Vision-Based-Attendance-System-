# Vision-Based Attendance System

This project is a solution designed to automate the process of attendance using facial recognition technology. It is developed to be used in educational instutes, where the attendance of students is taken manually by the teacher. The system will capture the image of the student and compare it with the images stored in the database. If the student is recognized, the attendance will be marked for that student. The system will also generate a report of the attendance of the students.
The project utilizes MTCNN for face detection and DeepFace for face recognition and to generate face embeddings. The system is developed using Python and Tkinter.

## Features
- Capture the image of the student
- Compare the image with the images stored in the database
- Mark the attendance of the student
- Generate a report of the attendance of the students

## Requirements
- Python 3.6 or higher
- OpenCV
- MTCNN
- DeepFace
- Tkinter
- Pandas
- Numpy

## Installation
1. Clone the repository
```bash
git clone  https://github.com/aqib0770/Vision-Based-Attendance-System-.git
```
2. Install the required libraries
```bash
pip install -r requirements.txt
```
3. Run the application
```bash
python app.py
```

## Usage
1. Enter the student's name in the text field
2. Add the images of the students in the database by clicking `Take Images`
button
3. Train the model on the images by clicking `Train Model` button
4. Now, click on `Start Attendance` button to start the attendance
