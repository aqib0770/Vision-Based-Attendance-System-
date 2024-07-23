import cv2 as cv
import os
import pandas as pd
from datetime import datetime


def create_output_dir():
    # Create a directory with a unique name based on the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join('./output', timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def rescaleFrame(frame, scale=0.75):

    # Works for Images, Video and Live Video

    width = int(frame.shape[1] * scale)  #frame.shape[1] represent width
    height = int(frame.shape[0] * scale)  #frame.shape[2] represent height
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def findCosineDistance(vector1, vector2):
        """
        Calculate cosine distance between two vector
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

import numpy as np

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist / len(source_vecs)

def findCosineDistance(test_vec, source_vec):
    # Normalize the vectors
    test_vec_norm = np.linalg.norm(test_vec)
    source_vec_norm = np.linalg.norm(source_vec)

    # Compute the cosine similarity
    cos_similarity = np.dot(test_vec, source_vec) / (test_vec_norm * source_vec_norm)

    # Convert the cosine similarity to a cosine distance
    cos_distance = 1.0 - cos_similarity

    return cos_distance

def create_attendence_df():
    if os.path.exists('attendance.csv'):
        try:
            df = pd.read_csv('attendance.csv')
            if df.empty or not set(['Name', 'Date', 'Time', 'Date_Time']).issubset(df.columns):
                raise ValueError("File exists but is empty or has missing columns")
        except (pd.errors.EmptyDataError, ValueError):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Date_Time'])
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Date_Time'])
    return df

def mark_attendance(name):
    attended = set()
    current_time = datetime.now()
    date = current_time.strftime('%Y-%m-%d')
    time = current_time.strftime('%H:%M:%S')
    if name not in attended and name != 'Unknown':
        attended.add(name)
        df = create_attendence_df()
        new_row = ({'Name': name, 'Date': date, 'Time': time, 'Date_Time': current_time})
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = df.sort_values('Date_Time').groupby('Name').first().reset_index()
        df = df.drop('Date_Time', axis=1)
        df.to_csv('attendance.csv', index=False)
        
        return True