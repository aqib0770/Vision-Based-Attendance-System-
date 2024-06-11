import cv2 as cv

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