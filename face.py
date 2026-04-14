'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    
    # Convert img to numpy array with (H, W, C) dimensions for face_recognition input
    img_face = img.permute(1, 2, 0).numpy()
    # Find face locations
    face_locations = face_recognition.face_locations(img_face)
    # Convert back to torch tensors
    torch.tensor(face_locations)
    # Extract (topleft_y, bottomright_x, bottomright_y, topleft_x) format 
    for face in face_locations:
        topleft_y, bottomright_x, bottomright_y, topleft_x = face
        # Convert to [topleft_x, topleft_y, box_width, box_height] format and float
        topleft_x = float(topleft_x)
        topleft_y = float(topleft_y)
        box_width = float(bottomright_x - topleft_x)
        box_height = float(bottomright_y - topleft_y)
        # Append bounding boxes to detection_results
        detection_results.append([topleft_x, topleft_y, box_width, box_height])

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    
    ### Extract a 128-dimensional vector for each cropped face ###
    
    # Initialized faces list
    faces = []
    # Iterate through each img in imgs
    img_names = list(imgs.keys())
    for name in img_names:
        img_array = imgs[name]
        # Convert img to numpy array with (H, W, C) dimensions for face_recognition input
        img = img_array.permute(1, 2, 0).numpy()
        # Find face locations
        boxes = face_recognition.face_locations(img)
        # Extract face encoding (128-dimensional vector)
        face = face_recognition.face_encodings(img, boxes)
        # Append face to faces list as torch tensor
        faces.append(torch.tensor(face))
    
    ### Build K-means clustering algorithm ###
    
    # Stack list face tensors into a 2D array
    F = torch.stack(faces)
    # Randomly select K points as starting cluster centroids
    N = F.shape[0]
    indices = torch.randperm(N)[:K]
    centroids = F[indices]
    # Initialize closest distance labels
    labels = torch.zeros(N)
    # Start K-means iterations
    for _ in range(100):
        # Compute distances between centroids and surrounding points
        distances = torch.norm(F.unsqueeze(1) - centroids.unsqueeze(0), dim=2)
        # Assign each point to the closest centroid
        new_labels = torch.argmin(distances, dim=1)
        # Check if labels changed this iteration
        if torch.equal(labels, new_labels):
            break
        # If labels changed, then assign new labels as labels
        labels = new_labels
        print(labels)
    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)
