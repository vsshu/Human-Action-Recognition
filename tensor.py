# -*- coding: utf-8 -*-
"""
>> Preprocess: normalization, scaling, sliding window cleaning, keyframe selection
>> Create tensors + one hot labeling 

@author: Vivi
"""

import json
import numpy as np
from scipy.spatial.distance import euclidean
import os
import csv
import winsound
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*KMeans is known to have a memory leak on Windows with MKL.*")


def read_alpha(file):
    """
    Reads AlphaPose keypoints from a json file and returns them as a list of keypoint arrays.
    """
    with open(file, "r") as f:
        data = json.load(f)

    keypoints_all = []
    key2d = []
    
    for i in range(len(data)):
        keypoints = [data[i]['keypoints'][j:j+2] for j in range(0, len(data[i]['keypoints']), 3)]
        key2d.append(keypoints)
        
    keypoints_all = key2d
    keypoints_array = np.array(keypoints_all)
    
    return keypoints_array

def normalize(keypoints):
    """
    Normalizes keypoints to a mean of zero and standard deviation of one.
    """
    # Compute mean and standard deviation over all frames and keypoints
    mean = np.mean(keypoints, axis=(0, 1))
    std = np.std(keypoints, axis=(0, 1))
    
    # Normalize the keypoints
    keypoints_normalized = (keypoints - mean) / std
    return keypoints_normalized


def scale(keypoints_normalized):
    """
    Scales each frame by the distance between the nose and neck coordinates.
    """
    # Extract nose and neck coordinates for each frame
    nose_coords = keypoints_normalized[:, 0, :]
    neck_coords = keypoints_normalized[:, 1, :]

    # Compute distance between nose and neck coordinates for each frame
    dists = np.sqrt(np.sum(np.square(nose_coords - neck_coords), axis=1))

    # Check for zero distances and replace with a very small value
    zero_mask = (dists == 0)
    dists[zero_mask] = 1e-6
    
    # Scale all keypoints for each frame based on the nose-neck distance
    scaled_keypoints = keypoints_normalized / dists[:, None, None]
    return scaled_keypoints


def norm_and_scale(file):
    """
    Wrapper function for normalization and scaling.

    """
    # Read keypoints from JSON files
    keypoints = read_alpha(file)
    
    # Normalize the keypoints
    keypoints_normalized = normalize(keypoints)

    # Scale the normalized keypoints
    scaled_keypoints = scale(keypoints_normalized)

    return scaled_keypoints

def sliding_window(keypoints):
    """
    Sliding window approach for wrong estimation correction.

    """
    window_size = 9
    threshold = 3
    num_frames, num_keypoints, num_coordinates = keypoints.shape
    new_keypoints = keypoints.copy()
    
    for i in range(num_frames - window_size + 1):
        for j in range(num_keypoints):
            for k in range(num_coordinates):
                window = keypoints[i:i+window_size, j, k]
                if np.max(window) - np.min(window) > threshold:
                    median_value = np.median(window)
                    new_keypoints[i:i+window_size, j, k] = median_value

    return new_keypoints

def k_selection(keypoints, num_frames, videoname):
    """
    Uses K-means clustering to select key frames from a set of keypoints.

    """
    # Flatten keypoints array
    flattened_keypoints = keypoints.reshape(-1, 52)
    
    try:
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_frames, n_init=10, random_state=0).fit(flattened_keypoints)
        
        # Get cluster centers as selected key frames
        selected_frames = kmeans.cluster_centers_.reshape(num_frames, 26, 2)
        
        return selected_frames
    
    except Exception as e:
        print(e, videoname)
        

def tensor(folder):
    """
    Wrapper function to create matrix sequences

    """

    print('Kmeans keyframe selection..')
    seq_raw = np.empty(shape=(0, 26, 2))
    for filename in os.listdir(folder):
        file = os.path.join(folder, filename)
        fixed = sliding_window(norm_and_scale(file))
        #seq_per_vid = k_selection(fixed, 5, filename)
        seq_raw = np.append(seq_raw, fixed, 0)

    np.save('raw_k5_nokm.npy', seq_raw)
    print(seq_raw.shape)
    seq_raw = np.load('raw_k5.npy')
    print(seq_raw.shape)
    print(seq_raw[0])

    num_videos, num_keypoints, num_coordinates = seq_raw.shape
    num_videos = num_videos/3475
    timesteps = 3475
    features = num_keypoints * num_coordinates
    data = seq_raw.reshape(int(num_videos), int(timesteps), int(features))
    #np.save('tensors_k_ttnokm.npy', data)
    print(f'Tensor has been saved in shape {data.shape}')
        

def labels(folder):
    # Load data from numpy file
    data = np.load('tensors_sw_tt5w9.npy')
    print(data.shape)
    
    file_names = os.listdir(folder)
    print(len(file_names))
    num_videos = len(data)
    num_classes = 5
    labels = np.zeros((num_videos, num_classes))
    
    # Target strings for each class
    #target_strings = ['A041', 'A043', 'A045']
    target_strings = ['A043', 'A044', 'A045', 'A046', 'A048']
    #target_strings = ['A041', 'A042', 'A043', 'A044', 'A045', 'A046', 'A047', 'A048', 'A049']
    
    # one-hot encoded vectors
    for i, file_name in enumerate(file_names):
        for j, target_str in enumerate(target_strings):
            if target_str in file_name:
                labels[i, j] = 1
                break
    
    # Verify that the number of labels matches the number of video sequences in the data
    assert labels.shape[0] == data.shape[0]
    
    # Save the labels to a numpy file
    np.save('labels_tt5w9.npy', labels)
    print(f'Labels have been saved in shape {labels.shape}')


########################################################################################

folder = 'C:/Users/Vivi/Documents/Thesis/alphapose/output/med_5/tr/'

#tensor(folder)
#labels(folder)


# Play a sound notification when the script is done
winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)