# -*- coding: utf-8 -*-
"""
>> Processes OpenPose/AlphaPose output
>> Calculates and writes MED and PMK to .csv

@author: Vivi
"""

import json
import numpy as np
from scipy.spatial.distance import euclidean
import math
import os
import csv
import winsound
import re
from sklearn.metrics import average_precision_score

def map_open(keypoints):
    """
    Maps the comparible keypoints from OpenPose to the order of the ground truth keypoints

    """
    mapping = {
    0: 3,   # nose -> head
    1: 2,   # neck 
    2: 4,   # Rshoulder
    3: 9,   # Relbow
    4: 6,   # Rwrist
    5: 4,   # Lshoulder 
    6: 5,   # Lelbow
    7: 6,   # Lwrist
    8: 0,   # midhip -> spinebase
    9: 16,  # Rhip
    10: 17, # Rknee
    11: 18, # Rankle
    12: 12, # Lhip
    13: 13, # Lknee
    14: 14, # Lankle 
    19: 15, # Lbigtoe -> Lfoot
    22: 19  # Rbigtoe -> Rfoot
    }
    
    new_array = np.zeros((17, 2))
    for i, idx in enumerate(mapping.values()):
        new_array[i] = keypoints[idx]
    return new_array

def read_open(folder, mini_folder, multi=False):
    """
    Reads OpenPose keypoints from all json files in a folder and returns them as a scaled list of keypoint arrays.
    """
    keypoints_all = []
    key2d = []
    
    if mini_folder == '0':
        scale_factor = 1
    elif mini_folder == '50':
        scale_factor = 2
    elif mini_folder == '75':
        scale_factor = 4
    else:
        print('Scale error')
        
    with os.scandir(folder) as entries:
        for entry in entries:
            filename = os.path.join(folder, entry.name)
            with open(filename, 'r') as file:
                data = json.load(file)
                if multi:
                    keypoints_1 = data['people'][0]['pose_keypoints_2d']
                    coordinates = [[keypoints_1[i], keypoints_1[i+1]] for i in range(0, len(keypoints_1), 3)] # Take only coordinates, leave out confidence score
                    key2d.append(map_open(coordinates)) # Keep and order relevant keypoint coordinates 
                    
                else:
                    keypoints_0 = data['people'][0]['pose_keypoints_2d']
                    coordinates = [[keypoints_0[i], keypoints_0[i+1]] for i in range(0, len(keypoints_0), 3)] # Take only coordinates, leave out confidence score
                    key2d.append(map_open(coordinates)) # Keep and order relevant keypoint coordinates 
             
    keypoints_all = np.array(key2d) * scale_factor

    return keypoints_all

def map_alpha(keypoints):
    """
    Maps the comparible keypoints from AlphaPose to the order of the ground truth keypoints

    """
    mapping = {
    0: 3,   # nose -> head
    5: 4,   # Lshoulder 
    6: 8,   # Rshoulder
    7: 5,   # Lelbow
    8: 9,   # Relbow
    9: 6,   # Lwrist 
    10: 10, # Rwrist
    11: 12, # Lhip
    12: 16, # Rhip 
    13: 13, # Lknee
    14: 17, # Rknee
    15: 14, # Lankle
    16: 18, # Rankle
    18: 2,  # neck
    19: 0,  # hip -> spinebase
    20: 15, # Lbigtoe -> Lfoot
    21: 19  # Rbigtoe -> Rfoot
    }
    
    new_array = np.zeros((17, 2))
    for i, idx in enumerate(mapping.values()):
        new_array[i] = keypoints[idx]
    return new_array
     
def read_alpha(file, mini_folder, multi=False):
    """
    Reads AlphaPose keypoints from a json file, scales them and returns them as a list of keypoint arrays.
    """
    with open(file, "r") as f:
        data = json.load(f)

    keypoints_all = []
    if mini_folder == '0':
        scale_factor = 1
    elif mini_folder == '50':
        scale_factor = 2
    elif mini_folder == '75':
        scale_factor = 4
    else:
        print('Scale error')
    
    if multi:
        key2d = []
        for i in range(len(data)-1):
            if data[i]['image_id'] == data[i+1]['image_id']: # Find second person info
                if data[i]['score'] >= 3 and data[i+1]['score'] >= 3:
                    coordinates = [data[i+1]['keypoints'][j:j+2] for j in range(0, len(data[i+1]['keypoints']), 3)] # Take only coordinates, leave out confidence score
                    key2d.append(map_alpha(coordinates)) # Keep and order relevant keypoint coordinates 
                else:
                    continue
            else:
                continue
        keypoints_all = np.array(key2d) * scale_factor
        # key2d = []
        # unique_ids = list(set([d['image_id'] for d in data]))
        # unique_ids.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        # first_image_id = unique_ids[0]
        # print(first_image_id)
        # for d in data:
        #     if d['image_id'] == first_image_id:
        #         coordinates = [d['keypoints'][j:j+2] for j in range(0, len(d['keypoints']), 3)] # Take only coordinates, leave out confidence score
        #         key2d.append(map_alpha(coordinates)) # Keep and order relevant keypoint coordinates 
        # keypoints_all = np.array(key2d) * scale_factor
        
    else:
        key2d = []
        for i in range(len(data)):
            if data[i]['score'] >= 3: # Catch strange additions that are not people but has json data
                coordinates = [data[i]['keypoints'][j:j+2] for j in range(0, len(data[i]['keypoints']), 3)] # Take only coordinates, leave out confidence score
                key2d.append(map_alpha(coordinates)) # Keep and order relevant keypoint coordinates 
            else:
                continue
            
        keypoints_all = np.array(key2d) * scale_factor

    return keypoints_all

def read_label_npy(filename):
    """
    Reads ground truth keypoints from a numpy .npy file and returns relevant keypoints as a numpy array.
    """

    data = np.load(filename, allow_pickle=True).item()

    keypoints = data['rgb_body0'][:, :, :2] # extract the keypoint information from the 'rgb_body0' array
    keypoints = keypoints.reshape(-1, data['njoints'], 2)  
    
    # Remove non matching joint indices (spinemid, Lhand, Rhand, spineshoulder, Lhandtip, Lthumb, Rhandtip, Rthumb)
    keypoints = np.delete(keypoints, [1, 7, 11, 20, 21, 22, 23, 24], axis=1)
    
    return keypoints

def align_pred(predicted, ground_truth):
    """
    Procrustes superimposition: translation + rotation + (scaling)
    """


    num_frames = predicted.shape[0]
    aligned_predicted = np.zeros_like(predicted)

    
    for i in range(num_frames):
        pred_keypoints = predicted[i]
        gt_keypoints = ground_truth[i]
        #print(type(gt_keypoints))

        # Compute centroids 
        if pred_keypoints.size > 0:
            pred_centroid = np.mean(pred_keypoints, axis=0)
        else:
            pred_centroid = np.zeros_like(pred_keypoints[0])
        if gt_keypoints.size > 0:
            gt_centroid = np.mean(gt_keypoints, axis=0)
        else:
            gt_centroid = np.zeros_like(gt_keypoints[0])

        # Translation
        translated_pred = pred_keypoints - pred_centroid
        translated_gt = gt_keypoints - gt_centroid


        distances = np.sqrt(np.sum((translated_pred - translated_gt)**2, axis=1))
        mean_distance = np.mean(distances)

        # Scaling
        scaled_pred = translated_pred / mean_distance
        scaled_gt = translated_gt / mean_distance

        # Rotation
        cov = np.cov(scaled_gt.T @ scaled_pred)
        u, s, vh = np.linalg.svd(cov)
        rotation = u @ vh

        aligned_pred = scaled_pred @ rotation

        # Final position
        aligned_pred += gt_centroid

        aligned_predicted[i] = aligned_pred
    
    #print(aligned_predicted[0][0])
    return aligned_pred

def mean_euclidean_distance(predictions, ground_truth):
    """
    Calculates the mean Euclidean distance between predicted keypoints and ground truth keypoints.
    """
    # Calculate mean euclidean distance between predictions and ground truth
    # print('tru' , ground_truth[1])
    # print('pred', predictions)
    
    distances = []
    for keypoints_prediction in predictions:
        distance_sum = 0.0
        num_keypoints = 0
        for i, kp in enumerate(ground_truth[0]):
            if np.all(keypoints_prediction[:2] != 0):
                distance_sum += math.dist(kp[:2].tolist(), keypoints_prediction[:2].tolist())
                #distance_sum += np.linalg.norm(kp[:2] - keypoints_prediction[:2])
                #print('tru', kp[:2].tolist())
                #print('pred', keypoints_prediction[:2].tolist())
                #print(distance_sum)
                num_keypoints += 1
        if num_keypoints > 0:
            distances.append(distance_sum / num_keypoints)
    mean_distance = np.mean(distances)

    return mean_distance

def percentage_missing_keypoints(keypoints):
    """
    Calculates the percentage of missing keypoints in a set of keypoints.
    """
    #if not isinstance(keypoints, np.ndarray):
    keypoints = np.array(keypoints)
    
    if keypoints.ndim != 3 or keypoints.shape[-1] != 2:
        raise ValueError(f"keypoints must be a numpy array with shape (num_frames, num_keypoints, 2), got {keypoints.shape}")
  
    num_missing = np.sum(np.all(keypoints == [0,0], axis=2))
    total_keypoints = keypoints.shape[1]*keypoints.shape[0]
    
    return (num_missing / total_keypoints) * 100

def openpose_calc(openpose_folder, label_folder, mini_folder):
    """
    Wrapper function for OpenPose.
    """
    # Read all OpenPose keypoints
    all_alligned =[]
    all_distances = []
    all_missing = []
    all_precision = []
    for filename in os.listdir(openpose_folder):
        # Read OpenPose keypoints from current file
        openpose_file = os.path.join(openpose_folder, filename)
        # Check file case
        for i in range(50, 61):
            if f'A0{i}' in filename:
                keypoints_open = read_open(openpose_file, mini_folder, True)
            else:
                keypoints_open = read_open(openpose_file, mini_folder, False)
    
        # Get corresponding ground truth file
        label_filename = filename[:-4] + '_rgb.npy'
        label_file = os.path.join(label_folder, label_filename)
    
        # Calculate mean Euclidean distance between OpenPose keypoints and ground truth keypoints
        alligned = align_pred(keypoints_open, read_label_npy(label_file))
        all_alligned.append(alligned)
        mean_dist = mean_euclidean_distance(alligned, read_label_npy(label_file))
        all_distances.append(mean_dist)
        # mAP = mean_average_precision(all_alligned, read_label_npy(label_file))
        # all_precision.append(mAP)
        
        #print(f'{filename}')
        # Calculate percentage of missing keypoints in OpenPose keypoints
        percent_missing = percentage_missing_keypoints(keypoints_open)
        #print("Percentage of missing keypoints OpenPose: {:.2f}%".format(percent_missing))
        all_missing.append(percent_missing)
        
    print(all_precision)
    
    # Calculate and return mean Euclidean distance and percentage of missing keypoints
    return all_distances, all_missing

def alphapose_calc(alphapose_folder, label_folder, mini_folder):
    """
    Wrapper function for AlphaPose.
    """
    # Read all AlphaPose keypoints
    all_distances = []
    all_missing = []
    for filename in os.listdir(alphapose_folder):
        # Read AlphaPose keypoints from current file
        alphapose_file = os.path.join(alphapose_folder, filename)
        # Check file case
        for i in range(50, 62):
            if f'A0{i}' in filename:
                keypoints_alpha = read_alpha(alphapose_file, mini_folder, True)
            else:
                keypoints_alpha = read_alpha(alphapose_file, mini_folder, False)
        
        # Get corresponding ground truth file
        label_filename = filename[:-9] + '_rgb.npy'
        label_file = os.path.join(label_folder, label_filename)
        
        # Calculate mean Euclidean distance between OpenPose keypoints and ground truth keypoints
        print(f'{filename}')
        alligned = align_pred(keypoints_alpha, read_label_npy(label_file))
        mean_dist = mean_euclidean_distance(alligned, read_label_npy(label_file))
        all_distances.append(mean_dist)
        
        #print(f'{filename}')
        # Calculate percentage of missing keypoints in OpenPose keypoints
        percent_missing = percentage_missing_keypoints(keypoints_alpha)
        #print("Percentage of missing keypoints AlphaPose: {:.2f}%".format(percent_missing))
        all_missing.append(percent_missing)

    # Calculate and return mean Euclidean distance and percentage of missing keypoints
    return all_distances, all_missing
    
def write_to_csv(all_distances_openpose, all_missing_openpose, all_distances_alphapose, all_missing_alphapose, mini_folder, writer):
    """
    Writes all OpenPose and AlphaPose results and their mean to a CSV file. 
    """
    # Calculate mean distance and add it to the beginning of the all_distances list
    mean_distance_openpose = np.mean(all_distances_openpose)
    all_distances_openpose.insert(0, mean_distance_openpose)
    
    mean_distance_alphapose = np.mean(all_distances_alphapose)
    all_distances_alphapose.insert(0, mean_distance_alphapose)

    # Calculate mean missing and add it to the beginning of the all_distances list
    mean_missing_openpose = np.mean(all_missing_openpose)
    all_missing_openpose.insert(0, mean_missing_openpose)
    
    mean_missing_alphapose = np.mean(all_missing_alphapose)
    all_missing_alphapose.insert(0, mean_missing_alphapose)

    # Transpose the lists to convert it to a column
    all_distances_openpose_column = [[d] for d in all_distances_openpose]
    percent_missing_openpose_column = [[p] for p in all_missing_openpose]
    all_distances_alphapose_column = [[d] for d in all_distances_alphapose]
    percent_missing_alphapose_column = [[p] for p in all_missing_alphapose]

    # Write headers
    writer.writerow(["Open MED" + mini_folder, "Open PMK " + mini_folder, "Alpha MED " + mini_folder, "Alpha PMK " + mini_folder])
    
    # Write data
    writer.writerows(zip(all_distances_openpose_column, percent_missing_openpose_column, all_distances_alphapose_column, percent_missing_alphapose_column))

def wrapper():
    """
    Main wrapper function
    """
    label_folder = 'D:/Vivi/Documents/NTU/Skeleton/all_npy/'
    
    # Loop over mini folders
    for mini_folder in ["0", "50", "75"]:
        filename = "values_" + mini_folder + ".csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            openpose_folder = '/Users/Vivi/Documents/Thesis/openpose/output/mini_{}/json/'.format(mini_folder)
            alphapose_folder = '/Users/Vivi/Documents/Thesis/alphapose/output/mini_{}/json/'.format(mini_folder)
            
            all_distances_o, all_missing_o = openpose_calc(openpose_folder, label_folder, mini_folder)
            print('Mean Euclidean Distance OpenPose:', np.mean(all_distances_o), 'Percentage Missing Keypoints:', np.mean(all_missing_o))
            
            print(mini_folder)
            all_distances_a, all_missing_a = alphapose_calc(alphapose_folder, label_folder, mini_folder)
            print('Mean Euclidean Distance AlphaPose:', np.mean(all_distances_a), 'Percentage Missing Keypoints:', np.mean(all_missing_a))
    
            # Write data to CSV
            #write_to_csv(all_distances_o, all_missing_o, all_distances_a, all_missing_a, mini_folder, writer)
    
        print("Results written to", filename)
        
    
## =========================================================================
wrapper()
# label_folder = 'D:/Vivi/Documents/NTU/Skeleton/med_mini_npy'
# data = np.load(label_folder, allow_pickle=True).item()
# print(data)

# Play a sound notification when the script is done
winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
