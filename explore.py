# -*- coding: utf-8 -*-
"""
>> Print dataset information
>> Plot extracted keypoints (1 frame) 
>> Saves individual frames of a video

"""

import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def video_info(folder_path): 
    total_duration = 0
    total_frames = 0
    num_videos = 0
    min_duration = float('inf')
    max_duration = 0
    max_num_frames = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.avi'):
            file_path = os.path.join(folder_path, filename)
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = num_frames / fps
            total_duration += duration
            total_frames += num_frames
            num_videos += 1
            cap.release()
    
            if duration < min_duration:
                min_duration = duration
                min_fps = fps
                min_filename = filename
            if duration > max_duration:
                max_duration = duration
                max_fps = fps
                max_filename = filename
            if num_frames > max_num_frames:  # update max_num_frames if needed
                max_num_frames = num_frames
    
    if num_videos > 0:
        avg_duration = total_duration / num_videos
        avg_fps = total_frames / total_duration
        avg_num_frames = total_frames / num_videos
        print(f'Average video duration: {avg_duration:.2f} seconds')
        print(f'Average video frame rate: {avg_fps:.2f} fps')
        print(f'Average number of frames: {avg_num_frames:.2f} frames')
        print(f'Shortest video: {min_filename} ({min_duration:.2f} seconds, {min_fps:.2f} fps)')
        print(f'Longest video: {max_filename} ({max_duration:.2f} seconds, {max_fps:.2f} fps)')
        print(f'Maximum number of frames: {max_num_frames}')
    else:
        print('No .avi videos found in folder.')
    
def count_substring(folder_path, substrings):
    files = [file_path for file_path in Path(folder_path).glob('**/*') if any(substring in file_path.name for substring in substrings)]
    return len(files)

def check_reso(file_path):
    # open the video file
    cap = cv2.VideoCapture(file_path)
    
    # get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video resolution: {width}x{height}")

check_reso('D:/Vivi/Documents/NTU/RGB/mini_270p/S001C001P001R001A001_rgb.avi')
    
def scatter(skel):
    if skel == 1:
        print('Scattering OpenPose keypoints')
        filename = 'C:/Users/Vivi/Documents/Thesis/openpose/output/mini_0/json/S001C001P001R001A001_rgb/S001C001P001R001A001_rgb_000000000000_keypoints.json'
        with open(filename, "r") as f:
            data = json.load(f)
        keypoints_0 = data['people'][0]['pose_keypoints_2d']
        coordinates = [[keypoints_0[i], keypoints_0[i+1]] for i in range(0, len(keypoints_0), 3)]
        
    elif skel == 2:
        print('Scattering AlphaPose keypoints')
        filename = 'C:/Users/Vivi/Documents/Thesis/alphapose/output/mini_0/json/S001C001P001R001A001_rgb.json'
        with open(filename, "r") as f:
            data = json.load(f)
            coordinates = [data[0]['keypoints'][j:j+2] for j in range(0, len(data[0]['keypoints']), 3)] 

    else: #grount truth
        print('Scattering ground truth keypoints')
        filename = 'D:/Vivi/Documents/NTU/Skeleton/all_npy/S001C001P001R001A001_rgb.npy'
        data = np.load(filename, allow_pickle=True).item()
        keypoints = data['rgb_body0'] 
        coordinates = [(int(round(x)), int(round(y))) for x, y in keypoints[0]]
        
    # separate x and y coordinates and filter out (0, 0) coordinates
    filtered_coordinates = [coord for coord in coordinates if coord != [0, 0]]
    x_coordinates = [coord[0] for coord in filtered_coordinates]
    y_coordinates = [coord[1] for coord in filtered_coordinates]
       
    # print the number of (0, 0) coordinates
    num_filtered = len(coordinates) - len(filtered_coordinates)
    if num_filtered > 0:
        print(f"{num_filtered} coordinates with values [0, 0] have been filtered out.")

    fig, ax = plt.subplots()
    ax.scatter(x_coordinates, y_coordinates)
    
    # flip the y-axis
    ax.invert_yaxis()
    
    # invert the x-axis
    #ax.invert_xaxis()
    
    # label each point with its respective number
    for i, coord in enumerate(coordinates):
        ax.annotate(str(i), coord)
    
    plt.show()

def extract_frames():
    path_save = 'C:/Users/Vivi/Documents/Thesis/Thesis scripts/Images/AlphaPose/A46/'
    video = cv2.VideoCapture('C:/Users/Vivi/Documents/Thesis/alphapose/output/med_5/video/AlphaPose_S010C002P019R002A046_rgb.avi')

    # Loop through the video frames
    frame_count = 0
    while True:
        # Read the next frame from the video
        ret, frame = video.read()
    
        # If the frame was not read successfully, break the loop
        if not ret:
            break
    
        # Save the frame as an image file
        frame_count += 1
        filename = os.path.join(path_save, f'frame_{frame_count}.jpg')
        cv2.imwrite(filename, frame)
    
    # Release the video file and close all windows
    video.release()
    cv2.destroyAllWindows()
    print('done')
######################################################

folder_path = 'C:/Users/Vivi/Documents/Thesis/alphapose/output/med_5/tr'
# label_folder = 'D:/Vivi/Documents/NTU/Skeleton/med_mini_npy'
sub = ['A048']
#print(count_substring(folder_path, sub))
#video_info(folder_path)
#extract_frames()
#scatter(3)
