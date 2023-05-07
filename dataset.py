# -*- coding: utf-8 -*-
"""
>> Creates subset dataset of the whole NTU RGB+D dataset containing actions:
    
>> Resizes the dataset to such that videos are at 100%, 50% and 25% resolution
    (e.g. 1080p, 540p, 270p)

"""
import os
import shutil
import cv2
import winsound

def shuffle_and_resize(shuffle=True, resize=True):
    # Set the input and output directories for file copying
    destdir = r'D:\Vivi\Documents\NTU\RGB\mini'
    datadir = r'D:\Vivi\Documents\NTU\RGB\nturgb+d_rgb'
    
    # Define the substrings to match in file names
    substrings = [f"A{i:03d}" for i in range(41, 50)]
    #substrings = [f'S001C00{i}P001R00{j}A{k:03d}' for i in range(1, 4) for j in range(1, 3) for k in range(1, 61)]
    
    # Set the input and output directories for video resizing
    input_dir = r'D:\Vivi\Documents\NTU\RGB\mini'
    output_dir_270p = r'D:\Vivi\Documents\NTU\RGB\mini_270p'
    output_dir_540p = r'D:\Vivi\Documents\NTU\RGB\mini_540p'
    
    # Define the target resolutions
    target_resolution_270p = (480, 270)
    target_resolution_540p = (960, 540)
    
    # Read the contents of the text file into a set
    with open(r'D:\Vivi\Documents\NTU\Skeleton\missing_txt.txt', 'r') as f:
        existing_filenames = set(f.read().splitlines())
        
    if shuffle:
        # Iterate over all files in the data directory for file copying
        for fragmentname in os.listdir(datadir):
            if fragmentname not in existing_filenames:
                for substring in substrings:
                    if substring in fragmentname:
                        src_file = os.path.join(datadir, fragmentname)
                        dst_file = os.path.join(destdir, fragmentname)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, destdir)
                            print(f"Copied {src_file} to {dst_file}")
                        else:
                            print(f"File {dst_file} already exists, skipping")
    
    if resize:
        # Iterate over all files in the input directory for video resizing
        for filename in os.listdir(input_dir):
            output_file_270p = os.path.join(output_dir_270p, filename)
            output_file_540p = os.path.join(output_dir_540p, filename)
            
            if os.path.exists(output_file_270p) and os.path.exists(output_file_540p):
                print(f"File {filename} has already been resized and is present in the output directories.")
                continue
        
            else:
                # Construct the full file paths
                input_file = os.path.join(input_dir, filename)
                output_file_270p = os.path.join(output_dir_270p, filename)
                output_file_540p = os.path.join(output_dir_540p, filename)
                
                # Open the input video file
                cap = cv2.VideoCapture(input_file)
                
                # Get the current resolution
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Create a VideoWriter object for the 270p output video file
                fourcc = cv2.VideoWriter_fourcc(*'XVID') # Set the video codec
                out_270p = cv2.VideoWriter(output_file_270p, fourcc, 30, target_resolution_270p)
                
                # Create a VideoWriter object for the 540p output video file
                out_540p = cv2.VideoWriter(output_file_540p, fourcc, 30, target_resolution_540p)
                
                # Iterate over all frames in the input video
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize the frame to the 270p target resolution
                    resized_frame_270p = cv2.resize(frame, target_resolution_270p)
                    
                    # Resize the frame to the 540p target resolution
                    resized_frame_540p = cv2.resize(frame, target_resolution_540p)
                    
                    # Write the resized frames to the output video files
                    out_270p.write(resized_frame_270p)
                    out_540p.write(resized_frame_540p)
                
                # Release the VideoCapture and VideoWriter objects for both resolutions
                cap.release()
                out_270p.release()
                out_540p.release()
                
                print(f"File {filename} processed successfully.")
                  
shuffle_and_resize(False, False)

# Play a sound notification when the script is done
winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
