# -*- coding: utf-8 -*-
"""
>> Runs OpenPose on different resolution reductions 
>> Calculates computational cost
         
"""

import os
import time
import pandas as pd
import argparse
import winsound
import openpose as op
# Create an argument parser
parser = argparse.ArgumentParser(description='Run AlphaPose on a set of videos with different resolution reductions.')
parser.add_argument('--option', type=int, choices=[0, 50, 75], default=0, help='Option value to use (0, 50, or 75). Default: 0')
args = parser.parse_args()

# Set the input and output directories based on the option value
if args.option == 0:
    input_dir = r'D:\Vivi\Documents\NTU\RGB\mini'
    output_dir = r'C:\Users\Vivi\Documents\Thesis\openpose\output\mini'
if args.option == 50:
    input_dir = r'D:\Vivi\Documents\NTU\RGB\mini_540p'
    output_dir = r'C:\Users\Vivi\Documents\Thesis\openpose\output\mini_50'
if args.option == 75:
    input_dir = r'D:\Vivi\Documents\NTU\RGB\mini_270p'
    output_dir = r'C:\Users\Vivi\Documents\Thesis\openpose\output\mini_75'

# Set the path to the OpenPose executable
openpose_dir = r'C:\Users\Vivi\Documents\Thesis\openpose'

# Create a list to store the results
results = []

# Iterate over all files in the input directory
total_cost = 0
for filename in os.listdir(input_dir):
    # Construct the full file paths
    input_file = os.path.join(input_dir, filename)
    output_vid = os.path.join(output_dir, 'video', os.path.splitext(filename)[0] + '.avi')
    output_json = os.path.join(output_dir, 'json', os.path.splitext(filename)[0])

    # Run OpenPose on the input video
    start_time = time.time()
    os.system(f'cd {openpose_dir} && bin\OpenPoseDemo.exe --video "{input_file}" --write_video "{output_vid}" --write_json "{output_json}" --display 0 > nul 2>&1')
    end_time = time.time()
    
    # Calculate the computational cost and add it to the total cost
    cost = end_time - start_time
    total_cost += cost
    
    print(f'Processed {filename} in {cost:.2f} seconds.')
    
    # Add the result to the results list
    results.append((filename, cost))

print(f'Total computational cost: {total_cost:.2f} seconds.')

# Create a dataframe from the results list
df = pd.DataFrame(results, columns=['Filename', 'Cost'])

# Add a row for the total cost
df.loc['Total'] = ['-', total_cost]

# Save the dataframe as a tab-separated text file
df.to_csv(os.path.join(output_dir, 'results.txt'), sep='\t', index=False)

# Play a sound notification when the script is done
winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
