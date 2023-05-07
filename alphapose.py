# -*- coding: utf-8 -*-
"""
>> Runs AlphaPose on different resolution reductions 
>> Calculates computational cost

>> Run in Anaconda Prompt
"""

import os
import time
#import pandas as pd
import argparse
import shutil
import winsound

# Create an argument parser
parser = argparse.ArgumentParser(description='Run AlphaPose on a set of videos with different resolution reductions.')
parser.add_argument('--option', type=int, choices=[0, 50, 75], default=0, help='Option value to use (0, 50, or 75). Default: 0')
args = parser.parse_args()

# Set the input and output directories based on the option value
if args.option == 0:
    print('Processing at resolution 1080p')
    input_dir = r'D:\Vivi\Documents\NTU\RGB\mini'
    output_dir = r'C:\Users\Vivi\Documents\Thesis\alphapose\output\mini'
if args.option == 50:
    print('Processing at resolution 540p')
    input_dir = r'D:\Vivi\Documents\NTU\RGB\mini_540p'
    output_dir = r'C:\Users\Vivi\Documents\Thesis\alphapose\output\mini_50'
if args.option == 75:
    print('Processing at resolution 270p')
    input_dir = r'D:\Vivi\Documents\NTU\RGB\mini_270p'
    output_dir = r'C:\Users\Vivi\Documents\Thesis\alphapose\output\mini_75'


# Set the path to the AlphaPose params
alphapose_dir = r'C:\Users\Vivi\Documents\Thesis\alphapose'
alphapose_cfg = 'configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
    #'configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml'
alphapose_checkpoint = 'pretrained_models/halpe26_fast_res50_256x192.pth'
    #'pretrained_models/multi_domain_fast50_regression_256x192.pth'

# Create a list to store the results
results = []
total_cost = 0

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    # Construct the full file paths
    input_file = os.path.join(input_dir, filename)
    output_file = os.path.join(output_dir, 'json', os.path.splitext(filename)[0])

    # Run AlphaPose on the input video
    start_time = time.time()
    os.system(f'cd {alphapose_dir} && python scripts/demo_inference.py --cfg  "{alphapose_cfg}" --checkpoint "{alphapose_checkpoint}" --video "{input_file}" --outdir "{output_file}" --save_video --vis_fast')
    end_time = time.time()
    
    # Calculate the computational cost and add it to the total cost
    cost = end_time - start_time
    total_cost += cost
    
    print(f'Processed {filename} in {cost:.2f} seconds.')
    
    # Add the result to the results list
    results.append((filename, cost))

print(f'Total computational cost: {total_cost:.2f} seconds.')

# Save the results to a dataframe
# df = pd.DataFrame(results, columns=['Filename', 'Cost'])
# df.loc['Total'] = ['-', total_cost]
# df.to_csv(os.path.join(output_dir, 'results.txt'), sep='\t', index=False)

#####################################################

# Set the input and output directories
# input_dir = r'C:\Users\Vivi\Documents\Thesis\alphapose\output\mini\json'
# output_dir = r'C:\Users\Vivi\Documents\Thesis\alphapose\output\mini'

# # Create the output directories if they don't exist
# os.makedirs(os.path.join(output_dir, 'video'), exist_ok=True)
# os.makedirs(os.path.join(output_dir, 'json'), exist_ok=True)

# # Iterate over all directories in the input directory
# for dir_name in os.listdir(input_dir):
#     # Construct the full directory path
#     dir_path = os.path.join(input_dir, dir_name)
#     if not os.path.isdir(dir_path):
#         continue

#     # Process all video files in the directory
#     for filename in os.listdir(dir_path):
#         if not filename.endswith('.avi'):
#             continue

#         # Construct the full file paths
#         input_file = os.path.join(dir_path, filename)
#         output_file = os.path.join(output_dir, 'video', filename.replace('AlphaPose_', ''))

#         # Copy the video file to the output directory
#         shutil.copy(input_file, output_file)

#     # Process all JSON files in the directory
#     for filename in os.listdir(dir_path):
#         if not filename.endswith('.json'):
#             continue

#         # Construct the full file paths
#         input_file = os.path.join(dir_path, filename)
#         output_file = os.path.join(output_dir, 'json', dir_name + '_' + filename)

#         # Copy the JSON file to the output directory and rename it
#         shutil.copy(input_file, output_file)

print('Done!')

# Play a sound notification when the script is done
#winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)