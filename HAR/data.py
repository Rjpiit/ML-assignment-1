import pandas as pd
import numpy as np
import os

# List of file paths to be processed (use relative paths)
file_paths = [
    "./Combined/Train/LAYING/Subject_1.csv",
    "./Combined/Train/SITTING/Subject_1.csv",
    "./Combined/Train/STANDING/Subject_1.csv",
    "./Combined/Train/WALKING/Subject_1.csv",
    "./Combined/Train/WALKING_DOWNSTAIRS/Subject_1.csv",
    "./Combined/Train/WALKING_UPSTAIRS/Subject_1.csv"
]

# Desired output directory (use a relative path)
output_dir = "./Linear_acc"
os.makedirs(output_dir, exist_ok=True)

# Loop through each file, process it, and save the output
for file_path in file_paths:
    print(f"Processing file: {file_path}")
    if os.path.exists(file_path):
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Print the first few rows of the data to check for anomalies
        print(data.head())
        
        # Calculate the Total Acceleration
        data['Total_Acceleration'] = np.sqrt(data['accx']**2 + data['accy']**2 + data['accz']**2)
        
        # Print basic statistics of the calculated Total Acceleration
        print(data['Total_Acceleration'].describe())
        
        # Extract the activity name from the file path
        path_parts = file_path.replace('\\', '/').split('/')
        if len(path_parts) >= 4:
            activity_name = path_parts[-2]
        else:
            print(f"Unexpected file path format: {file_path}")
            continue
        
        # Create a directory for the activity if it doesn't exist
        activity_dir = os.path.join(output_dir, activity_name)
        os.makedirs(activity_dir, exist_ok=True)
        
        # Generate the output file path with a .csv extension in the activity folder
        output_file = os.path.join(activity_dir, os.path.basename(file_path))
        
        # Save the result to the specified output path
        data.to_csv(output_file, index=False)
        
        # Print confirmation message
        print(f"Processed {file_path} and saved to {output_file}")
    else:
        print(f"File not found: {file_path}")
