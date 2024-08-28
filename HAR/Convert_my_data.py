

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load the original dataset
file_path = r"C:\Users\prave\OneDrive\Desktop\HAR\My_data\my_processed_data.csv"  # Replace with the correct path to your CSV file
data = pd.read_csv(file_path)

# Perform stratified split to maintain the distribution of activities
train_data, test_data = train_test_split(
    data, 
    test_size=0.3, 
    stratify=data['Activity'],  # Use 'Activity' column for stratification
    random_state=42  # For reproducibility
)

# Define output directory
output_dir = "./My_data"  
os.makedirs(output_dir, exist_ok=True)  

# Define file paths for saving the split data
train_file_path = os.path.join(output_dir, "my_processed_data_train.csv")
test_file_path = os.path.join(output_dir, "my_processed_data_test.csv")

# Save the split data into separate CSV files
train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print(f"Training data saved to: {train_file_path}")
print(f"Testing data saved to: {test_file_path}")
