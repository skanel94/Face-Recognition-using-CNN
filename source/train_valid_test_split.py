# execution command: python train_valid_test_split.py

import os
import shutil

def makedir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)

# Paths of train, validation, test directories
train_path = "train/"
valid_path = "valid/"
test_path = "test/"

# Create train, validation, test directories 
makedir(train_path)
makedir(valid_path)
makedir(test_path)

src_path="face_dataset/"

fs=os.listdir(src_path)

sum_len = 0
for d in fs:
    sum_len += len(os.listdir(src_path+d))

print('Initial dataset with Size {} consists of {} people\n'.format( sum_len, len(fs)))

# Split dataset into train and test set
train_frac = 0.70    # Fraction train set
valid_frac = 0.15    # Fraction validation set
test_frac  = 0.15    # Fraction test set

for directory in fs:
    label = directory.split(" ")[1]

    # Check if the train-validation-test fractions are right
    assert (test_frac + valid_frac + train_frac == 1.0), 'Test + Valid + Train Fractions must sum to 1.0'
    counter = 0 # Counter for files into directories
    
    # For each file into every single directory      
    for file in os.listdir(src_path+directory):
        
        if counter < ( len(os.listdir(src_path+directory)) * train_frac ):
            # Create a new folder with the name of this person (if it doesn't already exist)
            new_folder_name = "{}".format(label)
            makedir(train_path + new_folder_name)
            
            # Copy the file into the training folder
            src=os.path.join(src_path, directory, file)
            dst=os.path.join(train_path, new_folder_name)
            shutil.copy(src,dst)
            
            counter+=1
            
        elif counter < ( len(os.listdir(src_path+directory)) * (train_frac + valid_frac) ):
            # Create a new folder with the name of this person (if it doesn't already exist)
            new_folder_name = "{}".format(label)
            makedir(valid_path + new_folder_name)
            
            # Copy the file into the validation folder
            src=os.path.join(src_path, directory, file)
            dst=os.path.join(valid_path, new_folder_name)
            shutil.copy(src,dst)
            
            counter+=1
            
        else:
            # Create a new folder with the name of this person (if it doesn't already exist)
            new_folder_name = "{}".format(label)
            makedir(test_path + new_folder_name)
            
            # Copy the file into the validation folder
            src=os.path.join(src_path, directory, file)
            dst=os.path.join(test_path, new_folder_name)
            shutil.copy(src,dst)
            
            counter+=1
            
print('Dataset split into:')
sum_len = 0
for d in os.listdir(train_path):
    sum_len += len(os.listdir(train_path+d))

print('Train Set with Size: ', sum_len)

sum_len = 0
for d in os.listdir(valid_path):
    sum_len += len(os.listdir(valid_path+d))
    
print('Validation Set with Size: ', sum_len)

sum_len = 0
for d in os.listdir(test_path):
    sum_len += len(os.listdir(test_path+d))
    
print('Test Set with Size: ', sum_len)
print('\n[Dataset split was {}% for training - {}% for validating - {}% for testing]\n'.format(train_frac , valid_frac, test_frac))
  
