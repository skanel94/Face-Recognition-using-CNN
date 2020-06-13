# execution command: python downsample_dataset.py

import os
import numpy as np
import shutil

def makedir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
  
src_path="lfw_alig/"
dest_path="face_dataset/"

# Create new directory for the subset of initial dataset
makedir(dest_path)


fs=os.listdir(src_path)
print('Initial Dataset Size: ',len(fs))

num_classes  = 10
people_number = []

for f in fs:
    people_number.append( (f, len(os.listdir(src_path + f))) )

# Sort in descending order of people number of images
people_number = sorted(people_number, key=lambda x: x[1], reverse=True)

# For num_classes copy the classes with most images in their folders
for person in people_number[:num_classes]:
    name = person[0]
    
    formatted_num_images = str(person[1]).zfill(3)
    new_folder_name = "{} {}".format(formatted_num_images, name)
  
    src=os.path.join(src_path, name)
    dst=os.path.join(dest_path, new_folder_name)
    shutil.copytree(src,dst)


# Map each class to an integer label
class_mapping = {}
class_images = {}

# Create dictionary to map integer labels to individuals
# Class_images will record number of images for each class
for index, directory in enumerate(os.listdir(dest_path)):
    class_mapping[index] = directory.split(" ")[1]
    class_images[index] = int(directory.split(' ')[0])

# print(class_mapping)
# print(class_images)

total_num_images = np.sum(list(class_images.values()))

print("Individual \t Composition of Dataset\n")
for label, num_images in class_images.items():
    print("{:20} {:.2f}%".format(class_mapping[label], (num_images / total_num_images) * 100))
