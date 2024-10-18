#%%
# Import libraries
import os
import shutil
import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

#%%
# Create defect_img_df

# Load the instance.json file
with open('./original/annotations/instance.json') as f:
    data = json.load(f)

# Create a dictionary to store defects for each image
image_defects = defaultdict(list)

# Iterate through annotations to collect category_ids for each image
for annotation in data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    image_defects[image_id].append(category_id)

# Prepare the DataFrame data
df_data = []
for image in data['images']:
    image_id = image['id']
    file_name = image['file_name']
    defects = image_defects.get(image_id, [])  # Get defects or an empty list if none
    df_data.append({'image_path': file_name, 'defects': defects})

# Convert to DataFrame
anonnated_img_df = pd.DataFrame(df_data)

# Create a defect_img_df, which only contains images with defects. The defects 1 and 5 are acrually not defects. Each row might have multiple defects, value of the column is a list.
defect_img_df = anonnated_img_df[anonnated_img_df['defects'].apply(lambda x: len(x) > 0 and x[0] not in [1, 5])]

# Count the occurrence of each defect, skipping defects 1 and 5
defect_count = defaultdict(int)
for defects in defect_img_df['defects']:
    for defect in defects:
        if defect not in [1, 5]:
            defect_count[defect] += 1

#%%
# Create normal_img_df

normal_img_dir = './original/normal/A/'

# Get all image paths in the normal image directory including subdirectories
normal_img_paths = []
for path in Path(normal_img_dir).rglob('*.jpg'):
    normal_img_paths.append(path)

# Create a DataFrame for normal images
normal_img_df = pd.DataFrame(normal_img_paths, columns=['image_path'])

#%%
# Construct the custom_dataset directory

"""
custom_dataset/
├── train/
│   └── good/
│       ├── image1.png
│       ├── image2.png
│       └── ...
├── test/
│   ├── good/
│   │   ├── imageX.png
│   │   └── ...
│   └── defect_type_1/
│       ├── imageY.png
│       └── ...
"""

defect_img_dir = './original/images/'
custom_dataset_dir = Path('./leddd')
train_ratio = 0.8

# Create the custom_dataset directory
custom_dataset_dir.mkdir(parents=True, exist_ok=True)

# Create the train and test directories
train_dir = custom_dataset_dir / 'train'
test_dir = custom_dataset_dir / 'test'
train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

# Create the good and defect directories in train and test
good_train_dir = train_dir / 'good'
good_test_dir = test_dir / 'good'
good_train_dir.mkdir(exist_ok=True)
good_test_dir.mkdir(exist_ok=True)

defect_train_dirs = {}
defect_test_dirs = {}
for defect in defect_count.keys():
    defect_test_dirs[defect] = test_dir / f'defect_type_{defect}'
    defect_test_dirs[defect].mkdir(exist_ok=True)

# Split the normal images into train and test
normal_img_df = normal_img_df.sample(frac=1)  # Shuffle the DataFrame
normal_img_count = len(normal_img_df)
normal_img_train_count = int(normal_img_count * train_ratio)
normal_img_train_df = normal_img_df.iloc[:normal_img_train_count]
normal_img_test_df = normal_img_df.iloc[normal_img_train_count:]

# Copy the normal images to the train and test directories. Copy, not move.
for i, row in normal_img_train_df.iterrows():
    image_path = row['image_path']
    image_name = os.path.basename(image_path)
    new_image_path = good_train_dir / image_name
    shutil.copy(image_path, new_image_path)

for i, row in normal_img_test_df.iterrows():
    image_path = row['image_path']
    image_name = os.path.basename(image_path)
    new_image_path = good_test_dir / image_name
    shutil.copy(image_path, new_image_path)
    
# Copy the defect images to the test directories. Copy, not move.
for i, row in defect_img_df.iterrows():
    image_path = os.path.join(defect_img_dir, row['image_path'])
    image_name = os.path.basename(image_path)
    defects = row['defects']
    for defect in defects:
        if defect in defect_test_dirs:
            new_image_path = defect_test_dirs[defect] / image_name
            try:
                shutil.copy(image_path, new_image_path)
            except FileNotFoundError:
                print(f'cp {image_path} {new_image_path}')

#%%