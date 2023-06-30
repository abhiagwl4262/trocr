import os
import random

def create_data(image_dir, train_split_size=0.8):
    img_files = os.listdir(image_dir)
    names = [img_file.split("_")[0] for img_file in img_files]
    unique_names = list(set(names))
    
    num_train = int(len(unique_names)*train_split_size)
    num_val = len(unique_names) - num_train
    train_names = []
    val_names = random.sample(unique_names, num_val)

    for name in names:
        if name not in val_names:
            train_names.append(name)

    f = open("full_data/gt_train.txt", "a")
    for i, name in enumerate(names):
        if name in train_names:
            line = "\t".join([img_files[i], name]
            ) + "\n"
            f.write(line)
    f.close()

    f = open("full_data/gt_valid.txt", "a")
    for i, name in enumerate(names):
        if name in val_names:
            line = "\t".join([img_files[i], name]) + "\n"
            f.write(line)
    f.close()

os.remove("full_data/gt_train.txt")
os.remove("full_data/gt_valid.txt")

create_data("full_data/uganda", 1.0) 
create_data("full_data/name", 1.0)
create_data("name_insi/image", 0.75)