import os
import glob
import csv

root_path = './list/xd_train_feature'    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))
violents = []
normal = []

with open('list/xd_train.csv', 'w+') as f:  ## the name of feature list
    writer = csv.writer(f)
    writer.writerow(['path', 'label'])
    for i in range(len(files)):
        if i % 2 == 0:
            file = files[i]
            if '.npy' in file:
                if '_label_A' in file:
                    normal.append(file[:-10])
                else:
                    label = file.split('_label_')[1].split('_')[0]
                    writer.writerow([file[:-10], label])

    for file in normal:
        writer.writerow([file, 'A'])