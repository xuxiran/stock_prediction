import difflib
import pickle
import numpy as np

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    train_labels = pickle.load(f)

with open('val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)
    val_labels = pickle.load(f)

cnt = 0
j = 0
remove_val = np.zeros((62284,1))
for j in range(0, 495):
    val_str = val_data[j][0:64]
    if j%10 == 0:
        print(j)
    all_sele = []
    for i in range(0, 62284):

        # # accelerate
        # if abs(len(val_data[j])-len(train_data[i]))>128:
        #     continue

        train_str = train_data[i][0:64]
        matcher = difflib.SequenceMatcher(None, train_str, val_str)
        match = matcher.find_longest_match(0, len(train_str), 0, len(val_str))
        matchsize = match.size

        if matchsize > 0.8 * len(val_str):
            remove_val[i] = 1
            break

train_data = [train_data[i] for i in range(len(train_data)) if remove_val[i] == 0]
train_labels = [train_labels[i] for i in range(len(train_labels)) if remove_val[i] == 0]

with open('train_data_delk.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    pickle.dump(train_labels, f)

print(1)