import pickle
with open('all_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    train_labels = pickle.load(f)
    val_data = pickle.load(f)
    val_labels = pickle.load(f)

with open('train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    pickle.dump(train_labels, f)

with open('val_data.pkl', 'wb') as f:
    pickle.dump(val_data, f)
    pickle.dump(val_labels, f)
