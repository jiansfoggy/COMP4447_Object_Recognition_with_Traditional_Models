import numpy as np
import joblib, os, pickle
from skimage import util, transform, filters
from skimage.util import random_noise
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Augmentation function
def augment_image(img):
    # img: HxWxC in [0,1]
    # 1. add random noise with 20% probability
    if np.random.rand() < 0.2:
        img = random_noise(img, mode='gaussian')
    # 2. rotate with angle in [-20,20], 20% probability
    if np.random.rand() < 0.2:
        angle = np.random.uniform(-20, 20)
        img = transform.rotate(img, angle, mode='edge')
    # 3. horizontal flip with 20%
    if np.random.rand() < 0.2:
        img = np.fliplr(img)
    # 4. blur with gaussian, 20%
    if np.random.rand() < 0.2:
        sigma = np.random.uniform(0.5, 1.5)
        img = filters.gaussian(img, sigma=sigma, channel_axis=-1)
    return img

# Utility to unpickle CIFAR-10 batch files
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# Modified data loading and splitting function
def data_load_split():
    """
    Load CIFAR-10 data from local pickled batches, split into train/validation sets.

    Args:
        data_dir (str): Path to the folder containing CIFAR-10 batch files.
        test_ratio (float): Proportion of the training data to use as validation.
        random_state (int): Seed for reproducibility.

    Returns:
        X_train: np.ndarray of shape (num_train_samples, 3072), float32 in [0,1]
        X_val:   np.ndarray of shape (num_val_samples, 3072), float32 in [0,1]
        y_train: np.ndarray of shape (num_train_samples,)
        y_val:   np.ndarray of shape (num_val_samples,)
    """
    # 1. Load all five training batches
    X_list, y_list = [], []
    for i in range(1, 6):
        batch_path = f'./cifar-10-data/data_batch_{i}'
        batch = unpickle(batch_path)
        # data is uint8 [10000, 3072]
        X_list.append(batch[b'data'])
        y_list.extend(batch[b'labels'])

    X_train = np.concatenate(X_list, axis=0)         # shape (50000, 3072)
    y_train = np.array(y_list, dtype=np.int64)       # shape (50000,)

    batch_path = f'./cifar-10-data/test_batch'
    batch = unpickle(batch_path)
    X_test = np.array(batch[b'data'], dtype=np.float32) # shape (10000, 3072)
    y_test = np.array(batch[b'labels'], dtype=np.int64) # shape (10000,)

    # Reshape and normalize to [0, 1]
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    X_test  = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0

    return X_train, X_test, y_train, y_test

# def data_load_split():
#     # 1. Load CIFAR-10 and split into train/validation
#     (x, y), _ = cifar10.load_data()
#     y = y.flatten()
#     x = x.astype(np.float32) / 255.0  # scale to [0,1]

#     X_train, X_test, y_train, y_test = train_test_split(
#         x, y, test_size=0.1, random_state=42, stratify=y_all)
#     return X_train, X_test, y_train, y_test

# Augment and Normalize the images
def data_augment_normalization(X_train, X_test):
    # Apply augmentation to training set
    X_train_aug = np.array([augment_image(img) for img in X_train])

    # Flatten for classifier
    ns, h, w, c = X_train_aug.shape
    X_train_flat = X_train_aug.reshape(ns, -1)
    nv, _, _, _ = X_test.shape
    X_test_flat = X_test.reshape(nv, -1)

    # 5. z-score normalization using training set statistics
    mean = X_train_flat.mean(axis=0)
    std = X_train_flat.std(axis=0) + 1e-8
    X_train_norm = (X_train_flat - mean) / std
    X_test_norm  = (X_test_flat - mean) / std
    return X_train_norm, X_test_norm

# model function contains two models: SVC and RFC
def model(X_train, X_test, y_train):
    # 2a. Train SVM
    # C = [0.01,0.1,0.3,0.5,0.7,1,10]
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_clf.fit(X_train, y_train)
    joblib.dump(svm_clf, f'svm_weight.pkl')
    y_pred_svm = svm_clf.predict(X_test)

    # 2b. Train Randxom Forest
    # n_estimators = [10,50,100,120,150,200]
    # max_depth = [5,10,15,20,25]
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf, 'rf_weight.pkl')
    y_pred_rf = rf_clf.predict(X_test)
    return y_pred_svm, y_pred_rf

# evaluate the trained model
def evaluation(y_pred_svm, y_pred_rf, y_test):
    # 3. Evaluation
    classes = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    # print("=== SVM Classification Report ===")
    # print(classification_report(y_test, y_pred_svm, target_names=classes))
    # print("SVM Overall Accuracy:", accuracy_score(y_test, y_pred_svm))

    # print("\n=== Random Forest Classification Report ===")
    # print(classification_report(y_test, y_pred_rf, target_names=classes))
    # print("Random Forest Overall Accuracy:", accuracy_score(y_test, y_pred_rf))

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("\nSVM Confusion Matrix:\n", cm_svm)
    print("\nRandom Forest Confusion Matrix:\n", cm_rf)

    record = open(f"./results.txt", 'a') # -atlas
    record.write("="*50+"\n")
    record.write("=== SVM Classification Report ===\n")
    record.write(classification_report(y_test, y_pred_svm, target_names=classes))
    record.write(f"SVM Overall Accuracy: {accuracy_score(y_test, y_pred_svm)}\n")
    record.write("="*50+"\n")
    record.write("=== Random Forest Classification Report ===\n")
    record.write(classification_report(y_test, y_pred_rf, target_names=classes))
    record.write(f"Random Forest Overall Accuracy: {accuracy_score(y_test, y_pred_rf)}\n")
    record.write("="*50+"\n")
    record.write(f"SVM Confusion Matrix: {cm_svm}\n")
    record.write(f"Random Forest Confusion Matrix: {cm_rf}")
    record.write("="*50+"\n")
    record.close()

# ablation study to tune parameters -- C
def model_svm(X_train, X_test, y_train, ind):
    # 2a. Train SVM
    Cs = [0.01,0.1,0.5,0.7,1,10,15,20]
    svm_clf = SVC(kernel='rbf', C=Cs[ind], gamma='scale', random_state=42)
    svm_clf.fit(X_train, y_train)
    joblib.dump(svm_clf, f'svm_weight_C_{ind}.pkl')
    y_pred_svm = svm_clf.predict(X_test)
    para = f"C={Cs[ind]}"
    return y_pred_svm, para

# ablation study to tune hyperparameters -- n_estimators
def model_rf(X_train, X_test, y_train, ind):
    # 2b. Train Randxom Forest
    ns = [10,50,100,120,150,200,250,300]
    # max_depth = [5,10,15,20,25]
    rf_clf = RandomForestClassifier(n_estimators=ns[ind], random_state=42)
    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf, f'rf_weight_n_{ind}.pkl')
    y_pred_rf = rf_clf.predict(X_test)
    para = f"n_estimators={ns[ind]}"
    return y_pred_rf, para

# predict the result of selected hyperparameters.
def evaluation_model(y_pred, y_test, model_name, para):
    # 3. Evaluation
    classes = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    cm = confusion_matrix(y_test, y_pred)

    record = open(f"./results.txt", 'a') # -atlas
    record.write("="*50+"\n")
    record.write(f"=== {model_name} Classification Report ===\n")
    record.write(f"{para}\n")
    record.write(classification_report(y_test, y_pred, target_names=classes))
    record.write(f"{model_name} Overall Accuracy: {accuracy_score(y_test, y_pred)}\n")
    record.write("="*50+"\n")
    record.write(f"{model_name} Confusion Matrix: {cm}\n")
    record.write("="*50+"\n")
    record.close()

# main function to contain the whole workflow
def main():

    X_train, X_test, y_train, y_test = data_load_split()
    X_train_norm, X_test_norm = data_augment_normalization(X_train, X_test)
    for i in range(7,8):
        y_pred_svm, para = model_svm(X_train_norm, X_test_norm, y_train, i)
        evaluation_model(y_pred_svm, y_test, "SVC", para)
        y_pred_rf, para  = model_rf(X_train_norm, X_test_norm, y_train, i)
        evaluation_model(y_pred_rf, y_test, "RFC", para)
    
    y_pred_svm, y_pred_rf = model(X_train_norm, X_test_norm, y_train)
    evaluation(y_pred_svm, y_pred_rf, y_test)

if __name__ == '__main__':
    main()

    # SVC
    # C = [0.01,0.1,0.5,0.7,1,10]
    # SVC_Acc = [0.3612, 0.4569, 0.5179, 0.5305, 0.5434, 0.5603]

    # RF
    # n_estimators = [10,50,100,120,150,200]
    # RF_Acc = [0.3517, 0.4375, 0.4595, 0.4645, 0.4643, 0.4712]
