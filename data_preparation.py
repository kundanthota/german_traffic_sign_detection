import pandas as pd 
import mumpy as np
from PIL import Image
from sklearn.utils import shuffle

def data_loader(data):
    features = []
    labels = []
    for index in range(len(data)):
        features.append(np.array(Image.open(data['Path'][index].resize((28,28),Image.NEAREST)))
        labels.append(data['ClassId'][index])
    return features, labels

if __name__ == "__main__" :
    #loading data
    train_data = pd.read_csv('Train.csv')
    test_data = pd.read_csv('Test.csv')
    validation_data = pd.read_csv('Meta.csv')

    #shuffling training data
    x_train, y_train = shuffle(x_train, y_train)
    
    # to convert image from rgb scale to greyscale
    x_train_gray = np.sum(np.array(x_train)/3, axis = 3, keepdims = True)
    x_test_gray = np.sum(np.array(x_test)/3, axis = 3, keepdims = True)
    x_validate_gray = np.sum(np.array(x_validate)/3, axis = 3, keepdims = True)

    # to Normalize the image 
    x_train_gray_norm = (x_train_gray - 128)/128
    x_test_gray_norm = (x_test_gray - 128)/128
    x_validate_gray_norm = (x_validate_gray - 128)/128

    # saving loaded data into numpy files
    x_train_gray_norm.dump("data/train_features.npy")
    x_test_gray_norm.dump("data/test_features.npy")
    x_validate_gray_norm.dump("data/validate_features.npy")
    np.array(y_train).dump("train_labels.npy")
    np.array(y_test).dump("test_labels.npy")
    np.array(y_validate).dump("validate_labels.npy")