from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import AveragePooling2D,Dropout

def create_model(train_features):
  model = models.Sequential()
  model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=train_features.shape[1:]))
  model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(rate=0.25))
  model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(rate=0.25))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(rate=0.5))
  model.add(Dense(43, activation='softmax'))
  return model


if __name__ == "__main__" :
    x_train_gray_norm = np.load('train_features.npy', allow_pickle= True)
    y_train = np.load('train_labels.npy', allow_pickle= True)
    x_validate_gray_norm = np.load('validate_features.npy', allow_pickle= True)
    y_validate = np.load('validate_labels.npy', allow_pickle= True)

    
    model = create_model(x_train_gray_norm)

    model.compile(loss="sparse_categorical_crossentropy", optimizer ="adam", metrics=['accuracy'])
    model.fit(x_train_gray_norm, y_train,
          batch_size=256,
          epochs=30,
          verbose=1,
          validation_data=(x_validate_gray_norm,y_validate))
    model.save(f"model/trained_model")

    