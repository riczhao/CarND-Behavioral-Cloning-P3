import csv
import matplotlib.pyplot as plt
import numpy as np
import keras.layers as layers
import keras

crop_x = 58

def get_data(path):
    lines = []
    with open(path+"driving_log.csv") as f:
        reader = csv.reader(f)
        title = True
        for line in reader:
            if title:
                title = False
                continue
            lines.append(line)
    images = []
    steerings = []
    for line in lines:
        f_center = path + line[0]
        images.append(plt.imread(f_center)[crop_x:])
        steering = float(line[3])
        steerings.append(steering)
    return np.array(images),np.array(steerings)

def get_model():
    model = keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=(160-crop_x,320,3)))
    model.add(layers.Conv2D(6,(3,3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(16,(3,3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.compile('adam', 'mse')

    return model

if __name__ == '__main__':
    x,y = get_data('data/')
    print(x.shape,y.shape)
    model = get_model()
    model.fit(x, y, batch_size=100, epochs=1, validation_split=0.2, shuffle=True)
    model.save('model.h5')
