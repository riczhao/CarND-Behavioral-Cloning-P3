import csv
import matplotlib.pyplot as plt
import numpy as np
import keras.layers as layers
import keras
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import math

crop_top = 58
crop_bottom = 30
crop_x = crop_top + crop_bottom

class DataGenerator(object):
    def __init__(self, path='data/'):
        self.path = path
        self.h_flip_rate = 0.5
        self.noise_std = 0.
        self.n_samples = 0
        x,y = self.parse_path()
        self.train_valid_split(x, y)
        self.sample_showed = False

    def parse_path(self):
        ''' get x,y, x is picture path '''
        lines = []
        with open(self.path+"driving_log.csv") as f:
            reader = csv.reader(f)
            title = True
            for line in reader:
                if title:
                    title = False
                    continue
                lines.append(line)
        self.n_samples = len(lines)
        x = []
        y = []
        for line in lines:
            if abs(float(line[3])) < 0.05:
                x.append(line[1]) # left
                y.append(float(line[3])+0.3)
                x.append(line[2]) #right
                y.append(float(line[3])-0.3)
            elif abs(float(line[3])) > 0.7:
                x.append(line[0]) # center
                y.append(float(line[3]))
                x.append(line[0]) # center
                x.append(line[0]) # center
                y.append(float(line[3]))
                y.append(float(line[3]))
        y = np.array(y)
        return x,y

    def train_valid_split(self, x, y, valid_size=0.1):
        self.x_train,self.x_val,self.y_train,self.y_val = \
            train_test_split(x, y, test_size=valid_size)
        self.n_train = len(self.x_train)
        self.n_val = len(self.x_val)
        
    def shuffle_train(self):
        shuffle(self.x_train, self.y_train)

    def gen_batch_train(self, x_path, y):
        x = []
        for i in range(len(x_path)):
            fn = x_path[i]
            img = plt.imread(fn)[crop_top:-crop_bottom]/255.
            h_flip = random.random() <= self.h_flip_rate
            if h_flip:
                img = np.flip(img,1)
                y[i] = -y[i]
            if self.noise_std:
                img += np.random.normal(scale=self.noise_std,size=img.shape)
            x.append(img)
        return (np.array(x), y)
    
    def gen_batch_val(self, x_path, y):
        x = []
        for fn in x_path:
            img = plt.imread(fn)[crop_top:-crop_bottom]/255.
            x.append(img)
        return (np.array(x), y)
    
    def show_samples(self):
        self.shuffle_train()
        x,y = self.gen_batch_train(self.x_train[0:12],
                             self.y_train[0:12])
        for i in range(12):
            plt.subplot(4,3,i+1)
            plt.imshow(x[i])
            plt.title('y:'+str(y[i]))
        plt.show()
        print(len(self.x_train))
 
 
    def train_flow(self, batch_size):
        while True:
            self.shuffle_train()
            for idx in range(0,self.n_train,batch_size):
               yield self.gen_batch_train(self.x_train[idx:idx+batch_size],
                                          self.y_train[idx:idx+batch_size])
    
    def valid_flow(self, batch_size):
        while True:
            for idx in range(0, self.n_val, batch_size):
                yield self.gen_batch_val(self.x_val[idx:idx+batch_size],
                                         self.y_val[idx:idx+batch_size])

def get_model():
    model = keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=(160-crop_x,320,3)))
    model.add(layers.GaussianNoise(0.1))
    model.add(layers.Conv2D(6,(3,3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(16,(3,3), activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')

    return model

def get_nv_model():
    model = keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=(160-crop_x,320,3))) # 102x320
    #model.add(layers.GaussianNoise(0.1))
    model.add(layers.Conv2D(24,(5,5), activation='relu')) # 98x316
    model.add(layers.Dropout(0.4))
    model.add(layers.MaxPool2D(padding='same')) # 49x158
    model.add(layers.Conv2D(36,(5,5), activation='relu')) # 45x154
    model.add(layers.MaxPool2D(padding='same')) # 23x77
    model.add(layers.Conv2D(48,(5,5), activation='relu')) # 19x73
    model.add(layers.Dropout(0.4))
    model.add(layers.MaxPool2D(padding='same')) # 10x37
    model.add(layers.Conv2D(64,(3,3), activation='relu')) # 8x35
  #  model.add(layers.MaxPool2D(padding='same')) # 4x18
    model.add(layers.Conv2D(64,(3,3), activation='relu')) # 2x16
  #  model.add(layers.MaxPool2D(padding='same')) # 1x8
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.compile('adam', 'mse')

    return model

def get_nvidia_model(input_shape):
    model = keras.models.Sequential()
    model.add(layers.GaussianNoise(0.1, input_shape=input_shape))
    model.add(layers.Conv2D(24, 5, 5, activation='elu',
                          subsample = (2,2),border_mode='valid',init = 'he_normal'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(36, 5, 5,  activation='elu',
                          subsample = (2,2),border_mode='valid',init = 'he_normal'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(48, 5, 5,  activation='elu',
                          subsample = (2,2),border_mode='valid',init = 'he_normal'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, 3, 3,  activation='elu',
                          subsample = (1,1),border_mode='valid',init = 'he_normal'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, 3, 3,  activation='elu',
                          subsample = (1,1),border_mode='valid',init = 'he_normal'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1164,init='he_normal',activation='elu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(100,init='he_normal',activation='elu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(50,init='he_normal',activation='elu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10,init='he_normal',activation='elu'))
    model.add(layers.Dense(1,init='he_normal'))
    model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error')
    return model

if __name__ == '__main__':
    model_name = 'lenet'
    input_shape=(160-crop_x,320,3)
    gen = DataGenerator(path='c:\data\\')
    #for i in range(10):
    #    gen.show_samples()
    if model_name == 'nv':
        model = get_nv_model()
        batch_size = 32
    elif model_name == 'lenet':
        model = get_model()
        batch_size = 100
    else:
        model = get_nvidia_model(input_shape)
        batch_size = 32
    try:
        model.load_weights(model_name+'_weights.h5')
    except:
        pass
    keras.utils.vis_utils.plot_model(model, to_file=model_name+'.png')
    hist = model.fit_generator(gen.train_flow(batch_size), math.ceil(gen.n_train/batch_size)/21,
                        epochs=20,
                        validation_data=gen.valid_flow(batch_size),
                        validation_steps=math.ceil(gen.n_val/batch_size),
                        verbose=1)
    #model.fit(x, y, batch_size=100, epochs=1, validation_split=0.2, shuffle=True)
    model.save_weights(model_name+'_weights.h5', True)
    model.save(model_name+'.h5')

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()