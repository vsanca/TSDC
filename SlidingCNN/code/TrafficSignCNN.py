import numpy as np
import os
import glob
import h5py
from skimage import io, color, exposure, transform

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

#KERAS IMPORTS
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

#VARS
NUM_CLASSES = 43
IMG_SIZE = 48

#Preprocesiranje slike, 
def image_preprocess(img):
    #Normalizacija histograma
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    #Centriranje slike
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    #Skaliranje na jedinstvenu velicinu
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    #Prebacivanje ose slike kako bi bila u dobrom formatu za ulaz na CNN
    img = np.rollaxis(img,-1)

    return img

#Dobavljanje klase na osnovu naziva foldera u kojem se slika nalazi
def get_class(image_path):
    return int(image_path.split('/')[-2])
    

try:
    #Navesti korektnu putanju do fajla
    with  h5py.File('/home/student/TrafficSign/inputs.h5') as hf: 
        X, Y = hf['imgs'][:], hf['labels'][:]
    print("Loaded images from inputs.h5")
except (IOError, OSError, KeyError):  
    print("Error in reading inputs.h5. Processing all images...")
    #Navesti korektnu putanju do direktorijuma
    root_dir = '/home/student/Desktop/shared/Final_Training/Images/'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        try:
            img = image_preprocess(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs)%100 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with h5py.File('/home/student/TrafficSign/inputs.h5','w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)

#prebacivanje slike iz formata (dim,x,y) u (x,y,dim) radi prikaza
def prepare_for_display(img):
    return np.rollaxis(img,0,3)

#wrapper metoda za olaksavanje testiranja, vraca klasu objekta
def match(model,img):
    return model.predict(img.reshape(1,3,48,48)).argmax()

#definisanje CNN 
def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, IMG_SIZE, IMG_SIZE), activation='relu', name='Convolution 2D 1.'))
    model.add(Convolution2D(32, 3, 3, activation='relu', name='Convolution 2D 2.'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling 2D 1.'))
    model.add(Dropout(0.2, name='Dropout 1.'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', name='Convolution 2D 3.'))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='Convolution 2D 4.'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling 2D 2.'))
    model.add(Dropout(0.2, name='Dropout 2.'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='Convolution 2D 5.'))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='Convolution 2D 6.'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling 2D 3.'))
    model.add(Dropout(0.2, name='Dropout 3.'))

    model.add(Flatten(name='Flatten 1.'))
    model.add(Dense(512, activation='relu', name='Dense 1.'))
    model.add(Dropout(0.5, name='Dropout 4.'))
    model.add(Dense(43, activation='softmax', name='Output'))
    return model


def lr_schedule(epoch):
        return lr*(0.1**int(epoch/10))
        
        
def load_model(json = None, path = None):
    model = cnn_model()
    model = model_from_json(open('/home/student/TrafficSign/trained_model_json20.h5').read())
    model.load_weights('/home/student/TrafficSign/trained_model_weights20.h5')
    return model     
    
    
def run():
    #instanciranje CNN
    model = cnn_model()
    
    #podesavanje parametara i postavljanje CNN
    lr = 0.01
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
       
    batch_size = 32
    nb_epoch = 3
    
    #obucavanje CNN
    model.fit(X, Y,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_split=0.2,
              shuffle=True,
              callbacks=[LearningRateScheduler(lr_schedule),
                        ModelCheckpoint('/home/student/TrafficSign/model.h5',save_best_only=True)]
                )
    
    #cuvanje modela
    #open('/home/student/TrafficSign/trained_model_json5.h5','w').write(model.to_json())
    #model.save_weights('/home/student/TrafficSign/trained_model_weights5.h5')
    
    #ucitavanje modela
    model = model_from_json(open('/home/student/TrafficSign/trained_model_json5.h5').read())
    model.load_weights('/home/student/TrafficSign/trained_model_weights5.h5')
    
    
    #provera preciznosti klasifikacije
    import pandas as pd
    test = pd.read_csv('/home/student/Desktop/student/Final_Test/Images/GT-final_test.csv',sep=';')
    
    X_test = []
    y_test = []
    i = 0
    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('/home/student/Desktop/student/Final_Test/Images/',file_name)
        X_test.append(image_preprocess(io.imread(img_path)))
        y_test.append(class_id)
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    y_pred = model.predict_classes(X_test)
    acc = float(np.sum(y_pred==y_test))/np.size(y_pred)*100
    print("Test accuracy = {}%".format(acc))
    
    
    #dobavljanje i prikaz pogresno klasifikovanih slika
    #missmatched = np.where(y_test!=y_pred)
    #counter = 0;
    
    #for i in range(len(missmatched[0])):
    #    plt.imshow(plt.imread(os.path.join('/home/student/Desktop/student/Final_Test/Images/',test['Filename'][missmatched[0][i]])))    
    #    plt.figure(counter)
    #    counter = counter +1
    
    
    #racunanje confusion matrice i njeno preracunavanje u procentualni pogodak, cuvanje za MATLAB vizualizaciju
    tmp = [];
    testnp = np.array(test['ClassId'])
    
    for i in range(0,43):
        tmp.append(float(np.count_nonzero(testnp==i)))
    
    
    cm = confusion_matrix(y_test, y_pred)
    result = np.zeros(shape=(43,43))
    
    for i in range(0,43):
        np.divide(cm[i],tmp, result[i])
    
    np.savetxt("/home/student/Desktop/shared/result1.csv", result, fmt="%.6f",delimiter=",")  
    print("Results saved to a .csv file.")
    
    #ispis osobina modela CNN
    model.summary()