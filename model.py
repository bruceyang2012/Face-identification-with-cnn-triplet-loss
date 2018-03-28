import os.path
import numpy as np
from preprocessing import FaceDetector, FaceAligner, clip_to_range
import keras.backend as K
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Lambda, Input, merge, Conv2D, Activation, Dropout, MaxPooling2D, Flatten, GlobalMaxPooling2D, BatchNormalization

GREATER_THAN = 32
BATCH_SIZE = 128
IMSIZE = 217
IMBORDER = 5

def build_my_cnn(dim,n_class):
    model = Sequential()

    model.add(BatchNormalization(input_shape=(dim,dim,3)))
    
    model.add(Conv2D(32, kernel_size=(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128,kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(GlobalMaxPooling2D())
    
    model.add(Dense(64)) #512
    model.add(Activation('relu'))    
    
    model.add(Dense(n_class))
    model.add(Activation('sigmoid'))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

def triplet_loss(y_true, y_pred):
    return -K.mean(K.log(K.sigmoid(y_pred)))

def triplet_merge(inputs):
    a, p, n = inputs
    return K.sum(a * (p - n), axis=1)

def triplet_merge_shape(input_shapes):
    return (input_shapes[0][0], 1)

def build_tpe(n_in, n_out, W_pca=None):
    a = Input(shape=(n_in,))
    p = Input(shape=(n_in,))
    n = Input(shape=(n_in,))

    if W_pca is None:
        W_pca = np.zeros((n_in, n_out))

    base_model = Sequential()
    base_model.add(Dense(n_out, input_dim=n_in, bias=False, weights=[W_pca], activation='linear'))
    base_model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    a_emb = base_model(a)
    p_emb = base_model(p)
    n_emb = base_model(n)

    e = merge([a_emb, p_emb, n_emb], mode=triplet_merge, output_shape=triplet_merge_shape)

    model = Model(input=[a, p, n], output=e)
    predict = Model(input=a, output=a_emb)

    model.compile(loss=triplet_loss, optimizer='rmsprop')

    return model, predict

class Bottleneck:
    def __init__(self, model, layer):
        self.fn = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])

    def predict(self, data_x, batch_size=32, learning_phase=False):
        n_data = len(data_x)
        n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)

        result = None

        learning_phase = 1 if learning_phase else 0

        for i in range(n_batches):
            batch_x = data_x[i * batch_size:(i + 1) * batch_size]
            batch_y = self.fn([batch_x, 0])[0]

            if result is None:
                result = batch_y
            else:
                result = np.vstack([result, batch_y])

        return result

class FaceVerificator:
    def __init__(self, model_dir):
        self._model_dir = model_dir

        self._model_files = {
            'shape_predictor': os.path.join(model_dir, 'shape_predictor_68_face_landmarks.dat'),
            'face_template': os.path.join(model_dir, 'face_template.npy'),
            'mean': os.path.join(model_dir, 'mean.npy'),
            'stddev': os.path.join(model_dir, 'stddev.npy'),
            'cnn_weights': os.path.join(model_dir, 'weights_cnn.h5'),
            'tpe_weights': os.path.join(model_dir, 'weights_tpe.h5'),
        }

    def initialize_model(self):
        self._mean = np.load(self._model_files['mean'])
        self._stddev = np.load(self._model_files['stddev'])
        self._fd = FaceDetector()
        self._fa = FaceAligner(self._model_files['shape_predictor'],
                               self._model_files['face_template'])
        cnn = build_my_cnn(227, 24)
        cnn.load_weights(self._model_files['cnn_weights'])
        self._cnn = Bottleneck(cnn, ~1)
        _, tpe = build_tpe(24, 24)
        tpe.load_weights(self._model_files['tpe_weights'])
        self._tpe = tpe

    def normalize(self, img):
        img = clip_to_range(img)
        return (img - self._mean) / self._stddev

    def process_image(self, img):
        face_rects = self._fd.detect_faces(img, upscale_factor=2, greater_than=GREATER_THAN)

        if not face_rects:
            return []

        faces = self._fa.align_faces(img, face_rects, dim=IMSIZE, border=IMBORDER)
        faces = list(map(self.normalize, faces))

        faces_y = self._cnn.predict(faces, batch_size=BATCH_SIZE)
        faces_y = self._tpe.predict(faces_y, batch_size=BATCH_SIZE)

        return list(zip(face_rects, faces_y))

    def compare_many(self, dist, xs, ys):
        xs = np.array(xs)
        ys = np.array(ys)
        scores = xs @ ys.T
        return scores, scores > dist
