import os
import os.path

import numpy as np

from skimage import io
from preprocessing import FaceDetector, FaceAligner, clip_to_range
from tqdm import tqdm
from itertools import repeat

fd = FaceDetector()
fa = FaceAligner('../model/shape_predictor_68_face_landmarks.dat', '../model/face_template.npy')

IMAGE_FORMATS = {'.jpg', '.jpeg', '.png'}


def get_images(path):
    return list(filter(lambda s: os.path.isfile(os.path.join(path, s)) and os.path.splitext(s)[1] in IMAGE_FORMATS, os.listdir(path)))


def get_folders(path):
    return list(filter(lambda s: os.path.isdir(os.path.join(path, s)), os.listdir(path)))


def list_data(data_path):
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    dev_dir = os.path.join(data_path, 'dev')

    train = []
    test = []
    subjects = get_folders(train_dir)
    subjects.sort()

    for subj in subjects:
        subj_train_dir = os.path.join(train_dir, subj)
        subj_test_dir = os.path.join(test_dir, subj)

        train_files = get_images(os.path.join(train_dir, subj))
        test_files = get_images(os.path.join(test_dir, subj))

        train_files.sort()
        test_files.sort()

        train_files = list(map(lambda s: os.path.join(subj_train_dir, s), train_files))
        test_files = list(map(lambda s: os.path.join(subj_test_dir, s), test_files))

        subj = int(subj.split('_')[1])

        train.extend(zip(train_files, repeat(subj)))
        test.extend(zip(test_files, repeat(subj)))

    dev = get_images(dev_dir)
    dev.sort(key=lambda s: int(os.path.splitext(s)[0]))
    dev = list(map(lambda s: os.path.join(dev_dir, s), dev))
    dev = list(zip(dev, repeat(-1)))

    return train, test, dev


def load_file(filename, imsize=96, border=0):
    total_size = imsize + 2 * border

    img = io.imread(filename)
    faces = fd.detect_faces(img, get_top=1)

    if len(faces) == 0:
        return None

    face = fa.align_face(img, faces[0], dim=imsize, border=border).reshape(1, total_size, total_size, 3)
    face = clip_to_range(face)

    del img

    return face.astype(np.float32)


def load_data(data, not_found_policy='throw_away', available_subjects=None, imsize=96, border=0):
    n_data = len(data)

    total_size = imsize + 2 * border

    images = np.zeros((n_data, total_size, total_size, 3), dtype=np.float32)
    labels = np.zeros((n_data,), dtype=np.int)

    if available_subjects is not None:
        available_subjects = set(available_subjects)

    black = np.zeros((1, total_size, total_size, 3), dtype=np.float32)

    face_not_found_on = []

    img_ptr = 0
    for filename, subject in tqdm(data):
        if available_subjects is not None:
            if subject not in available_subjects:
                continue

        face_img = load_file(filename, imsize=imsize, border=border)

        if face_img is None:
            face_not_found_on.append(filename)
            if not_found_policy == 'throw_away':
                continue
            elif not_found_policy == 'replace_black':
                face_img = black
            else:
                raise Exception('Face not found on {}'.format(filename))

        images[img_ptr] = face_img
        labels[img_ptr] = subject
        img_ptr += 1

    images = images[:img_ptr]
    labels = labels[:img_ptr]

    if len(face_not_found_on) > 0:
        print('[Warning] Faces was not found on:')
        for f in face_not_found_on:
            print(' - {}'.format(f))

    return images, labels


IMSIZE = 217
BORDER = 5


train, test, dev = list_data('../data')

###
print('Loading train files...')
train_x, train_y = load_data(train, imsize=IMSIZE, border=BORDER, not_found_policy='throw_away')

del train

mean = train_x.mean(axis=0)
stddev = train_x.std(axis=0)

np.save('../model/mean', mean)
np.save('../model/stddev', stddev)

train_x -= mean
train_x /= stddev

np.save('../data/train_x', train_x)
np.save('../data/train_y', train_y)
###

del train_x

###
print('Loading test files...')
test_x, test_y = load_data(test, imsize=IMSIZE, border=BORDER, not_found_policy='throw_away', available_subjects=train_y)

del test, train_y

test_x -= mean
test_x /= stddev

np.save('../data/test_x', test_x)
np.save('../test_y', test_y)
###

del test_x, test_y

###
print('Loading dev files...')
dev_x, _ = load_data(dev, imsize=IMSIZE, border=BORDER, not_found_policy='replace_black')

del dev

dev_x -= mean
dev_x /= stddev

np.save('../data/dev_x', dev_x)
###
