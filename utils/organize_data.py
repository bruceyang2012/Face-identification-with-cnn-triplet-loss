import os
import os.path
import random
import math
import itertools
import shutil

import numpy as np

from collections import namedtuple


FORMATS = {'.jpg', '.jpeg', '.png'}
DATA_DIR = './data/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
DEV_DIR = DATA_DIR + 'dev/'
PROTOCOLS_DIR = './data/'

BASE_DIR = 'C:/DeepLearning/faceDetection_and_recognition/face_data/'
# FEI_DIR = BASE_DIR + 'fei/'
CAL_DIR = BASE_DIR + 'caltech_faces/'
print(CAL_DIR)
# GT_DIR = BASE_DIR + 'gt_db/'
# LFW2_DIR = BASE_DIR + 'lfw2/'


DEV_RATIO = 0.04
TEST_RATIO = 0.06

DO_NOT_COPY = False
Entry = namedtuple('Entry', ['subject', 'name', 'path'])

def grab_db_plain(path, divisor):
    res = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        ext = os.path.splitext(file)[1]
        if os.path.isfile(file_path) and ext in FORMATS:
            subject, name = file.split(divisor)
            res.append(Entry(path + subject, name, file_path))

    return res

def grab_db_folders(path):
    res = []

    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                ext = os.path.splitext(file)[1]
                if os.path.isfile(file_path) and ext in FORMATS:
                    res.append(Entry(path + dir, file, file_path))

    return res

def get_entry_subjects(xs):
    return list(set(map(lambda e: e.subject, xs)))

def get_subjects(entries):
    subjects = get_entry_subjects(entries)
    n_subjects = len(subjects)
    n_dev_subjects = max(1, math.ceil(n_subjects * DEV_RATIO))
    random.shuffle(subjects)

    return subjects[:n_dev_subjects], subjects[n_dev_subjects:]

def copy_files(files, dest_dir):
    if DO_NOT_COPY:
        return

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    h = 0
    for file in files:
        ext = os.path.splitext(file)[1]
        shutil.copy(file, os.path.join(dest_dir, str(h) + ext))
        h += 1

def main():
    for dir in {DATA_DIR, TRAIN_DIR, TEST_DIR, DEV_DIR, PROTOCOLS_DIR}:
        if not os.path.exists(dir):
            print('Creating {}'.format(dir))
            os.makedirs(dir)

    entries = [
        #('fei', grab_db_plain(FEI_DIR, '-')),
        ('caltech_faces', grab_db_folders(CAL_DIR)),
        #('gt', grab_db_folders(GT_DIR))
        # ('lfw2', grab_db_folders(LFW2_DIR))
    ]

    all_entries = list(itertools.chain.from_iterable((map(lambda p: p[1], entries))))
    all_subjects = get_entry_subjects(all_entries)
    n_files_total = len(all_entries)
    n_subjects_total = len(all_subjects)

    print('-' * 10)

    print('Taking for development set {:.2f}% of subjects'.format(DEV_RATIO * 100))

    subjects_dev = []
    subjects = []

    print('-' * 10)

    for name, e in entries:
        n_e_files = len(e)
        print('Total files in "{}" set: {}'.format(name, n_e_files))
        subjects_dev_e, subjects_e = get_subjects(e)
        subjects_dev += subjects_dev_e
        subjects += subjects_e

    print('-' * 10)
    print('Total files: {}'.format(n_files_total))
    print('Total subjects: {}'.format(n_subjects_total))
    print('-' * 10)

    n_subjects_dev = len(subjects_dev)
    n_subjects = len(subjects)

    print('Number of subjects for development set: {}'.format(n_subjects_dev))
    print('Number of subjects for train/test set: {}'.format(n_subjects))

    dev_files = []
    protocol_data = []

    for subj in subjects_dev:
        subj_entries = list(map(lambda e: e.path, filter(lambda e: e.subject == subj, all_entries)))
        n_subjects_entries = len(subj_entries)
        dev_files.extend(subj_entries)
        protocol_data.append(n_subjects_entries)

    n_dev_files = sum(protocol_data)
    protocol = np.zeros((n_dev_files, n_dev_files), dtype=np.bool)

    k = 0
    for i in protocol_data:
        protocol[k:k + i, k:k + i] = 1
        k += i

    print('-' * 10)
    print('Total dev files: {}'.format(n_dev_files))

    # print(dev_files)

    copy_files(dev_files, DEV_DIR)

    np.save(PROTOCOLS_DIR + 'dev_protocol', protocol)

    n_test_files = 0

    h = 0
    for subj in subjects:
        subj_name = 'subject_' + str(h)
        h += 1

        subj_entries = list(map(lambda e: e.path, filter(lambda e: e.subject == subj, all_entries)))
        n_subj_entries = len(subj_entries)
        random.shuffle(subj_entries)

        for_test = 0

        if n_subj_entries > 1:
            for_test = max(1, math.ceil(n_subj_entries * TEST_RATIO))

        n_test_files += for_test

        entries_test, entries_train = subj_entries[:for_test], subj_entries[for_test:]

        subj_train_dir = os.path.join(TRAIN_DIR, subj_name)
        subj_test_dir = os.path.join(TEST_DIR, subj_name)

        copy_files(entries_train, subj_train_dir)
        copy_files(entries_test, subj_test_dir)

    print('Test files: {}'.format(n_test_files))
    print('Train files: {}'.format(n_files_total - k - n_test_files))

    print('Done!')


if __name__ == '__main__':
    main()
