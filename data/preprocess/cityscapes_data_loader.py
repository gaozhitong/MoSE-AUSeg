
import h5py
from utils import utils
import numpy as np
import gzip
import struct
import os
from matplotlib import pyplot as plt
from pathlib import Path
from data.cityscapes_labels import labels as cityscapes_labels_tuple

# value from https://github.com/inferno-pytorch/inferno/blob/master/inferno/io/box/cityscapes.py
CITYSCAPES_MEAN = [0.28689554, 0.32513303, 0.28389177]
CITYSCAPES_STD = [0.18696375, 0.19017339, 0.18720214]

# Dunction to load idx, copied  from Morpho_MNIST/io
def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data

def load_idx(path):
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)

def normalize(img):

    return utils.normalise_images(img / 255, CITYSCAPES_MEAN, CITYSCAPES_STD)

def encode_bb_pred(bb_pred):

    orig_labels = np.arange(-1,34)
    train_labels = np.array([24,24,24,24,24,24,24,24,0,1,24,24,2,3,4,24,24,24,5,24,6,7,8,9,10,11,12,13,14,15,24,24,16,17,18])

    arr = np.empty(orig_labels.max() + 1, dtype=np.uint8)
    arr[orig_labels] = train_labels
    bb_pred = arr[bb_pred]

    return bb_pred

def map_labels_to_trainId(arr):
    """Remap ids to corresponding training Ids. Note that the inplace mapping works because id > trainId here!"""
    id2trainId = {label.id: label.trainId for label in cityscapes_labels_tuple}
    for id, trainId in id2trainId.items():
        arr[arr == id] = trainId
    return arr

def label_transform_fn(img):
    # 1. map to training label
    img = map_labels_to_trainId(img)
    # 2. get multiple label and prob
    img, prob = utils.get_array_of_modes(img)
    return img, prob

def pred_transform_fn(img):
    #map to training label
    img = map_labels_to_trainId(img)
    return img

def all_file_paths(dir, mode):
    # resplit part of training to validation set, and use original validation set as test set.
    if mode == 'val':
        path = Path(dir) / 'train'
    elif mode == 'test':
        path = Path(dir) / 'val'
    else:
        path = Path(dir) / mode
    img_files = []
    img_uid_files = []
    label_files = []
    label_uid_files = []
    # i = 0
    # Follow ProbUnet, contain 3 cities of Darmstadt, Mönchengladbach and Ulm
    val_cities = ['darmstadt', 'monchengladbach', 'ulm']
    for city in path.iterdir():
        if mode == 'val' and city.stem not in val_cities:
            continue
        # i+=1
        # if i >= 2:
        #     break
        pictures = sorted(city.iterdir())
        for p in pictures:
            if "leftImg8bit" in str(p.stem):
                img_files.append(np.load(p))
                img_uid_files.append(p.stem)
            if "gtFine_labelIds" in str(p.stem):
                label_files.append(np.load(p))
                label_uid_files.append(p.stem)

    return img_files, img_uid_files, label_files, label_uid_files

def all_file_paths_bpred(dir, dir_bb, mode, label_transform = False, input_normalize = False):
    # resplit part of training to validation set, and use original validation set as test set.
    if mode == 'val':
        path = Path(dir) / 'train'
        path_bb = Path(dir_bb) / 'train'
    elif mode == 'test':
        path = Path(dir) / 'val'
        path_bb = Path(dir_bb) / 'val'
    else:
        path = Path(dir) / mode
        path_bb = Path(dir_bb) / mode
    img_files = []
    img_uid_files = []
    label_files = []
    label_uid_files = []
    prediction_files = []
    prediction_uid_files = []
    prob_files = []
    # i = 0
    # Follow ProbUnet, contain 3 cities of Darmstadt, Mönchengladbach and Ulm
    val_cities = ['darmstadt', 'monchengladbach', 'ulm']
    paths = sorted(path.iterdir())
    for city in paths:
        if mode == 'val' and city.stem not in val_cities:
            continue
        if mode == 'train' and city.stem in val_cities:
            continue

        pictures = sorted(city.iterdir())
        for p in pictures:

            if "leftImg8bit" in str(p.stem):
                if input_normalize:
                    input = normalize(np.load(p))
                else:
                    input = np.load(p)
                img_files.append(input)
                img_uid_files.append(p.stem)
            if "gtFine_labelIds" in str(p.stem):
                if label_transform:
                    gt, prob = label_transform_fn(np.load(p))
                    prob_files.append(prob)
                else:
                    gt =  np.load(p)
                label_files.append(gt)
                label_uid_files.append(p.stem)
                pred_path = path_bb / city.stem / 'bb_preds' / p.stem.replace('gtFine_labelIds', 'prior_preds_trainIds.npy')

                if label_transform:
                    pred = (pred_transform_fn(np.load(pred_path)))
                else:
                    pred =  np.load(pred_path)
                prediction_files.append(pred)
                prediction_uid_files.append(pred_path.stem)

    if prob_files:
        return img_files, img_uid_files, label_files, label_uid_files, prediction_files, prediction_uid_files, prob_files

    return img_files, img_uid_files, label_files, label_uid_files, prediction_files, prediction_uid_files


def prepare_data(input_file, output_file):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    hdf5_file = h5py.File(output_file, "w")

    '''
    groups = {
        'train': {images}, {labels}, {uids}
        'val': {images}, {labels}, {uids}
        'test':{images}, {labels}, {uids}
    }
    '''
    groups = {}

    for tt in ['train', 'test', 'val']:
        groups[tt] = hdf5_file.create_group(tt)

    for tt in ['test', 'train', 'val']:
        img_files, img_uid_files, label_files, label_uid_files = all_file_paths(input_file, mode=tt)
        # groups[tt].create_dataset('uids', data=np.asarray(img_uid_files, dtype=np.int))
        groups[tt].create_dataset('labels', data=np.asarray(label_files, dtype=np.uint8))
        groups[tt].create_dataset('images', data=np.asarray(img_files, dtype=np.int))

    hdf5_file.close()

def prepare_data_with_bbpred(input_file, output_file, bbpred_file):

    hdf5_file = h5py.File(output_file, "w")

    '''
    groups = {
        'train': {images}, {labels}, {bbpred}
        'val': {images}, {labels}, {bbpred}
        'test':{images}, {labels}, {bbpred}
    }
    '''
    groups = {}

    for tt in ['test', 'train', 'val']:
        groups[tt] = hdf5_file.create_group(tt)

    for tt in ['test', 'train', 'val']:
        img_files, img_uid_files, label_files, label_uid_files, bbpred_files, bbpred_uid_files \
            = all_file_paths_bpred(input_file, bbpred_file, mode=tt,)
        groups[tt].create_dataset('bbpreds', data=np.asarray(bbpred_files, dtype=np.uint8))
        groups[tt].create_dataset('labels', data=np.asarray(label_files, dtype=np.uint8))
        groups[tt].create_dataset('images', data=np.asarray(img_files, dtype=np.int))

    hdf5_file.close()

def prepare_data_all(input_file, output_file, bbpred_file):

    hdf5_file = h5py.File(output_file, "w")

    '''
    groups = {
        'train': {images}, {labels}, {bbpred}
        'val': {images}, {labels}, {bbpred}
        'test':{images}, {labels}, {bbpred}
    }
    '''
    groups = {}

    for tt in ['test', 'train', 'val']:
        groups[tt] = hdf5_file.create_group(tt)

    for tt in ['test', 'train', 'val']:
        img_files, img_uid_files, label_files, label_uid_files, bbpred_files, bbpred_uid_files, prob_files \
            = all_file_paths_bpred(input_file, bbpred_file, mode=tt,  label_transform = True, input_normalize = True)
        groups[tt].create_dataset('bbpreds', data=np.asarray(bbpred_files, dtype=np.float32))
        groups[tt].create_dataset('labels', data=np.asarray(label_files, dtype=np.uint8))
        groups[tt].create_dataset('images', data=np.asarray(img_files, dtype=np.float32))
        groups[tt].create_dataset('probs', data=np.asarray(prob_files, dtype=np.float32))
    hdf5_file.close()

def load_and_maybe_process_data(input_file,
                                preprocessing_folder,
                                bbpred_file = None,
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the created MNIST data

    :param input_folder: Folder where the raw MNIST data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    data_file_name = 'data_cityscapes_5.hdf5'

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)


    if not os.path.exists(data_file_path) or force_overwrite:
        prepare_data_all(input_file, data_file_path, bbpred_file)
    else:
        pass

    return h5py.File(data_file_path, 'r')

def save_npy(data, vis_dir):
    stages = ['train', 'val', 'test']
    types = ['images', 'labels', 'probs', 'bbpreds']
    for s in stages:
        print(s)
        for t in types:
            if not os.path.exists(os.path.join(vis_dir, s, t)):
                os.makedirs(os.path.join(vis_dir, s, t))
            for index, image in enumerate(data[s][t]):
                np.save(os.path.join(vis_dir, s, t, str(index) + '.npy'), image)

    return

# def vis(data, vis_dir):
#     stages = ['test', 'train', 'val']
#     for s in stages:
#         if not os.path.exists(os.path.join(vis_dir, s)):
#             os.makedirs(os.path.join(vis_dir, s))
#         # for t in types:
#
#         index = 0
#         vis_num = 20
#         L = vis_num
#         N = 5
#
#         while index+N < L:
#             image_list = np.array(data[s]['images'])[:vis_num]
#             label_list = np.array(data[s]['labels'])[:vis_num]
#             pred_list = np.array(data[s]['bbpreds'])[:vis_num]
#             prob_list = np.array(data[s]['probs'])[:vis_num]
#             # uid_list = np.array(data[s]['uids'])[:vis_num]
#             # index_sort = np.argsort(uid_list)
#             # uid_list = np.sort(uid_list)
#             # image_list = [image_list[i] for i in index_sort]
#             # label_list = [label_list[i] for i in index_sort]
#             i = 0
#             M = 2 + label_list[0].shape[0]
#
#             plt.figure(figsize = (M*3,N*3))
#             for image_id in range(index, index + N):
#                 labels =label_list [image_id]
#                 image = image_list[image_id]
#                 pred = pred_list[image_id]
#                 prob = prob_list[image_id]
#                 i += 1
#                 plt.subplot(N,M,i)
#                 plt.imshow(image.transpose(1,2,0))
#                 plt.axis('off')
#                 i += 1
#                 plt.subplot(N,M,i)
#                 plt.imshow(pred)
#                 plt.axis('off')
#
#
#                 if len(labels.shape) == 2:
#                     labels = np.expand_dims(labels, 0)
#                 for label_idx in range(labels.shape[0]):
#                     # import ipdb; ipdb.set_trace()
#                     label = labels[label_idx,:,:]
#                     i += 1
#                     plt.subplot(N,M,i)
#                     plt.imshow(label)
#                     plt.title(prob[label_idx])
#                     plt.axis('off')
#             plt.savefig(os.path.join(vis_dir, s, str(image_id) + '.png'))
#             index = image_id
#
#     return

if __name__ == '__main__':
    data_root = '../data/cityscapes/processed/quarter'
    bbpred_root = '../data/cityscapes/bb_preds'
    preproc_folder = '../data/preproc'
    save_npy_dir =  '../data/cityscape_npy_5/'

    save_npy(load_and_maybe_process_data(data_root, preproc_folder, bbpred_root, True), save_npy_dir)

