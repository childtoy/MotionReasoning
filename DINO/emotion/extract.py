import re, os, ntpath
import numpy as np
from . import utils
import scipy.io

# def get_emotion_set(data_dir, window=40, offset=10, train=True, stats=False, is_normalize = True):
#     """
#     Extract the same test set as in the article, given the location of the post rigging mat files.

#     :param data_dir: Path to the dataset files(mat file)
#     :param window: width  of the sliding windows (in timesteps)
#     :param offset: offset between windows (in timesteps)
#     :return: tuple:
#         X: local positions
#         Q: local quaternions
#         Q_vel : velocity of local q
#         Q_acc : acceleration of local q
#         seq_name : label emotion 
#         scene : label scene
#         subject : label subject
#         repeat : label repeatation 
#     """
#     subjects = []
#     seq_names = []
#     scenes = []
#     repeats = []
#     X = []
#     Q = []
#     Q_vel = []
#     Q_acc = []
#     labels = []
#     # Extract
#     mat_files = sorted(os.listdir(data_dir))
#     print(mat_files)
#     for file in mat_files:
#         if file.endswith(".mat"):
#             file_info = file.split('_')
#             subject = file_info[1]
#             seq_name = file_info[2]
#             if seq_name == 'angry':
#                 label = 0
#             elif seq_name == 'disgust':
#                 label = 1
#             elif seq_name == 'fearful':
#                 label = 2
#             elif seq_name == 'happy':
#                 label = 3
#             elif seq_name == 'neutral':
#                 label = 4
#             elif seq_name == 'sad':
#                 label = 5
#             elif seq_name == 'surprise':
#                 label = 6
#             else : 
#                 assert('No matched Label')
#             scene = file_info[3]
#             repeat = file_info[4].split('.')[0]
#             if train == True and int(repeat) == 2 : 
#                 continue 
#             elif train == False and int(repeat) != 2 : 
#                 continue
#             seq_path = os.path.join(data_dir, file)
#             data = scipy.io.loadmat(seq_path)
#             q_base = np.concatenate([data['q_revs_post'], np.zeros([1,data['q_revs_post'].shape[1]])],axis=0)
#             q_vel = q_base[1:, :] - q_base[:-1,:]
#             q_vel_base = np.concatenate([np.zeros([1,q_vel.shape[1]]), q_vel], axis=0)
#             q_acc = q_vel_base[1:, :] - q_vel_base[:-1, :]
            
#             # Sliding windows
#             i = 0
#             window=window # 2 sec
#             offset = 10 # 0.5sec
#             print(data['T_roots_post'].shape[1])
#             while i + window < data['T_roots_post'].shape[1]:
#                 Q.append(data['q_revs_post'][i:i+window,:])
#                 Q_vel.append(q_vel[i:i+window, :])
#                 Q_acc.append(q_acc[i:i+window, :])
#                 X.append(data['T_roots_post'][:,i:i+window])                
#                 subjects.append(subject)
#                 seq_names.append(seq_name)
#                 labels.append(label)
#                 scenes.append(scene)
#                 repeats.append(repeat)
#                 i += offset
#     X = np.asarray(X)
#     Q = np.asarray(Q)
#     Q_vel = np.asarray(Q_vel)
#     Q_acc = np.asarray(Q_acc)
#     subjects = np.asarray(subjects)
#     seq_names = np.asarray(seq_names)
#     labels = np.asarray(labels)
#     scenes = np.asarray(scenes)
#     repeats = np.asarray(repeats) 

#     if is_normalize : 
#         mean_q = np.asarray(Q).mean(axis=(0,1))
#         std_q = np.std(np.asarray(Q), axis=(0,1))            
#         Q = (Q - mean_q)/std_q
#         mean_vel = np.asarray(Q_vel).mean(axis=(0,1))
#         std_vel = np.std(np.asarray(Q_vel), axis=(0,1))
#         Q_vel = (Q_vel - mean_vel)/std_vel
#         mean_acc = np.asarray(Q_acc).mean(axis=(0,1))
#         std_acc = np.std(np.asarray(Q_acc), axis=(0,1))
#         Q_acc = (Q_acc - mean_acc)/std_acc

#     return X, Q, Q_vel, Q_acc, seq_names, labels, subjects, scenes, repeats
def get_emotion_set(data_dir, window=40, offset=10, train=True, stats=False, is_normalize = True):
    """
    Extract the same test set as in the article, given the location of the post rigging mat files.

    :param data_dir: Path to the dataset files(mat file)
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    :return: tuple:
        X: local positions
        Q: local quaternions
        Q_vel : velocity of local q
        Q_acc : acceleration of local q
        seq_name : label emotion 
        scene : label scene
        subject : label subject
        repeat : label repeatation 
    """
    subjects = []
    seq_names = []
    scenes = []
    repeats = []
    X = []
    Q = []
    Q_vel = []
    Q_acc = []
    labels = []
    start_frame = []
    file_name = []
    # Extract
    mat_files = sorted(os.listdir(data_dir))
    print(mat_files)
    for file in mat_files:
        if file.endswith(".mat"):
            file_info = file.split('_')
            subject = file_info[1]
            seq_name = file_info[2]
            if seq_name == 'angry':
                label = 0
            elif seq_name == 'disgust':
                label = 1
            elif seq_name == 'fearful':
                label = 2
            elif seq_name == 'happy':
                label = 3
            elif seq_name == 'neutral':
                label = 4
            elif seq_name == 'sad':
                label = 5
            elif seq_name == 'surprise':
                label = 6
            else : 
                assert('No matched Label')
            # if label == 0  or label == 1:
            #     print('a')
            # else : 
            #     continue
            scene = file_info[3]
            repeat = file_info[4].split('.')[0]
            if train == True and int(repeat) == 2 : 
                continue 
            elif train == False and int(repeat) != 2 : 
                continue
            seq_path = os.path.join(data_dir, file)
            data = scipy.io.loadmat(seq_path)
            q_base = np.concatenate([data['q_revs_post'], np.zeros([1,data['q_revs_post'].shape[1]])],axis=0)
            q_vel = q_base[1:, :] - q_base[:-1,:]
            q_vel_base = np.concatenate([np.zeros([1,q_vel.shape[1]]), q_vel], axis=0)
            q_acc = q_vel_base[1:, :] - q_vel_base[:-1, :]
            
            # Sliding windows
            i = 0
            window=window # 2 sec
            offset = 10 # 0.5sec
            while i + window < data['T_roots_post'].shape[1]:
                Q.append(data['q_revs_post'][i:i+window,:])
                Q_vel.append(q_vel[i:i+window, :])
                Q_acc.append(q_acc[i:i+window, :])
                X.append(data['T_roots_post'][:,i:i+window])                
                subjects.append(subject)
                seq_names.append(seq_name)
                labels.append(label)
                scenes.append(scene)
                repeats.append(repeat)
                i += offset
    X = np.asarray(X)
    Q = np.asarray(Q)
    Q_vel = np.asarray(Q_vel)
    Q_acc = np.asarray(Q_acc)
    subjects = np.asarray(subjects)
    seq_names = np.asarray(seq_names)
    labels = np.asarray(labels)
    scenes = np.asarray(scenes)
    repeats = np.asarray(repeats) 

    if is_normalize : 
        mean_q = np.asarray(Q).mean(axis=(0,1))
        std_q = np.std(np.asarray(Q), axis=(0,1))            
        Q = (Q - mean_q)/std_q
        mean_vel = np.asarray(Q_vel).mean(axis=(0,1))
        std_vel = np.std(np.asarray(Q_vel), axis=(0,1))
        Q_vel = (Q_vel - mean_vel)/std_vel
        mean_acc = np.asarray(Q_acc).mean(axis=(0,1))
        std_acc = np.std(np.asarray(Q_acc), axis=(0,1))
        Q_acc = (Q_acc - mean_acc)/std_acc

    return X, Q, Q_vel, Q_acc, seq_names, labels, subjects, scenes, repeats, start_frame, file_name


# def get_train_stats(bvh_folder, train_set):
#     """
#     Extract the same training set as in the paper in order to compute the normalizing statistics
#     :return: Tuple of (local position mean vector, local position standard deviation vector, local joint offsets tensor)
#     """
#     print("Building the train set...")
#     xtrain, qtrain, parents, _, _, _ = get_lafan1_set(
#         bvh_folder, train_set, window=50, offset=20, train=True, stats=True
#     )

#     print("Computing stats...\n")
#     # Joint offsets : are constant, so just take the first frame:
#     offsets = xtrain[0:1, 0:1, 1:, :]  # Shape : (1, 1, J, 3)

#     # Global representation:
#     q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)

#     # Global positions stats:
#     x_mean = np.mean(
#         x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]),
#         axis=(0, 2),
#         keepdims=True,
#     )
#     x_std = np.std(
#         x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]),
#         axis=(0, 2),
#         keepdims=True,
#     )

#     return x_mean, x_std, offsets
