import re, os, ntpath
import numpy as np
from . import utils
channelmap = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}

channelmap_inv = {
    "x": "Xrotation",
    "y": "Yrotation",
    "z": "Zrotation",
}

ordermap = {
    "x": 0,
    "y": 1,
    "z": 2,
}


class Anim(object):
    """
    A very basic animation object
    """

    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones


def read_bvh(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line:
            continue
        if "MOTION" in line:
            continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = len(parents) - 1
            continue

        if "{" in line:
            continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(
            r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line
        )
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis : 2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = len(parents) - 1
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(" ")

        if dmatch:
            try:
                data_block = np.array(list(map(float, dmatch)))
                N = len(parents)
                fi = i - start if start else i
                if channels == 3:
                    positions[fi, 0:1] = data_block[0:3]
                    rotations[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    positions[fi, :] = data_block[:, 0:3]
                    rotations[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1
            except:
                print('aa')
    f.close()

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)


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
    # Extract
    subjects = sorted(os.listdir(data_dir))
    bvh_files = []
    bvh_names = []
    for i in subjects[1:]:
        for j in os.listdir(os.path.join(data_dir,i)):
            bvh_names.append(j)
    bvh_names.sort()
    for file in bvh_names:
        if file.endswith(".bvh"):
            subject = file[:3]
            seq_name = file
            if 'SA' in seq_name :
                label = 0
                scene = seq_name.split('SA')[1][0]
                repeat = seq_name.split('V')[1][0]
            elif 'A' in seq_name :
                label = 4
                scene = seq_name.split('A')[1][0]
                repeat = seq_name.split('V')[1][0]                
            elif 'D' in seq_name :
                label = 3
                scene = seq_name.split('D')[1][0]
                repeat = seq_name.split('V')[1][0]                
            elif 'F' in seq_name :
                label = 2
                scene = seq_name.split('F')[1][0]
                repeat = seq_name.split('V')[1][0]                
            elif 'H' in seq_name :
                label = 1
                scene = seq_name.split('H')[1][0]
                repeat = seq_name.split('V')[1][0]                
            elif 'N' in seq_name :
                label = 5
                scene = seq_name.split('N')[1][0]
                repeat = seq_name.split('V')[1][0]
            elif 'SU' in seq_name :
                label = 6
                scene = seq_name.split('SU')[1][0]
                repeat = seq_name.split('V')[1][0]
            else : 
                assert('No matched Label')
            if label == 0  or label == 1:
                print('a')
            else : 
                continue
            repeat = file.split('V')[1][0]
            if train == True and int(repeat) == 2 : 
                continue 
            elif train == False and int(repeat) != 2 : 
                continue
            chosen_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 36, 37, 38, 39]
            print(os.path.join(data_dir,subject,file))
            anim = read_bvh(os.path.join(data_dir,subject,file))
            print(anim.quats.shape)
            q_base = np.concatenate([anim.quats[:,chosen_joints], np.zeros([1,len(chosen_joints),4])],axis=0)
            q_vel = q_base[1:, :] - q_base[:-1,:]
            print(q_vel.shape)
            q_vel_base = np.concatenate([np.zeros([1,q_vel.shape[1],4]), q_vel], axis=0)
            q_acc = q_vel_base[1:, :] - q_vel_base[:-1, :]
            
            # Sliding windows
            i = 0
            window=window # 2 sec
            offset = 10 # 0.5sec
            while 6*i + window < anim.pos.shape[0]:
                Q.append(anim.quats[6*i:6*i+window,chosen_joints])
                Q_vel.append(q_vel[6*i:6*i+window])
                Q_acc.append(q_acc[6*i:6*i+window])
                X.append(anim.pos[6*i : 6*i + window])                
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

    return X, Q, Q_vel, Q_acc, seq_names, labels, subjects, scenes, repeats


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
