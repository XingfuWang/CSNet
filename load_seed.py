import os
import pickle
import sys
import math
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from scipy.io import loadmat


def one_session_segment(file, name, seg_len):
    All_data = loadmat(file)
    all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    segments = np.empty([0, 62, seg_len])
    labels = np.array([])
    for trial_idx in range(15):  # one session has 15 trials
        one_trial = All_data[name + '_eeg' + str(trial_idx + 1)]  # (62, 47001) =>
        num_segment = int(len(one_trial[0]) / seg_len)  # 235
        print(f'trial_{trial_idx + 1}: {len(one_trial[0])} samples, {num_segment} segments')

        for i in range(num_segment):
            start_idx = i * seg_len
            end_idx = (i + 1) * seg_len
            segments = np.vstack([segments, np.expand_dims(one_trial[:, start_idx:end_idx], axis=0)])
            labels = np.append(labels, [all_label[trial_idx]])

    return segments, labels


filename_list = ['1_20131027.mat', '1_20131030.mat', '1_20131107.mat', '2_20140404.mat', '2_20140413.mat', '2_20140419.mat',
                 '3_20140603.mat', '3_20140611.mat', '3_20140629.mat', '4_20140621.mat', '4_20140702.mat', '4_20140705.mat',
                 '5_20140411.mat', '5_20140418.mat', '5_20140506.mat', '6_20130712.mat', '6_20131016.mat', '6_20131113.mat',
                 '7_20131027.mat', '7_20131030.mat', '7_20131106.mat', '8_20140511.mat', '8_20140514.mat', '8_20140521.mat',
                 '9_20140620.mat', '9_20140627.mat', '9_20140704.mat', '10_20131130.mat', '10_20131204.mat', '10_20131211.mat',
                 '11_20140618.mat', '11_20140625.mat', '11_20140630.mat', '12_20131127.mat', '12_20131201.mat', '12_20131207.mat',
                 '13_20140527.mat', '13_20140603.mat', '13_20140610.mat', '14_20140601.mat', '14_20140615.mat', '14_20140627.mat',
                 '15_20130709.mat', '15_20131016.mat', '15_20131105.mat']

short_name = ['djc', 'djc', 'djc', 'jl', 'jl', 'jl', 'jj', 'jj', 'jj',
              'lqj', 'lqj', 'lqj', 'ly', 'ly', 'ly', 'mhw', 'mhw', 'mhw',
              'phl', 'phl', 'phl', 'sxy', 'sxy', 'sxy', 'wk', 'wk', 'wk',
              'ww', 'ww', 'ww', 'wsf', 'wsf', 'wsf', 'wyw', 'wyw', 'wyw',
              'xyl', 'xyl', 'xyl', 'ys', 'ys', 'ys', 'zjy', 'zjy', 'zjy']

if __name__ == '__main__':
    data_path = "F:/Dataset/Adaptive_Decoding/Emotion Recognition/Raw_data/" # "your_result_path/"
    segment_len = 1000  # used in CSNet
    result_dir = f"F:/Dataset/Adaptive_Decoding/Emotion Recognition/Demo/"

    combined_segments = []
    combined_labels = []

    for idx in range(len(filename_list)):  # 15 Sub × 3 session
        file_path = data_path + filename_list[idx]
        print(f"------------------------loading file:{file_path}------------------------")

        segments, labels= one_session_segment(file=file_path, name=short_name[idx], seg_len=segment_len)

        # add to combined lists
        combined_segments.append(segments)
        combined_labels.append(labels)

        # three sessions done, save subject data
        if (idx + 1) % 3 == 0:
            subject = idx // 3 + 1
            os.makedirs(result_dir + f'sub{subject}_train', exist_ok=True)
            os.makedirs(result_dir + f'sub{subject}_test', exist_ok=True)

            # concat all sessions' data for the subject
            subject_train_segments = np.concatenate(combined_segments, axis=0)
            subject_train_labels = np.concatenate(combined_labels, axis=0)

            # save data
            segment_len_code = f"{segment_len // 20:04d}"  # 0050 for 5s duration (1000segment_len/200Hz)
            np.savez(result_dir + f'sub{subject}_train/Data{segment_len_code}.npz', x_data=subject_train_segments, y_data=subject_train_labels)
            print(
                f"sub{subject} train data:{subject_train_segments.shape},train label:{subject_train_labels.shape}.")

            # clear for next session
            combined_train_segments = []
            combined_train_labels = []
