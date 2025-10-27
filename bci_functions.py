import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', show=False, cmap=plt.cm.Blues,
                          save_path=None):
    """
    - cm: Computed values of the confusion matrix
    - classes: Labels corresponding to each row and column of the confusion matrix
    - normalize: True to display percentages, False to display raw counts
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    plt.figure(figsize=(18, 15), dpi=200)
    plt.rcParams.update({'font.size': 30})
    plt.rc('font', family='Times New Roman')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=50)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=30,
                 horizontalalignment="right",
                 verticalalignment="baseline",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=50)
    plt.xlabel('Predicted label', fontsize=50)
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()


def mkdir(path):
    import os
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' has been created successfully.')
    else:
        print(path + ' already exists.')


def to_onehot(label_array, num_classes):
    label_onehot = np.eye(num_classes)[label_array]
    return label_onehot

def dataset_loader(dataset_name: str, subject_id: list | None, mode: str = 'cross-session', l_freq: int = 1, high_freq: int = 48, norm: bool = True):
    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subject_id)

    # Preprocessing
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(lambda x: x * 1e6),
        Preprocessor('filter', l_freq=l_freq, h_freq=high_freq)
    ]
    if norm:
        preprocessors.append(
            Preprocessor(lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True))
        )
    preprocess(dataset, preprocessors)

    def get_unified_event_mapping(dataset):
        all_events = set()
        for ds in dataset.datasets:
            all_events.update(ds.raw.annotations.description)
        return {desc: i for i, desc in enumerate(sorted(all_events))}

    windows_dataset = create_windows_from_events(
        concat_ds=dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
        mapping=get_unified_event_mapping(dataset)
    )

    # trial-level description
    desc_list = []
    for ds in windows_dataset.datasets:
        desc_list.extend([ds.description] * len(ds))
    desc_df = pd.DataFrame(desc_list).reset_index(drop=True)

    X = np.array([x for x, y, _ in windows_dataset])
    Y = np.array([y for x, y, _ in windows_dataset])

    Config = {
        'sfreq': windows_dataset.datasets[0].raw.info['sfreq'],
        'n_times': X.shape[-1],
        'n_channels': X.shape[1],
        'n_classes': len(np.unique(Y)),
        'input_shape': (X.shape[1], X.shape[-1]),
        'target': 'Target' in dataset.datasets[0].raw.annotations.description
    }

    if mode == 'cross-session':
        def refine_split(session_list, run_list, X):
            session_list = np.array(session_list)
            run_list = np.array(run_list)
            unique_sessions = list(dict.fromkeys(session_list))

            if len(unique_sessions) > 1:
                train_sessions = [s for s in unique_sessions if 'train' in s.lower()]
                test_sessions = [s for s in unique_sessions if 'test' in s.lower()]
                if train_sessions and test_sessions:
                    train_idx = np.isin(session_list, train_sessions)
                    test_idx = np.isin(session_list, test_sessions)
                else:
                    train_sessions = unique_sessions[:-1] if len(unique_sessions) > 2 else [unique_sessions[0]]
                    test_session = unique_sessions[-1] if len(unique_sessions) > 2 else unique_sessions[1]
                    train_idx = np.isin(session_list, train_sessions)
                    test_idx = session_list == test_session
            else:
                train_mask = np.array(['train' in r.lower() for r in run_list])
                test_mask = np.array(['test' in r.lower() for r in run_list])
                if train_mask.any() and test_mask.any():
                    train_idx = train_mask
                    test_idx = test_mask
                else:
                    unique_runs = list(dict.fromkeys(run_list))
                    if 2 <= len(unique_runs) <= 3:
                        test_run = unique_runs[-1]
                        train_runs = unique_runs[:-1]
                        train_idx = np.isin(run_list, train_runs)
                        test_idx = run_list == test_run
                    else:
                        n = len(X)
                        split = n // 2
                        train_idx = np.zeros(n, dtype=bool)
                        test_idx = np.zeros(n, dtype=bool)
                        train_idx[:split] = True
                        test_idx[split:] = True

            return train_idx, test_idx

        session_list = desc_df['session'].values
        run_list = desc_df['run'].values.astype(str)
        train_idx, test_idx = refine_split(session_list, run_list, X)

        return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx], Config

    return X, Y, Config


if __name__ == '__main__':
    dataset_list = [
        'BNCI2014_001',
        # 'BNCI2014_002',
        # 'Zhou2016',
    ]
    for dataset in dataset_list:
        print(f'loading {dataset}...')
        # subject_ids=[num] for num-th subject data, subjects=None for all subject data
        subject_ids = [1]
        X_train, X_test, Y_train, Y_test, Config = dataset_loader(dataset, subject_id=subject_ids, mode='cross-session')
        print("shape:", X_train.shape, X_test.shape)
