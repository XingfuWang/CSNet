# This code is the official code repository for the paper "Cholesky Space for Brain–Computer Interfaces".

import itertools
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from bci_functions import plot_confusion_matrix, mkdir, to_onehot, dataset_loader
from CSNet import CSNet, CSNet_ST, CSNet_woCh, CSNet_MST, CSNet_SMT

warnings.filterwarnings("ignore")

# =======================
# Dataset and Model Setup
# =======================

# List of EEG datasets to be evaluated.
# Uncomment the datasets you wish to include in the current run.
dataset_list = [
    'BNCI2014_001',  # Dataset 1 (BCI IV 2a): 4-class MI — ['feet', 'left_hand', 'right_hand', 'tongue']
]

# List of models to be trained and evaluated.
model_list = [
    # -------------------- Proposed Method --------------------
    'CSNet',  # Our proposed model based on Cholesky space EEG representation

    # ---------------- Ablation Studies (see paper for details) ----------------
    'CSNet_ST',
    'CSNet_woCh',
    'CSNet_MST',
    'CSNet_SMT',
]

# Set the number of epochs for training.
EPOCH = 100  # or 200
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')
for dataset_name, model_name in itertools.product(dataset_list, model_list):
    folder_name = f'Cross-Ses-{model_name}(norm)_{dataset_name}'
    postfix = f'{folder_name}_epoch{EPOCH}'

    for subject_num in np.arange(1, 10):  # num of subject
        best_acc_save = []
        y_pred_all = []
        y_true_all = []
        save_path = f'output_folder/{folder_name}/Subject_{subject_num}_{postfix}'
        try:
            # subject_ids=[num] for num-th subject data, subjects=None for all subject data
            train_data, test_data, train_label, test_label, config = dataset_loader(dataset_name=dataset_name, subject_id=[int(subject_num)])
        except:
            continue
        mkdir(save_path)
        sfreq, n_times, n_chans, n_outputs, target = config['sfreq'], config['n_times'], config['n_channels'], config['n_classes'], config['target']

        train_label = to_onehot(train_label.astype(int), num_classes=n_outputs)
        test_label = to_onehot(test_label.astype(int), num_classes=n_outputs)

        print('train_data shape:', train_data.shape, 'train_label shape:', train_label.shape)
        print('test_data shape:', test_data.shape, 'test_label shape:', test_label.shape)
        k_f = 1

        if model_name == 'CSNet':
            model = CSNet(n_chans=n_chans, n_class=n_outputs)
        elif model_name == 'CSNet_woCh':
            model = CSNet_woCh(n_chans=n_chans, n_class=n_outputs)
        elif model_name == 'CSNet_MST':
            model = CSNet_MST(n_chans=n_chans, n_class=n_outputs)
        elif model_name == 'CSNet_SMT':
            model = CSNet_SMT(n_chans=n_chans, n_class=n_outputs)
        elif model_name == 'CSNet_ST':
            model = CSNet_ST(n_chans=n_chans, n_class=n_outputs)
        else:
            raise ValueError(f"Unknown model: {folder_name}")
        model.to(device)

        ratio = np.sum(train_label, axis=0) / len(train_label)
        class_weights = 1.0 / ratio
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float())

        loss_fn.to(device)
        learning_rate = 5e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        print('Data loaded successfully!')

        # Dataloader
        batch_size = 64
        train_data = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_data = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
        train_label = DataLoader(dataset=train_label, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_label = DataLoader(dataset=test_label, batch_size=batch_size, shuffle=False, pin_memory=True)
        # Recorder
        train_step = 0
        test_step = 0
        test_accuracy_save = []
        test_loss_save = []
        train_accuracy_save = []
        train_loss_save = []
        best_acc = 0
        cost_time = 0
        for epoch in range(EPOCH):
            if epoch % 10 == 0:
                print(f'------------- {folder_name}  Subject{subject_num}  Epoch {epoch} ------------')
            # Train programs starts
            model.train()
            train_loss = 0
            train_corr = 0
            train_count = 0
            outputs = []
            labels = []
            for item_data, item_label in zip(train_data, train_label):
                data_, label_ = item_data.float().to(device), item_label.to(device)
                start_time = time.time()  # start of training
                output = model(data_)
                loss = loss_fn(output, label_)
                train_loss = train_loss + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cost_time = time.time() - start_time + cost_time  # end of training
                train_step = train_step + 1
                if train_step % 500 == 0:
                    print('On {} train, Loss: {:.3f}'.format(train_step, loss.item()))
                outputs.extend(output.argmax(1).tolist())
                labels.extend(label_.argmax(1).tolist())
            if target and n_outputs == 2:  # use AUC for P300
                try:
                    train_accuracy = roc_auc_score(labels, outputs)
                except:
                    train_accuracy = 0.0
            else:
                train_count = len(labels)
                train_corr = sum([1 for o, l in zip(outputs, labels) if o == l])
                train_accuracy = (train_corr / train_count)
            if epoch % 10 == 0:
                print('Loss on train: {:.3f}'.format(train_loss))
                print('Acc on train: {:.3f}'.format(train_accuracy))
            train_accuracy_save.append(train_accuracy)
            train_loss_save.append(train_loss)

            # Test program starts
            model.eval()
            test_loss = 0
            test_corr = 0
            outputs_list = []
            labels_list = []

            with torch.no_grad():
                for item_data, item_label in zip(test_data, test_label):
                    data_, label_ = item_data.float().to(device), item_label.to(device)
                    output = model(data_)

                    test_loss += loss_fn(output, label_).item()

                    pred_labels = output.argmax(dim=1)
                    true_labels = label_.argmax(dim=1)

                    test_corr += (pred_labels == true_labels).sum().item()

                    outputs_list.append(pred_labels.cpu())
                    labels_list.append(true_labels.cpu())

            y_pred = torch.cat(outputs_list)
            y_true = torch.cat(labels_list)

            if target and n_outputs == 2:  # use AUC for P300
                try:
                    test_accuracy = roc_auc_score(y_true.numpy(), y_pred.numpy())
                except:
                    test_accuracy = 0.0
            else:
                test_accuracy = test_corr / len(y_pred)

            test_accuracy_save.append(test_accuracy)
            test_loss_save.append(test_loss)

            if epoch % 10 == 0:
                print(f'Loss on test: {test_loss:.3f}')
                print(f'Acc on test: {test_accuracy:.3f}')

            if test_accuracy >= best_acc:
                y_pred_best = y_pred
                y_true_best = y_true
                best_acc = test_accuracy
                print('\033[91mBest_acc updated to {:.3f}\033[0m'.format(best_acc))
                cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
                plot_confusion_matrix(cm=cm, classes=['feet', 'left', 'right', 'tongue'],
                                      save_path=f'{save_path}/Confusion_matrix{k_f}.jpg')
                plt.close()
                torch.save(model, f'{save_path}/best_acc_model_k{k_f}.pth')

            if int(test_accuracy) == 1:
                break
        print('Time cost: %.2f' % cost_time)
        np.savez(f'{save_path}/Time{cost_time:.4f}S.npz', time=cost_time)
        y_pred_all += y_pred_best
        y_true_all += y_true_best
        plt.figure(figsize=(10, 10), dpi=200)
        plt.rc('font', family='Times New Roman')
        plt.rcParams.update({'font.size': 20})
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(test_accuracy_save)), test_accuracy_save, label='Test Accuracy')
        plt.plot(np.arange(len(train_accuracy_save)), train_accuracy_save, label='Train Accuracy')
        plt.legend(loc='lower right')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(test_loss_save)), test_loss_save, label='Test loss')
        plt.plot(np.arange(len(train_loss_save)), train_loss_save, label='Train loss')
        plt.legend(loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'{save_path}/Accuracy_Loss_K{k_f}.jpg')
        plt.close()
        np.savetxt(f'{save_path}/k{k_f}_acc{best_acc:.4f}.txt', [best_acc])
        np.savez(f'{save_path}/history_data_k{k_f}.npz',
                 train_accuracy=train_accuracy_save, val_accuracy=test_accuracy_save,
                 train_loss=train_loss_save, val_loss=test_loss_save,
                 true_data=y_true_best, pred_data=y_pred_best)
