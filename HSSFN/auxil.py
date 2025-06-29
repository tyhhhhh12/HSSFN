import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from operator import truediv
import h5py
import matplotlib.pyplot as plt


def loadData(name, num_components=None):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'indian_pines.mat'))['indian_pines']
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
        class_name = [ "1", "2", "3", "4", "5",  "6", "7", "8", "9", "10","11","12","13","14","15","16"]
    elif name =='tree':
        f = h5py.File(os.path.join(data_path, 'data_band98.mat'))
        data = f['hyperspectral_data_98bands'][:]
        labels = sio.loadmat(os.path.join(data_path, 'SZUTreeData_R1_typeid_with_labels_5cm.mat'))['data']
        data = data.transpose(1,2,0)
        labels = labels.transpose(1,0)
        class_name = ["阿江榄仁", "吊瓜", "大王椰子", "大叶榕", "凤凰木",
                      "假槟榔", "荔枝", "芒果", "南洋杉", "蒲葵",
                      "人面xin", "相思", "小叶榕", "印度榕", "银合欢",
                      "羊蹄甲","白千层","黄葛","鸡蛋花","木麻黄","桃心花木",]
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
        class_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10","11", "12", "13"]
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas.mat'))['salinas']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        class_name = ["1", "2", "3", "4", "5","6", "7", "8", "9", "10","11",
                      "12", "13","14","15","16"]
    elif name == 'BA':
        data = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
        class_name = ["water", "Hippo grass", "Floodplain grasses 1", "Floodplain grasses 2", "Reeds",
                      "Riparian", "Firescar", "Island interior", "Acacia woodlands", "Acacia shrublands",
                      "Acacia grasslands", "short mopane", "Mixed mopane","dsadas"]
    elif name == 'Di':
        data = sio.loadmat(os.path.join(data_path, 'Dioni.mat'))['ori_data']
        labels = sio.loadmat(os.path.join(data_path, 'Dioni_gt_out68.mat'))['map']
        class_name = ["1", "2", "3", "4", "5",  "6", "7", "8", "9", "10","11","12"]
    elif name == 'H13':
        data = sio.loadmat(os.path.join(data_path, 'HustonU_IM.mat'))['hustonu']
        labels = sio.loadmat(os.path.join(data_path, 'HustonU_gt.mat'))['hustonu_gt']
        class_name = ["1", "2", "3", "4", "5",  "6", "7", "8", "9", "10","11","12","13","14","15"]
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
        class_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif name == "H18":
        data = sio.loadmat(os.path.join(data_path, 'houstonU2018.mat'))['houstonU']
        labels = sio.loadmat(os.path.join(data_path, 'houstonU2018.mat'))['houstonU_gt']
        class_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15","16","17","18","19","20"]
    elif name == "xuzhou":
        data = sio.loadmat(os.path.join(data_path, 'xuzhou.mat'))['xuzhou']
        labels = sio.loadmat(os.path.join(data_path, 'xuzhou_gt.mat'))['xuzhou_gt']
        class_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif name == "Longkou":
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
        class_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif name == "Honghu":
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']
        class_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                      "19", "20","21","22"]
    elif name == "Hanchuan":
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
        class_name = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
    else:
        print("NO DATASET")
        exit()


    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])


    if num_components != None:
        label = labels.reshape((labels.shape[0] * labels.shape[1]))
        data = umap.UMAP(n_components=num_components).fit_transform(data, label)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    data = StandardScaler().fit_transform(data)
    plot = plt.scatter(data[:,0],data[:,1],c=labels,cmap='tab20')
    plt.legend(handles=plot.legend_elements()[0],
               labels=["1", "2", "3", "4", "5",  "6", "7", "8", "9", "10","11","12","13"])


    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP Projection (30D)')
    plt.tight_layout()
    plt.show()
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels))
    return data, labels, num_class,class_name






def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes

        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1] #i=[x,y]
    return new_assign


def select_patch(matrix, pos_row, pos_col, ex_len):
    # 矩阵
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):

    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension),dtype='float32')

    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data

def generate_data(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, gt, batch_size=100):


    gt_all = gt[total_indices] -1
    y_train = gt[train_indices] -1
    y_test = gt[test_indices] -1



    all_data = []
    for i in range(0, len(total_indices), batch_size):
        batch_idx = total_indices[i:i + batch_size]
        batch_data = select_small_cubic(len(batch_idx), batch_idx, whole_data, PATCH_LENGTH, padded_data,
                                        INPUT_DIMENSION)
        all_data.append(batch_data)
    all_data = np.concatenate(all_data, axis=0)



    train_data = []
    for i in range(0, len(train_indices), batch_size):
        batch_idx = train_indices[i:i + batch_size]
        batch_data = select_small_cubic(len(batch_idx), batch_idx, whole_data, PATCH_LENGTH, padded_data,
                                        INPUT_DIMENSION)
        train_data.append(batch_data)
    train_data = np.concatenate(train_data, axis=0)


    test_data = []
    for i in range(0, len(test_indices), batch_size):
        batch_idx = test_indices[i:i+batch_size]
        batch_data = select_small_cubic(len(batch_idx), batch_idx, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
        test_data.append(batch_data)
    test_data = np.concatenate(test_data, axis=0)

    x_train = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION))
    x_test_all = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION))
    return all_data, x_train, x_test_all, gt_all, y_train, y_test

def accuracy(output, target, topk=(1,)):
    #计算top-k的准确率
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test, name):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))



def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')