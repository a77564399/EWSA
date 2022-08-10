import numpy as np
# from pylab import *
from sklearn.metrics import *
from my_function import *


def null_list(num):
    lis = []
    for i in range(num): lis.append([])
    return lis


def equal_len_list(a):      # 按比例采样
    row_len = []
    for i in a:
        row_len.append(len(i))
    min_len = min(row_len)
    equal_len_A = []
    for i in a:
        tem_list = []
        multi = len(i)/min_len
        for j in range(min_len):
            tem_list.append(i[int(j*multi)])
        equal_len_A.append(tem_list)
    return equal_len_A


def model_evaluate(epo):
    f = 0
    # para = str(BATCH_SIZE) + str(ATT_SIZE) + str(FUN_SIZE)
    # DTI = np.loadtxt('../dataset/mat_drug_protein.txt')
    # y_pred = np.loadtxt('result/pre_fold_' + str(f) + '_' + str(para) + '.txt')[:,1]
    # test_index = np.loadtxt('result/idx_fold_' + str(f) + '_' + str(para) + '.txt')
    DTI = np.loadtxt('../../data/mat_drug_protein.txt')
    y_pred = np.loadtxt('../test_result/score_all1_' + str(epo) + '.txt')[:, 1]
    test_index = np.loadtxt('../test_result/idx_all1_' + str(epo) + '.txt')
    numD = 708
    numP = 1512

    pred = null_list(numD)
    true = null_list(numD)

    for i in range(test_index.shape[0]):
        d = int(test_index[i] / numP)
        p = int(test_index[i] % numP)
        pred[int(d)].append(y_pred[i])
        true[int(d)].append(DTI[d, p])

    auc_list = []
    aupr_list = []
    tpr = []
    fpr = []
    precision = []
    recall = []
    for i in tqdm(range(0, numD)):
        row_label = np.array(true[i])
        row_score = np.array(pred[i])
        if np.sum(row_label) == 0:
            continue
        else:
            tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(row_label, row_score)
            tpr.append(tpr1)
            fpr.append(fpr1)
            precision.append(precision1)
            recall.append(recall1)
            auc_list.append(auc(fpr1, tpr1))
            aupr_list.append(auc(recall1, precision1) + recall1[0] * precision1[0])
    e_TPR = equal_len_list(tpr)
    e_FPR = equal_len_list(fpr)
    e_P = equal_len_list(precision)
    e_R = equal_len_list(recall)
    mean_FPR = mean(e_FPR, axis=0)
    mean_TPR = mean(e_TPR, axis=0)
    mean_P = mean(e_P, axis=0)
    mean_R = mean(e_R, axis=0)

    np.savetxt('FPR.txt', np.array(mean_FPR))
    np.savetxt('TPR.txt', np.array(mean_TPR))
    np.savetxt('P.txt', np.array(mean_P))
    np.savetxt('R.txt', np.array(mean_R))

    # print('auc: ', auc(mean_FPR, mean_TPR))
    # print('aupr: ', auc(mean_R, mean_P) + mean_R[0] * mean_P[0])
    return auc(mean_FPR, mean_TPR), auc(mean_R, mean_P) + mean_R[0] * mean_P[0]

if __name__ == '__main__':
    # for i in range(89, 99, 10):
   print( model_evaluate(90)   )