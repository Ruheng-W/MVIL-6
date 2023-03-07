from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd
import numpy as np
# infile = sys.argv[1]
# outfile = sys.argv[2]
# # import pandas as pd


def caculate_metric(pred_y, labels, pred_prob):
    # print('labels', labels) # [n_sample, num_class]
    # print('pred_y', pred_y) # [n_sample, num_class]
    # print('pred_prob', pred_prob) # [n_sample, num_class]

    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # print('tp\tfp\ttn\tfn')
    # print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

    ACC = float(tp + tn) / test_num

    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # labels = labels.cpu()
    # pred_prob = pred_prob.cpu()
    labels = labels.tolist()
    pred_prob = pred_prob.tolist()

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 默认1就是阳性
    AUC = auc(fpr, tpr)

    return ACC, Sensitivity, Specificity, F1, AUC, MCC

data = pd.read_csv('test_stack.csv')

#fusion
x = 0.57
data.stack_sore = x*data.sore_1+(1-x)*data.sore_2
data.stack_lable = round(data.stack_sore)
pred_prob = np.array(data.stack_sore)
labels = np.array(data.real_label)
pred_y = np.array(data.stack_lable)

# singl
# pred_prob = np.array(data.sore_cnn)
# labels = np.array(data.real_label)
# pred_y = np.array(round(data.sore_cnn))

ACC, Sensitivity, Specificity, F1, AUC, MCC = caculate_metric(pred_y, labels, pred_prob)
print(ACC, Sensitivity, Specificity, F1, AUC, MCC)
