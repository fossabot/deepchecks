from itertools import cycle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve, auc

from mlchecks.utils import get_plt_base64
from mlchecks import SingleDatasetBaseCheck, CheckResult, Dataset


def classifcation_preformace(ds: Dataset, model):
    label = ds._label
    res = dict()
    ds_x = ds[ds.features()]
    ds_y = ds[label]
    multi_y = (ds_y[:,None] == np.unique(ds_y)).astype(int)
    n_classes = ds_y.nunique()
    y_pred_prob = model.predict_proba(ds_x)
    y_pred = model.predict(ds_x)
    is_binary = ds_y.nunique() == 2
    
    if is_binary:
        auc = sklearn.metrics.roc_auc_score(ds_y, y_pred_prob[:,1])
    else:
        auc = sklearn.metrics.roc_auc_score(ds_y, y_pred_prob, multi_class='ovr')
        
    plt.cla()
    plt.clf()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(multi_y[:, i], y_pred_prob[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green', 'orange', 'yellow'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                label='ROC curve of class {0} (auc = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    roc_base64 = get_plt_base64()
    
    confusion_matrix = sklearn.metrics.confusion_matrix(ds_y, y_pred)
    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot()
    conf_base64 = get_plt_base64()
    
    
    macro_performance = pd.DataFrame(sklearn.metrics.precision_recall_fscore_support(ds_y, y_pred))
    macro_performance.index = ['precision', 'recall', 'f_score', 'support']
    
    res['macro_performance'] = macro_performance.to_dict()
    res['confusion_matrix'] = confusion_matrix
    res['auc'] = auc
    print(res)
    html = f'{macro_performance.to_html()}{roc_base64}{conf_base64}/>'
    return CheckResult(res, display={'text/html': html})

class PreformaceReport(SingleDatasetBaseCheck):
    def run(self, dataset=None, model=None) -> CheckResult:
        return classifcation_preformace(dataset, model)
