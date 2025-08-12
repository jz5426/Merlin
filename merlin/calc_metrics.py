import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score
from merlin.train_utils import PATHOLOGIES

def find_threshold(probabilities, true_labels):
    """
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        probabilities (numpy.ndarray): Predicted probabilities.
        true_labels (numpy.ndarray): True labels.

    Returns:
        float: Optimal threshold.
    """
    best_threshold = 0
    best_roc = 10000

    # Iterate over potential thresholds
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        confusion = confusion_matrix(true_labels, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        current_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if current_roc <= best_roc:
            best_roc = current_roc
            best_threshold = threshold

    return best_threshold

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, default='/cluster/home/t135419uhn/Merlin/shell_script/ctrate_zeroshot/prior_finetune_results.csv')
args = parser.parse_args()

# get the ground truth labels and the predicted labels
label_csv = pd.read_csv('/cluster/projects/mcintoshgroup/publicData/CT-RATE/dataset/multi_abnormality_labels/dataset_multi_abnormality_labels_train_predicted_labels.csv')
result = pd.read_csv(args.csv_file)

# filter only the prediction columns
columns = list(result[['_'.join(p.split(' '))+'_prob' for p in PATHOLOGIES]])
disease_mapper = {'_'.join(p.split(' '))+'_prob': p for p in PATHOLOGIES}

auc_scores_macro = {}
auc_scores_weighted = {}
auc_scores_micro = {}

spec_scores = {}
sens_scores = {}
acc_scores = {}

f1_scores_macro = {}
f1_scores_weighted = {}
f1_scores_micro = {}

prec_scores = {}

for column in columns:
    labels = []
    probs = []
    
    abnormality = disease_mapper[column]
    progress = tqdm(zip(result['VolumeName'], result[column]), desc=f"Column {column}", leave=False)
    for file_name, prob in progress:
        if np.isnan(prob): # very few cases
            prob = 0
        probs.append(prob)
        labels.append(label_csv[label_csv['VolumeName'] == file_name][abnormality].values[0])

    # auc per abnormality and then average
    auc_scores_macro[abnormality] = roc_auc_score(labels, probs, average='macro')
    auc_scores_weighted[abnormality] = roc_auc_score(labels, probs, average='weighted')
    auc_scores_micro[abnormality] = roc_auc_score(labels, probs, average='micro')

    # find the best threshold for each label and use that. instead of using 0.5
    threshold = find_threshold(np.array(probs), np.array(labels))
    
    pd_labels = (np.array(probs) > threshold).astype(int)
    gt_labels = np.array(labels)

    tp = ((pd_labels == 1) & (gt_labels == 1)).sum().item()
    fp = ((pd_labels == 1) & (gt_labels == 0)).sum().item()
    tn = ((pd_labels == 0) & (gt_labels == 0)).sum().item()
    fn = ((pd_labels == 0) & (gt_labels == 1)).sum().item()

    if tn + fp != 0:
        spec = tn / (tn + fp)
    else:
        spec = np.nan

    if tp + fn != 0:
        sens = tp / (tp + fn)
    else:
        sens = np.nan

    spec_scores[abnormality] = spec
    sens_scores[abnormality] = sens

    f1_macro = f1_score(gt_labels, pd_labels, average="macro")
    f1_weighted = f1_score(gt_labels, pd_labels, average="weighted")
    f1_micro = f1_score(gt_labels, pd_labels, average="micro")

    prec = precision_score(gt_labels, pd_labels)

    f1_scores_macro[abnormality] = f1_macro
    f1_scores_weighted[abnormality] = f1_weighted
    f1_scores_micro[abnormality] = f1_micro

    prec_scores[abnormality] = prec

    acc = (tp + tn) / (tp + tn + fp + fn)
    acc_scores[abnormality] = acc

print('\n\n')

print('Average AUC macro of Ours:', round(np.mean(list(auc_scores_macro.values())), 3))
print('Average AUC weighted of Ours:', round(np.mean(list(auc_scores_weighted.values())), 3))
print('Average AUC micro of Ours:', round(np.mean(list(auc_scores_micro.values())), 3))

print('Average F1 macro of Ours:', round(np.mean(list(f1_scores_macro.values())), 3))
print('Average F1 weighted of Ours:', round(np.mean(list(f1_scores_weighted.values())), 3))
print('Average F1 micro of Ours:', round(np.mean(list(f1_scores_micro.values())), 3))

print('Average ACC of Ours:', round(np.mean(list(acc_scores.values())), 3))

print('Average Spec of Ours:', round(np.mean(list(spec_scores.values())), 3))
print('Average Sens of Ours:', round(np.mean(list(sens_scores.values())), 3))
print('Average Prec of Ours:', round(np.mean(list(prec_scores.values())), 3))
