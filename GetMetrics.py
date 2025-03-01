import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)

def get_f_beta_score(precision, recall, beta):
    """Returns the F_{`beta`}-score for the provided `precision` and `recall`.

    The F-score is defined as zero if both Precision and Recall are zero.
    """
    beta_squared = beta**2
    try:
        f_score = np.nan_to_num(
            (1 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall)
        )
        return f_score
    except (
        ZeroDivisionError
    ):  # only thrown if `precision` and `recall` are not ndarrays
        return 0.0
    
def precision_recall_f1_pertype(
	labels, preds
):
	"""Point-based (faster) implementation."""
	ano_labels = [l for l in np.unique(labels) if l > 0]
	precision, recall_dict, f_score_dict, thresholds = None, dict(), dict(), None
	for k in ["mixed"] + ano_labels:
		labels_mask = labels > 0 if k == "mixed" else labels == k
		masked_labels = labels_mask.astype(np.int8)
		if k == "mixed":
			precision, recall_dict["mixed"], f_score_dict["mixed"], _ = precision_recall_fscore_support(
				masked_labels, preds, average='binary'
			)
		else:
			_, recall_dict[k], _, _ = precision_recall_fscore_support(
				masked_labels, preds, average='binary'
			)
			f_score_dict[k] = get_f_beta_score(
				precision, recall_dict[k], 1
			)
	if len(ano_labels) > 0:
		recall_dict["avg"] = sum([recall_dict[k] for k in ano_labels]) / len(
			ano_labels
		)
		f_score_dict["avg"] = get_f_beta_score(
			precision, recall_dict["avg"], 1
		)
	return (
				f_score_dict,
				precision,
				recall_dict
			)

def ap_at_k(test_scores, y_test, k):
    # Pair each score with its label and sort by score in descending order
    paired_scores_labels = list(zip(test_scores, y_test))
    paired_scores_labels.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate precision at each position in the ranked list up to k
    precisions = []
    relevant_found = 0
    for i in range(k):
        if paired_scores_labels[i][1] == 1:  # If the item is relevant (an anomaly)
            relevant_found += 1
        precisions.append(relevant_found / (i + 1))
    
    # Calculate the average precision at k
    apk = sum(precisions) / k
    
    return apk, precisions

def ar_at_k(test_scores, y_test, k):
    NumAnom = sum(y_test)
    # Pair each score with its label and sort by score in descending order
    paired_scores_labels = list(zip(test_scores, y_test))
    paired_scores_labels.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate precision at each position in the ranked list up to k
    recalls = []
    relevant_found = 0
    for i in range(k):
        if paired_scores_labels[i][1] == 1:  # If the item is relevant (an anomaly)
            relevant_found += 1
        recalls.append(relevant_found / NumAnom)
    
    # Calculate the average precision at k
    ark = sum(recalls) / k
    
    return ark, recalls

def get_ap_at_perType(test_scores, y_test, k, Types=[2, 3, 4]):
	apks = []
	arks = []
	for T in Types:
		y_test_T = (y_test == T).astype(int)
		apk_T, precisions_T = ap_at_k(test_scores, y_test_T, 200)
		ark_T, recalls_T = ar_at_k(test_scores, y_test_T, 200)
		apks.append(apk_T)
		arks.append(ark_T)
	return apks, arks