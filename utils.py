import numpy as np
import json
import sklearn.metrics as sk_metrics

def shuffle(f, seed=42):
    indexes = np.arange(len(f))
    np.random.RandomState(seed).shuffle(indexes)
    return np.array(f)[indexes]

def load_dataset(dataset, noise_ratio = 0, num_calib_pos=0, num_calib_neg=0, seed=42, pos_limit=10000, neg_limit=10000):
    max_noise_ratio = 1.0
    max_num_calib_pos = 100
    max_num_calib_neg = 100

    with open('datasets/%s.json'%dataset) as f:
        image_data = json.load(f)

    pos_train_images = np.array(image_data['pos_train'])
    pos_train_images = shuffle(pos_train_images, seed=seed)

    pos_test_images = np.array(image_data['pos_test'])
    pos_test_images = shuffle(pos_test_images, seed=seed)

    neg_train_images = np.array(image_data['neg_train'])
    neg_train_images = shuffle(neg_train_images, seed=seed)

    neg_test_images = np.array(image_data['neg_test'])
    neg_test_images = shuffle(neg_test_images, seed=seed)

    pos_calib_images = pos_train_images[:num_calib_pos]
    pos_train_images = pos_train_images[max_num_calib_pos:]

    neg_calib_images = neg_train_images[:num_calib_neg]
    neg_train_images = neg_train_images[max_num_calib_neg:]

    min_num_neg = 200

    num_neg_to_transfer = min(int(len(pos_train_images) * max_noise_ratio), len(neg_train_images)-min_num_neg)
    num_pos = int(num_neg_to_transfer/max_noise_ratio)

    noise_images = neg_train_images[:num_neg_to_transfer]
    pos_train_images = pos_train_images[:num_pos]
    neg_train_images = neg_train_images[num_neg_to_transfer:]

    if noise_ratio > 0.0:
        num_noise = int(len(pos_train_images)*noise_ratio)
        num_pos = len(pos_train_images) - num_noise

        pos_train_images = np.concatenate([noise_images[:num_noise], pos_train_images[:num_pos]])


    pos_train_images = shuffle(pos_train_images, seed=seed)
    neg_train_images = shuffle(neg_train_images, seed=seed)

    pos_train_images = pos_train_images[:pos_limit]
    neg_train_images = neg_train_images[:neg_limit]
    
    test_images = np.concatenate([pos_test_images, neg_test_images])
    test_labels = [1]*len(pos_test_images) + [0]*len(neg_test_images)
    
    return {
        'pos_train_images': pos_train_images,
        'neg_train_images': neg_train_images,
        'pos_test_images': pos_test_images,
        'neg_test_images': neg_test_images,
        
        'pos_calib_images': pos_calib_images,
        'neg_calib_images': neg_calib_images,
        
        'test_images': test_images,
        'test_labels': test_labels,

        'noise_ratio': noise_ratio
    }

def get_metrics(y_true, y_pred, verbose=False):
    if len(np.unique(y_true)) != 2:
        return None
    
    test_labels = np.array(y_true)
    pred_raw = np.array(y_pred).copy()
    scores = []
    threshold_range = np.arange(0., 1., 0.01); 
    for threshold in threshold_range:
        p = 1.*(pred_raw >= threshold)
        precision = sk_metrics.precision_score(test_labels, p)
        recall = sk_metrics.recall_score(test_labels, p)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        if(np.isnan(f1_score)):
            f1_score = 0.
        scores.append([threshold, precision, recall, f1_score])
    scores = np.array(scores, dtype='float16')

    auc = sk_metrics.auc(threshold_range, scores[:,-1])

    max_index = np.argmax(scores[:,-1])
    threshold, precision, recall, f1_score = scores[max_index]

    roc_auc = sk_metrics.roc_auc_score(y_true, y_pred)
    avg_precision = sk_metrics.average_precision_score(y_true, y_pred)
    
    p = 1.*(pred_raw >= 0.5)
    precision_at_50 = sk_metrics.precision_score(test_labels, p)
    recall_at_50 = sk_metrics.recall_score(test_labels, p)
    f1_score_at_50 = 2 * (precision_at_50 * recall_at_50) / (precision_at_50 + recall_at_50)
    if(np.isnan(f1_score_at_50)):
        f1_score_at_50 = 0.
    
    return {'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'f1_auc': auc,
            'threshold': threshold,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'precision_at_50': precision_at_50,
            'recall_at_50': recall_at_50,
            'f1_score_at_50': f1_score_at_50
           }