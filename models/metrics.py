from typing import Optional, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

from models.image_matcher import ImageMatcher


def get_roc_auc_curve_metrics(image_matcher: ImageMatcher,
                              ds_pairs: tf.data.Dataset,
                              n_batches: int=50)-> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y_pred = np.array([], dtype=np.float32)
    y_true = np.array([], dtype=np.uint8)

    for x, y_true_batch in tqdm(ds_pairs.take(n_batches), total=n_batches):
        y_pred_batch = image_matcher.model.predict(x, verbose=0)  # type: ignore
        y_pred = np.hstack([y_pred, y_pred_batch])
        y_true = np.hstack([y_true, y_true_batch])

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, roc_auc
