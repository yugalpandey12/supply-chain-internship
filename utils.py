import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error"""
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error"""
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + K.epsilon()))

def coeff_determination(y_true, y_pred):
    """Calculates R-squared (Coefficient of Determination) using Keras backend"""
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())