import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import load_config, path
from src.features.phase1_features import build_dataset, SEQ_NUMERIC, NET_FEATURES


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = keras.backend.epsilon()
        p = tf.clip_by_value(y_pred, eps, 1 - eps)
        pt = tf.where(tf.equal(y_true, 1), p, 1 - p)
        at = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(-at * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return loss

def build_model(window, n_seq_features, n_categories, n_net_features,
                embed_dim=8, lstm_units=64, dense_head=32, dropout=0.3,
                bidirectional=True):
    # ----- sequence branch -----
    seq_in = keras.Input((window, n_seq_features), name="seq_input")
    cat = layers.Lambda(lambda x: x[:, :, 0])(seq_in)        # column 0 = category index
    rest = layers.Lambda(lambda x: x[:, :, 1:])(seq_in)      # columns 1+ = the float features
    emb = layers.Embedding(n_categories, embed_dim)(cat)     # category -> learned vector
    merged = layers.Concatenate()([emb, rest])
    masked = layers.Masking(0.0)(merged)                     # ignore padded steps
    lstm = layers.LSTM(lstm_units)
    if bidirectional:
        lstm = layers.Bidirectional(lstm)                   # forward + backward
    seq_repr = lstm(masked)

    # ----- network branch -----
    net_in = keras.Input((n_net_features,), name="net_input")
    net_repr = layers.Dense(16, activation="relu")(net_in)

    # ----- fusion + head -----
    fused = layers.Concatenate()([seq_repr, net_repr])
    h = layers.Dense(dense_head, activation="relu")(fused)
    h = layers.Dropout(dropout)(h)
    out = layers.Dense(1, activation="sigmoid")(h)
    return keras.Model([seq_in, net_in], out)