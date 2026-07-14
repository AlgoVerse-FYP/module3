import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf2onnx, onnx
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

def make_splits(train_data):
    Xs, Xn, y = train_data["X_seq"], train_data["X_net"], train_data["y"]
    is_val = train_data["is_val"]                 # True = out-of-time validation rows
    tr, va = ~is_val, is_val
    return (Xs[tr], Xn[tr], y[tr]), (Xs[va], Xn[va], y[va])


def train_model(train_data, cfg, bidirectional=True, seed=42):
    tf.random.set_seed(seed); np.random.seed(seed)
    (Xs, Xn, y), (Xsv, Xnv, yv) = make_splits(train_data)   # was: make_splits(train_data, cfg["data"]["val_fraction"])

    m = build_model(cfg["features"]["window"], 1 + len(SEQ_NUMERIC),
                    train_data["n_categories"], len(NET_FEATURES),
                    embed_dim=cfg["model"]["embed_dim"],
                    lstm_units=cfg["model"]["lstm_units"],
                    dropout=cfg["model"]["dropout"],
                    bidirectional=bidirectional)
    m.compile(optimizer=keras.optimizers.Adam(cfg["training"]["learning_rate"]),
              loss=focal_loss())
    es = keras.callbacks.EarlyStopping(monitor="val_loss",
                                       patience=cfg["training"].get("early_stopping_patience", 2),
                                       restore_best_weights=True)
    hist = m.fit([Xs, Xn], y, validation_data=([Xsv, Xnv], yv),
                 epochs=cfg["training"]["epochs"],
                 batch_size=cfg["training"]["batch_size"],
                 callbacks=[es], verbose=2)
    return m, hist, (Xsv, Xnv, yv)

def _run():
    cfg = load_config()
    nrows = cfg["data"]["nrows"]
    train = build_dataset(path(cfg["data"]["train_csv"]), nrows=nrows,
                      val_fraction=cfg["data"]["val_fraction"])
    m, hist, _ = train_model(train, cfg, bidirectional=cfg["model"]["bidirectional"])
    m.save(path("results/models/module3_model.keras"))

    try:
         # type: ignore[import-not-found]
        spec = (tf.TensorSpec((None, cfg["features"]["window"], 1 + len(SEQ_NUMERIC)),
                              tf.float32, name="seq_input"),
                tf.TensorSpec((None, len(NET_FEATURES)), tf.float32, name="net_input"))
        om, _ = tf2onnx.convert.from_keras(m, input_signature=spec, opset=15)
        onnx.save(om, path("results/models/module3_model.onnx"))
        print("onnx exported")
    except Exception as e:
        print("onnx skipped:", e)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("focal loss"); plt.legend()
    plt.title("Phase 4 - training curve"); plt.tight_layout()
    plt.savefig(path("results/figures/phase4_training_curve.png"), dpi=130)
    print("saved -> module3_model.keras + phase4_training_curve.png")

if __name__ == "__main__":
    _run()
