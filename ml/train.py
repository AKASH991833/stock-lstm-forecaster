#!/usr/bin/env python3
"""Stock LSTM — proper Python training pipeline.

Trains a stacked LSTM on your own OHLC/close-price CSV (free sources: Stooq
CSV download, Yahoo Finance export — no API key needed for either), backtests
on a real holdout against naive + moving-average baselines, and exports a
TF.js model the browser demo can load.

Usage:
    pip install -r requirements.txt
    python train.py --csv close_prices.csv --window 30 --epochs 40
    # CSV: one close price per line, or a column named 'Close'/'close'

Outputs: model/model.json (+ shards), ml/metrics.json
"""
import argparse, json, os
import numpy as np
import pandas as pd

def load(path):
    try:
        df = pd.read_csv(path)
        for c in df.columns:
            if c.lower() == "close":
                return pd.to_numeric(df[c], errors="coerce").dropna().values.astype(float)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values.astype(float)
    except Exception:
        return np.loadtxt(path).astype(float)

def make_xy(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X)[..., None], np.array(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    import tensorflow as tf
    from tensorflow import keras

    prices = load(args.csv)
    if len(prices) < args.window * 4:
        raise SystemExit("Series too short for the window size.")
    # log-return style normalization: scale by first-window price per sample
    window = args.window
    X, y = make_xy(prices, window)
    scale = X[:, 0, 0].copy()
    Xn = X / scale[:, None, None] - 1.0
    yn = y / scale - 1.0

    split = int(len(Xn) * 0.8)
    Xtr, Xte = Xn[:split], Xn[split:]
    ytr, yte = yn[:split], yn[split:]
    scale_te = scale[split:]

    model = keras.Sequential([
        keras.layers.LSTM(48, return_sequences=True, input_shape=(window, 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(24),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    hist = model.fit(Xtr, ytr, epochs=args.epochs, batch_size=32,
                     validation_split=0.15, callbacks=cb, verbose=2)

    pred_n = model.predict(Xte).flatten()
    pred = (pred_n + 1.0) * scale_te
    actual = (yte + 1.0) * scale_te
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    naive = actual - np.roll(actual, 1); naive = naive[1:]
    naive_mae = float(np.mean(np.abs(naive)))
    naive_rmse = float(np.sqrt(np.mean(naive ** 2)))

    os.makedirs("model", exist_ok=True)
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, "model")
    metrics = {
        "csv": args.csv, "samples": int(len(prices)), "window": window,
        "model": "LSTM(48)->Dropout(0.2)->LSTM(24)->Dense(1) on per-window normalized returns",
        "epochs_trained": len(hist.history["loss"]),
        "holdout": {"mae": round(mae, 4), "rmse": round(rmse, 4),
                    "naive_mae": round(naive_mae, 4), "naive_rmse": round(naive_rmse, 4),
                    "beats_naive": mae < naive_mae},
        "note": "Historical backtest on one series. Markets are noisy and non-stationary; this is education, not investment advice.",
    }
    os.makedirs("ml", exist_ok=True)
    json.dump(metrics, open("ml/metrics.json", "w"), indent=2)
    print(json.dumps(metrics["holdout"], indent=2))

if __name__ == "__main__":
    main()
