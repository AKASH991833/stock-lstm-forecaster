# ML pipeline (Python)

Trains the LSTM offline on your own price CSV and exports a TF.js model for
the live browser demo.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python ml/train.py --csv close_prices.csv --window 30 --epochs 40
```

- Input: CSV with a `Close` column (or a single numeric column). Free keyless
  sources: Stooq CSV export, Yahoo Finance download.
- Model: LSTM(48) → Dropout(0.2) → LSTM(24) → Dense on per-window normalized
  returns; early stopping on validation loss.
- Honest backtest: 20% holdout, MAE/RMSE vs a naive persistence baseline —
  printed in `ml/metrics.json` with `beats_naive` so a bad result is visible,
  not hidden.

This is an educational tool, not investment advice.
