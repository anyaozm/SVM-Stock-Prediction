"""Educational SVR example for a Yahoo-style historical stock CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = ['adjusted_close', 'high_low_pct', 'open_close_pct', 'volume']

def load_prices(csv_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    required = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    raw = raw.dropna(subset=['Date']).sort_values('Date').drop_duplicates('Date', keep='last')
    number = lambda key: pd.to_numeric(raw[key], errors='coerce').to_numpy()
    close = number('Close')
    frame = pd.DataFrame(index=pd.DatetimeIndex(raw['Date'], name='Date'))
    frame['adjusted_close'] = number('Adj Close') if 'Adj Close' in raw else close
    frame['high_low_pct'] = 100 * (number('High') - number('Low')) / close
    frame['open_close_pct'] = 100 * (close - number('Open')) / number('Open')
    frame['volume'] = number('Volume')
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

def evaluate_and_forecast(prices: pd.DataFrame, horizon: int = 5, test_fraction: float = .2) -> dict:
    if horizon < 1 or not 0 < test_fraction < .5:
        raise ValueError('horizon must be positive and test_fraction between 0 and 0.5')
    frame = prices.copy()
    frame['target'] = frame['adjusted_close'].shift(-horizon)
    labeled = frame.dropna(subset=['target'])
    test_start = int(len(labeled) * (1 - test_fraction))
    train_end = test_start - horizon  # Purge labels that reach into the test period.
    if train_end < 20 or len(labeled) - test_start < 5:
        raise ValueError('Need more rows for this horizon and holdout size')
    train, test = labeled.iloc[:train_end], labeled.iloc[test_start:]
    model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
    model.fit(train[FEATURES], train['target'])  # Train-only scaler prevents look-ahead.
    predictions = model.predict(test[FEATURES])
    metrics = {
        'test_mae': float(mean_absolute_error(test['target'], predictions)),
        'baseline_mae': float(mean_absolute_error(test['target'], test['adjusted_close'])),
        'test_r2': float(r2_score(test['target'], predictions)),
    }
    model.fit(labeled[FEATURES], labeled['target'])
    return {
        'horizon_trading_rows': horizon,
        'training_rows': len(train), 'test_rows': len(test),
        'train_last_date': str(train.index[-1].date()),
        'test_first_date': str(test.index[0].date()),
        'metrics': metrics,
        'forward_estimates': [float(value) for value in model.predict(frame.iloc[-horizon:][FEATURES])],
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path, help='Daily price CSV')
    parser.add_argument('--horizon', type=int, default=5, help='Trading rows ahead to predict')
    args = parser.parse_args()
    print(json.dumps(evaluate_and_forecast(load_prices(args.csv), args.horizon), indent=2))

if __name__ == '__main__':
    main()
