# SVM Stock Prediction

An educational support vector regression example using daily stock prices. The original notebook is preserved as a historical artifact in `ISBANK_SVM.ipynb`; the runnable implementation is `svm_stock_prediction.py`.

## Run it

1. Download historical daily data as a CSV from [Yahoo Finance](https://finance.yahoo.com/quote/ISCTR.IS/history/) or another source with `Date`, `Open`, `High`, `Low`, `Close`, and `Volume` columns. `Adj Close` is optional. The example originally used ISCTR.IS on Borsa Istanbul.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Run: `python svm_stock_prediction.py --csv path/to/ISCTR.csv --horizon 5`

The script prints JSON with holdout mean absolute error, a last-close baseline, R², and estimates for the next five trading rows. It never downloads market data or places trades.

## Changes from the original notebook

- Removed the hard-coded path to one computer and the retired `sklearn.cross_validation` import.
- Corrected the `Open` to `Close` change feature, which was always zero in the notebook.
- Replaced the random train/test split with a chronological split and a horizon-sized gap between training targets and test dates.
- Fit the feature scaler only on training data during evaluation to prevent look-ahead leakage.
- Reported mean absolute error against a simple baseline. R² alone is easy to misread as “accuracy.”

Run `python -m unittest discover -s tests` to check data preparation and holdout behavior.

## Limits

This is a demonstration, not an investment recommendation or a validated trading strategy. One historical holdout does not establish future performance. Market regimes change, adjusted price data can be revised, and a model that misses the simple baseline should not be treated as useful. The estimates have no confidence interval and do not include transaction costs or market holidays.
