import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from svm_stock_prediction import evaluate_and_forecast, load_prices

class StockPredictionTests(unittest.TestCase):
    def test_features_and_time_ordered_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prices.csv'
            dates = pd.bdate_range('2024-01-01', periods=100)
            close = np.linspace(10, 20, len(dates)) + np.sin(np.arange(len(dates)) / 5)
            pd.DataFrame({'Date': dates, 'Open': close - 1, 'High': close + 2,
                          'Low': close - 2, 'Close': close, 'Adj Close': close,
                          'Volume': np.arange(len(dates)) + 1000}).to_csv(path, index=False)
            features = load_prices(path)
            self.assertNotEqual(features.iloc[0]['open_close_pct'], 0)
            result = evaluate_and_forecast(features, horizon=5)
            self.assertLess(result['train_last_date'], result['test_first_date'])
            self.assertEqual(len(result['forward_estimates']), 5)
            self.assertGreaterEqual(result['metrics']['test_mae'], 0)

    def test_missing_columns_are_explained(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.csv'
            path.write_text('Date,Close\n2024-01-01,5\n')
            with self.assertRaisesRegex(ValueError, 'Missing CSV columns'):
                load_prices(path)

if __name__ == '__main__':
    unittest.main()
