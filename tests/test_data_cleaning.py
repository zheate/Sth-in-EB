import unittest
import pandas as pd
from utils.data_cleaning import ensure_numeric, drop_zero_current, clean_current_metric

class TestDataCleaning(unittest.TestCase):
    def test_ensure_numeric_strict(self):
        df = pd.DataFrame({"a": ["1", "2"], "b": ["3.0", None]})
        out = ensure_numeric(df, ["a", "b"], strict=True)
        self.assertTrue(pd.api.types.is_numeric_dtype(out["a"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(out["b"]))

    def test_drop_zero_current(self):
        df = pd.DataFrame({"I": [0.0, 1e-7, 0.1], "v": [1, 2, 3]})
        out = drop_zero_current(df, "I", tol=1e-6)
        self.assertEqual(len(out), 2)

    def test_clean_current_metric(self):
        df = pd.DataFrame({"I": [0.0, 0.2, 0.1], "m": [1.0, 2.0, 3.0]})
        out = clean_current_metric(df, "I", "m")
        self.assertEqual(list(out.columns), ["current", "value"])
        self.assertTrue((out["current"] > 0).all())

if __name__ == "__main__":
    unittest.main()