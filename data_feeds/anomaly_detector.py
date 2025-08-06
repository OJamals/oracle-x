import pandas as pd
import numpy as np
from typing import List, Dict, Union

def _as_numeric_series(signal: Union[pd.Series, pd.DataFrame, List[Dict], List[float], List[int]]) -> pd.Series:
    """
    Normalize input into a numeric pandas Series for anomaly detection.
    Accepts:
      - pd.Series -> returned as float series
      - pd.DataFrame with 'volume'/'Volume' -> that column
      - pd.DataFrame with a single numeric column -> that column
      - List[Dict] with 'volume' keys -> builds a Series
      - List[numeric] -> builds a Series
    Raises ValueError if no suitable numeric series found.
    """
    # Series
    if isinstance(signal, pd.Series):
        return pd.to_numeric(signal, errors='coerce').dropna()

    # DataFrame
    if isinstance(signal, pd.DataFrame):
        for col in ('volume', 'Volume'):
            if col in signal.columns:
                return pd.to_numeric(signal[col], errors='coerce').dropna()
        numeric_cols = signal.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 1:
            return pd.to_numeric(signal[numeric_cols[0]], errors='coerce').dropna()
        raise ValueError("No suitable numeric series found in DataFrame (expected 'volume'/'Volume' or a single numeric column).")

    # List[Dict]
    if isinstance(signal, list) and signal and isinstance(signal[0], dict):
        if 'volume' in signal[0] or 'Volume' in signal[0]:
            vals = []
            for item in signal:
                if isinstance(item, dict):
                    vals.append(item.get('volume', item.get('Volume')))
                else:
                    # Skip non-dict items defensively
                    continue
            return pd.to_numeric(pd.Series(vals), errors='coerce').dropna()
        raise ValueError("List[Dict] provided but no 'volume'/'Volume' key found.")

    # List[numeric]
    if isinstance(signal, list):
        return pd.to_numeric(pd.Series(signal), errors='coerce').dropna()

    raise TypeError("Unsupported input type for anomaly detection")

def detect_price_volume_anomalies(signal: Union[pd.Series, pd.DataFrame, List[Dict], List[float], List[int]], threshold: float = 3.0) -> List[int]:
    """
    Detect anomalies using Z-score method on a numeric time series.

    Parameters:
      signal: Series/DataFrame/List[Dict]/List[numeric] representing volumes
      threshold: absolute Z-score threshold to mark anomalies

    Returns:
      List[int]: indices where |z| > threshold
    """
    s = _as_numeric_series(signal)
    if len(s) < 5:
        return []
    std = s.std()
    if std == 0 or np.isnan(std):
        return []
    z = ((s - s.mean()) / std).abs()
    return z[z > threshold].index.tolist()
