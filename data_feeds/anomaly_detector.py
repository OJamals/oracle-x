import pandas as pd

from typing import List, Dict

def detect_price_volume_anomalies(price_data: List[Dict]) -> List[Dict]:
    """
    Detect price/volume anomalies using z-score method on volume spikes.
    TODO: Expand anomaly detection logic as needed for your use case.
    Args:
        price_data (List[Dict]): List of price/volume dicts.
    Returns:
        List[Dict]: List of detected anomalies.
    """
    import pandas as pd
    df = pd.DataFrame(price_data)
    mean_vol = df["volume"].mean()
    std_vol = df["volume"].std()
    df["zscore"] = (df["volume"] - mean_vol) / std_vol

    anomalies = df[df["zscore"].abs() > 2]
    return anomalies.to_dict(orient="records")
