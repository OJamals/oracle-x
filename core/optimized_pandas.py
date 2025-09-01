"""
Optimized Pandas Operations for Oracle-X
Implements efficient DataFrame operations with vectorization and memory optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Literal
import logging

logger = logging.getLogger(__name__)

class OptimizedPandasProcessor:
    """Optimized pandas operations for improved performance"""

    def __init__(self):
        self.categorical_threshold = 0.5  # Convert to categorical if < 50% unique values

    def optimize_dataframe_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame schema for memory efficiency and performance
        """
        # Convert object columns to categorical where beneficial
        for col in df.select_dtypes(include=['object', 'string']):
            if self._should_convert_to_categorical(df[col]):
                df[col] = df[col].astype('category')
                logger.info(f"Converted column '{col}' to categorical")

        # Downcast numeric types for memory efficiency
        df = self._downcast_numeric_types(df)

        # Convert date columns to datetime64
        date_columns = self._identify_date_columns(df)
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def _should_convert_to_categorical(self, series: pd.Series) -> bool:
        """Determine if a series should be converted to categorical"""
        if series.dtype.name == 'category':
            return False

        n_unique = series.nunique()
        n_total = len(series)

        # Convert if less than threshold of values are unique
        return (n_unique / n_total) < self.categorical_threshold and n_unique > 1

    def _downcast_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast numeric types to save memory"""
        for col in df.select_dtypes(include=['int64']):
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float64']):
            df[col] = pd.to_numeric(df[col], downcast='float')

        return df

    def _identify_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that contain dates"""
        date_columns = []

        for col in df.columns:
            if df[col].dtype == 'object':
                # Sample a few values to check if they look like dates
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    try:
                        pd.to_datetime(sample, errors='coerce')
                        # If conversion succeeds for most values, it's likely a date column
                        success_rate = (pd.to_datetime(sample, errors='coerce').notna().sum() / len(sample))
                        if success_rate > 0.8:
                            date_columns.append(col)
                    except:
                        continue

        return date_columns

    def vectorized_query_operations(self, df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform complex filtering using vectorized operations
        """
        mask = pd.Series(True, index=df.index)

        for column, condition in conditions.items():
            if column not in df.columns:
                continue

            if isinstance(condition, dict):
                # Complex conditions
                if 'min' in condition:
                    mask &= (df[column] >= condition['min'])
                if 'max' in condition:
                    mask &= (df[column] <= condition['max'])
                if 'in' in condition:
                    mask &= df[column].isin(condition['in'])
                if 'not_in' in condition:
                    mask &= ~df[column].isin(condition['not_in'])
            else:
                # Simple equality
                mask &= (df[column] == condition)

        return df[mask]

    def efficient_groupby_operations(self, df: pd.DataFrame,
                                   groupby_cols: List[str],
                                   agg_operations: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Perform efficient groupby operations with optimized aggregations
        """
        # Use categorical for groupby columns if beneficial
        for col in groupby_cols:
            if col in df.columns and self._should_convert_to_categorical(df[col]):
                df[col] = df[col].astype('category')

        # Perform groupby with specified aggregations
        grouped = df.groupby(groupby_cols, observed=True)

        # Build aggregation dictionary
        agg_dict = {}
        for col, operations in agg_operations.items():
            if col in df.columns:
                for op in operations:
                    if op in ['mean', 'sum', 'count', 'min', 'max', 'std', 'var']:
                        agg_dict[f"{col}_{op}"] = (col, op)
                    elif op == 'first':
                        agg_dict[f"{col}_first"] = (col, 'first')
                    elif op == 'last':
                        agg_dict[f"{col}_last"] = (col, 'last')

        result = grouped.agg(**agg_dict)

        # Flatten column names if needed
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip() for col in result.columns]

        return result.reset_index()

    def rolling_window_operations(self, df: pd.DataFrame,
                                value_col: str,
                                window_sizes: List[int],
                                operations: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Perform efficient rolling window operations
        """
        result_df = df.copy()

        for window in window_sizes:
            for op in operations:
                col_name = f"{value_col}_rolling_{op}_{window}"
                if op == 'mean':
                    result_df[col_name] = df[value_col].rolling(window=window, min_periods=1).mean()
                elif op == 'std':
                    result_df[col_name] = df[value_col].rolling(window=window, min_periods=1).std()
                elif op == 'sum':
                    result_df[col_name] = df[value_col].rolling(window=window, min_periods=1).sum()
                elif op == 'min':
                    result_df[col_name] = df[value_col].rolling(window=window, min_periods=1).min()
                elif op == 'max':
                    result_df[col_name] = df[value_col].rolling(window=window, min_periods=1).max()

        return result_df

    def memory_efficient_merge(self, left_df: pd.DataFrame,
                             right_df: pd.DataFrame,
                             left_on: str,
                             right_on: str,
                             how: Literal['left', 'right', 'outer', 'inner', 'cross'] = 'left') -> pd.DataFrame:
        """
        Perform memory-efficient merge operations
        """
        # Optimize dtypes before merge
        left_df = self.optimize_dataframe_schema(left_df)
        right_df = self.optimize_dataframe_schema(right_df)

        # Use copy=False to avoid unnecessary copying
        result = pd.merge(
            left_df, right_df,
            left_on=left_on, right_on=right_on,
            how=how
        )

        return result

    def vectorized_string_operations(self, df: pd.DataFrame,
                                   text_columns: List[str]) -> pd.DataFrame:
        """
        Perform vectorized string operations for text processing
        """
        result_df = df.copy()

        for col in text_columns:
            if col in df.columns:
                # Vectorized string cleaning
                result_df[f"{col}_clean"] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .str.replace(r'[^\w\s]', '', regex=True)
                )

                # Vectorized length calculation
                result_df[f"{col}_length"] = df[col].astype(str).str.len()

                # Vectorized word count
                result_df[f"{col}_word_count"] = (
                    df[col]
                    .astype(str)
                    .str.split()
                    .str.len()
                )

        return result_df

    def chunked_file_processing(self, file_path: str,
                              chunk_size: int = 10000,
                              processing_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Process large CSV files in chunks for memory efficiency
        """
        chunks = []
        total_rows = 0

        # Read file in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            total_rows += len(chunk)

            # Optimize chunk
            chunk = self.optimize_dataframe_schema(chunk)

            # Apply processing function if provided
            if processing_func:
                chunk = processing_func(chunk)

            chunks.append(chunk)

            # Log progress
            logger.info(f"Processed {total_rows} rows so far...")

        # Combine chunks efficiently
        if chunks:
            result = pd.concat(chunks, ignore_index=True, copy=False)
            logger.info(f"Successfully processed {len(result)} total rows")
            return result
        else:
            return pd.DataFrame()

# Global instance for reuse
_optimized_pandas = OptimizedPandasProcessor()

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame for memory efficiency and performance
    """
    return _optimized_pandas.optimize_dataframe_schema(df)

def vectorized_query(df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    """
    Perform vectorized query operations
    """
    return _optimized_pandas.vectorized_query_operations(df, conditions)

def efficient_groupby(df: pd.DataFrame, groupby_cols: List[str],
                     agg_operations: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Perform efficient groupby operations
    """
    return _optimized_pandas.efficient_groupby_operations(df, groupby_cols, agg_operations)

def rolling_window_analysis(df: pd.DataFrame, value_col: str,
                          window_sizes: List[int] = [5, 10, 20],
                          operations: List[str] = ['mean', 'std']) -> pd.DataFrame:
    """
    Perform rolling window analysis
    """
    return _optimized_pandas.rolling_window_operations(df, value_col, window_sizes, operations)

def memory_efficient_merge(left_df: pd.DataFrame, right_df: pd.DataFrame,
                         left_on: str, right_on: str,
                         how: Literal['left', 'right', 'outer', 'inner', 'cross'] = 'left') -> pd.DataFrame:
    """
    Perform memory-efficient merge
    """
    return _optimized_pandas.memory_efficient_merge(left_df, right_df, left_on, right_on, how)

def process_large_csv(file_path: str, chunk_size: int = 10000,
                     processing_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Process large CSV files in chunks
    """
    return _optimized_pandas.chunked_file_processing(file_path, chunk_size, processing_func)
