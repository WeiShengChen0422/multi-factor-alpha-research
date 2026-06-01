import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from functools import wraps
import os


class Factor:
    name = None
    dependencies = []

    def Calculation_Process(self, data_):
        raise NotImplementedError("This method must be overridden in your factor implementation.")


# Universal timing decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Starting function: {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__} completed, elapsed time: {elapsed_time:.4f} seconds")
        return result

    return wrapper


def get_data(stocks_name, dependencies, start_date, end_date):
    """
    Load stock data from CSV files and extract specified fields.

    Parameters:
    - stocks_name: list, list of stock names (e.g., ["AAPL", "GOOG"]).
    - dependencies: list, fields to extract (e.g., ["Volume", "Close"]).
    - start_date: str, start date in "YYYY-MM-DD" format.
    - end_date: str, end date in "YYYY-MM-DD" format.

    Returns:
    - data_dict: dict, contains data for each field where keys are the field names
                 and values are DataFrames with the extracted data.
    """
    data_dict = {dep: [] for dep in dependencies}  # Initialize dictionary with lists for each field

    for stock in stocks_name:
        file_path = f"D:/桌面/semester1/MF703 Programming/Project/stock_data/{stock}.csv"  # File path, e.g., "AAPL.csv"
        try:
            # Load specified columns and the date
            df = pd.read_csv(file_path, usecols=dependencies + ['Date'], index_col='Date', parse_dates=True)

            # Filter data by date range
            df = df.loc[start_date:end_date]

            for dep in dependencies:
                # Rename each field to the stock name and append to the corresponding list
                renamed_df = df[[dep]].rename(columns={dep: stock})
                data_dict[dep].append(renamed_df)
        except FileNotFoundError:
            print(f"File {file_path} not found, skipping.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Combine the DataFrame list for each field
    for dep in dependencies:
        if data_dict[dep]:
            data_dict[dep] = pd.concat(data_dict[dep], axis=1)  # Concatenate along columns
        else:
            data_dict[dep] = pd.DataFrame()  # Return an empty DataFrame if no data is available

    return data_dict


@timing_decorator
def Calc_Factors(stocks_name, factor_objects):
    """
    Calculate factor values for a list of stocks using factor objects.

    Parameters:
    - stocks_name: list, stock names.
    - factor_objects: list, factor objects containing dependencies and calculation methods.

    Returns:
    - results: dict, calculated factor values with factor names as keys and DataFrames as values.
    """
    results = {}  # Initialize the results dictionary

    for factor in factor_objects:
        # Get required data for the factor
        dependencies = factor.dependencies
        data_ = get_data(stocks_name, dependencies, start_date='2015-01-01', end_date='2024-11-01')

        # Call the Calculation_Process method of the factor
        factor_result = factor.Calculation_Process(data_).dropna()

        # Store the result in the dictionary with the factor name as the key
        results[factor.name] = factor_result

    return results


@timing_decorator
def Neutralize(data, how=None, date=None, axis=1):
    """
    Neutralize data by removing the effect of specified variables.

    Parameters:
    - data: dict or DataFrame, input data (dictionary with factor names as keys or a single DataFrame).
    - how: list, variables to neutralize against (default ['Volume']).
    - date: str, optional, specific date for neutralization.
    - axis: int, axis for neutralization (1: rows, 0: columns).

    Returns:
    - neutralized: dict or DataFrame, maintains the original data structure.
    """
    if how is None:
        how = ['Volume']

    # If the input is a single DataFrame, convert it to a dictionary for consistent handling
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        data = {'Factor': data}

    neutralized_data = {}

    for key, df in data.items():
        print(f"Neutralizing: {key}")
        # Extract stock names
        stocks = df.columns.tolist() if axis == 1 else df.index.tolist()

        # Get required data for neutralization
        dependencies = get_data(stocks, how, start_date='2015-01-01', end_date='2024-11-01')

        # If a specific date is provided, filter the data for that date
        if date is not None:
            dependencies = {factor: dep.loc[date] for factor, dep in dependencies.items()}
            df = df.loc[date]

        # Initialize neutralization results for the current data
        neutralized = (
            pd.DataFrame(index=df.index, columns=df.columns) if axis == 1
            else pd.DataFrame(index=df.columns, columns=df.index).T
        )

        # Perform neutralization along the specified axis
        if axis == 1:  # Neutralize by rows (dates)
            for idx in df.index:
                y = df.loc[idx]  # Factor values for the current date
                X = pd.concat([dependencies[factor].loc[idx] for factor in how], axis=1)  # Combine independent variables
                X = sm.add_constant(X)  # Add an intercept term
                model = sm.OLS(y, X, missing='drop').fit()
                neutralized.loc[idx] = model.resid  # Extract residuals as neutralized values

        elif axis == 0:  # Neutralize by columns (stocks)
            for col in df.columns:
                y = df[col]  # Factor values for the current stock
                X = pd.concat([dependencies[factor][col] for factor in how], axis=1)  # Combine independent variables
                X = sm.add_constant(X)  # Add an intercept term
                model = sm.OLS(y, X, missing='drop').fit()
                neutralized[col] = model.resid  # Extract residuals as neutralized values

        # Ensure correct data types
        neutralized = neutralized.apply(pd.to_numeric, errors='coerce')

        # Store the neutralized data in the dictionary
        neutralized_data[key] = neutralized

    # If the input was a single DataFrame, return a single DataFrame
    if is_dataframe:
        return neutralized_data['Factor']

    return neutralized_data


@timing_decorator
def Winsorize(data, qrange=[0.05, 0.95], inclusive=True, inf2nan=True, axis=1):
    """
    Apply Winsorization to the data to limit extreme values.

    Parameters:
    - data: DataFrame, numpy array, or dict, input data.
    - qrange: list, percentile range for Winsorization [lower bound, upper bound].
    - inclusive: bool, whether to include boundary values.
    - inf2nan: bool, whether to replace infinity values with NaN.
    - axis: int, axis to apply Winsorization (1: rows, 0: columns).

    Returns:
    - winsorized_data: Winsorized data (same structure as input).
    """

    def winsorize_single(data):
        is_dataframe = isinstance(data, pd.DataFrame)
        original_data = data
        if is_dataframe:
            data = data.values

        # Calculate percentile bounds
        lower_q, upper_q = qrange
        if axis == 0:  # Apply by columns
            lower_bounds = np.nanquantile(data, lower_q, axis=0)
            upper_bounds = np.nanquantile(data, upper_q, axis=0)
        elif axis == 1:  # Apply by rows
            lower_bounds = np.nanquantile(data, lower_q, axis=1).reshape(-1, 1)
            upper_bounds = np.nanquantile(data, upper_q, axis=1).reshape(-1, 1)
        else:
            raise ValueError("axis must be 0 (columns) or 1 (rows)")

        # Apply Winsorization
        if inclusive:
            data = np.where(data < lower_bounds, lower_bounds, data)
            data = np.where(data > upper_bounds, upper_bounds, data)
        else:
            data = np.where(data <= lower_bounds, np.nan, data)
            data = np.where(data >= upper_bounds, np.nan, data)

        # Replace infinity values with NaN
        if inf2nan:
            data = np.where(np.isinf(data), np.nan, data)

        # Convert back to DataFrame if the original input was a DataFrame
        if is_dataframe:
            data = pd.DataFrame(data, columns=original_data.columns, index=original_data.index)

        return data

    # If the input is a dictionary, process each key separately
    if isinstance(data, dict):
        winsorized_data = {}
        for key, df in data.items():
            print(f"Processing: {key}")
            winsorized_data[key] = winsorize_single(df)
        return winsorized_data

    # Otherwise, process directly
    return winsorize_single(data)


@timing_decorator
def Standardize(data, axis=1):
    """
    Standardize data to have zero mean and unit variance.

    Parameters:
    - data: dict, DataFrame, or Series, input data.
    - axis: int, axis for standardization (1: rows, 0: columns).

    Returns:
    - standardized_data: dict, DataFrame, or Series, retains the input data structure.
    """
    # Check input type
    is_dataframe = isinstance(data, pd.DataFrame)
    is_series = isinstance(data, pd.Series)

    # If the input is a single DataFrame or Series, convert to dictionary for consistent handling
    if is_dataframe or is_series:
        data = {'Factor': data}

    standardized_data = {}

    for key, df in data.items():
        print(f"Standardizing: {key}")
        is_dataframe = isinstance(df, pd.DataFrame)
        is_series = isinstance(df, pd.Series)

        # Convert Pandas data to NumPy array
        original_data = df
        if is_dataframe or is_series:
            df = df.values

        # Perform standardization
        if is_dataframe:
            if axis == 0:  # By columns
                mean = np.nanmean(df, axis=0)
                std = np.nanstd(df, axis=0)
            elif axis == 1:  # By rows
                mean = np.nanmean(df, axis=1).reshape(-1, 1)
                std = np.nanstd(df, axis=1).reshape(-1, 1)
            else:
                raise ValueError("axis must be 0 (columns) or 1 (rows)")
        else:
            mean = np.nanmean(df)
            std = np.nanstd(df)

        # Avoid division by zero
        std = np.where(std == 0, np.nan, std)

        # Standardize
        standardized_df = (df - mean) / std

        # Convert back to Pandas if the input was a DataFrame or Series
        if is_dataframe:
            standardized_df = pd.DataFrame(
                standardized_df, index=original_data.index, columns=original_data.columns
            )
        elif is_series:
            standardized_df = pd.Series(standardized_df, index=original_data.index)

        # Store the standardized data in the dictionary
        standardized_data[key] = standardized_df

    # If the input was a single DataFrame/Series, return a single standardized result
    if len(standardized_data) == 1 and ('Factor' in standardized_data):
        return standardized_data['Factor']

    return standardized_data


def Processing_data(data):
    """
    Process data through neutralization, winsorization, and standardization.

    Parameters:
    - data: dict or DataFrame, input data.

    Returns:
    - res: DataFrame or dict, processed data.
    """
    res = Neutralize(data, how=None, date=None, axis=1)
    res = Winsorize(res, qrange=[0.05, 0.95], inclusive=True, inf2nan=True, axis=1)
    res = Standardize(res, axis=1)
    return res
