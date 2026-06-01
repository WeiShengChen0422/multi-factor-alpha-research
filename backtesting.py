import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay


def generate_rebalancing_dates(factor_table, frequency):
    """
    Generate rebalancing dates based on the specified frequency.

    Parameters:
    - factor_table: DataFrame, factor values with dates as the index and stocks as columns.
    - frequency: str, rebalancing frequency ('Daily', 'Weekly', 'Monthly').

    Returns:
    - rebalance_dates: DatetimeIndex, rebalancing dates.
    """
    if frequency == 'Daily':
        rebalance_dates = factor_table.index
    elif frequency == 'Weekly':
        # Resample to weekly frequency and get the last available trading day of the week
        rebalance_dates = factor_table.resample('W-FRI').apply(
            lambda x: x.index[-1] if not x.empty else pd.NaT).dropna().index
    elif frequency == 'Monthly':
        # Get the first available trading day of each month
        rebalance_dates = factor_table.groupby(factor_table.index.to_period('M')).apply(
            lambda x: x.index[0]).dropna()
        rebalance_dates = pd.to_datetime(rebalance_dates.values)
    else:
        raise ValueError("Unsupported frequency. Choose from 'Daily', 'Weekly', 'Monthly'.")
    return rebalance_dates


def backtesting_settlement(factors_dict, price_table, quantiles_to_hold_dict, rebalance_dates, all_quantiles):
    """
    Perform backtesting based on factor values and returns (daily settlement, periodic rebalancing).

    Parameters:
    - factors_dict: dict, contains multiple factor DataFrames, with keys as factor names and values as DataFrames.
    - price_table: DataFrame, stock prices with dates as the index and stocks as columns.
    - quantiles_to_hold_dict: dict, specifies the quantiles to hold for each factor.
    - rebalance_dates: DatetimeIndex, rebalancing dates.
    - all_quantiles: int, total number of quantiles.

    Returns:
    - ret_series: Series, daily portfolio returns with dates as the index.
    """
    # Calculate daily returns
    ret_daily = price_table.pct_change().dropna()

    # Initialize holding status
    current_long = []

    # Initialize return series
    ret_series = pd.Series(0.0, index=ret_daily.index)

    # Iterate over daily returns in chronological order
    for current_date in ret_daily.index:
        # Check if it's a rebalancing date
        if current_date in rebalance_dates:
            # Get factor values from the previous day to avoid look-ahead bias
            prev_date = current_date - BDay(1)
            while prev_date not in price_table.index:
                prev_date -= BDay(1)
                if prev_date < price_table.index[0]:
                    prev_date = None
                    break

            if prev_date is None:
                # Skip rebalancing if the previous day's data is unavailable
                pass
            else:
                # Store selected stocks for each factor
                factor_selected_stocks = []
                for factor_name, factor_df in factors_dict.items():
                    try:
                        factors_at_date = factor_df.loc[prev_date].dropna()
                    except KeyError:
                        factors_at_date = pd.Series(dtype='float64')

                    if factors_at_date.empty:
                        # Skip if factor values are empty
                        continue
                    else:
                        try:
                            # Divide factor values into specified quantiles
                            factor_quantiles = pd.qcut(factors_at_date.rank(method='first'), all_quantiles,
                                                       labels=False) + 1  # labels from 1 to all_quantiles
                        except ValueError as e:
                            print(f"Error during quantile calculation on {prev_date} for factor {factor_name}: {e}")
                            continue  # Skip the date if an error occurs
                        # Get stocks in the specified quantiles
                        quantiles_to_hold = quantiles_to_hold_dict[factor_name]
                        stocks_in_quantiles = factor_quantiles[factor_quantiles.isin(quantiles_to_hold)].index.tolist()
                        factor_selected_stocks.append(set(stocks_in_quantiles))

                if factor_selected_stocks:
                    # Take union of selected stocks across multiple factors (change to intersection if needed)
                    selected_stocks = set.union(*factor_selected_stocks)
                    current_long = list(selected_stocks)
                else:
                    current_long = []

        # Calculate portfolio return for the day
        if current_long:
            # Handle potential missing values
            long_ret = ret_daily.loc[current_date, current_long].dropna().mean()
        else:
            long_ret = 0.0

        # Assign return to the return series
        ret_series.loc[current_date] = long_ret

    # Remove dates with zero returns (usually before holding any positions)
    ret_series = ret_series[ret_series != 0.0]

    return ret_series


def calculate_performance_metrics(ret_series, risk_free_rate=0.0):
    """
    Calculate annualized return, annualized volatility, and Sharpe ratio.

    Parameters:
    - ret_series: Series, portfolio returns with dates as the index.
    - risk_free_rate: float, risk-free rate (default is 0).

    Returns:
    - metrics: dict, contains annualized return, annualized volatility, and Sharpe ratio.
    """
    periods_per_year = 252
    mean_return = ret_series.mean() * periods_per_year
    volatility = ret_series.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility != 0 else np.nan

    metrics = {
        'Annualized Return': mean_return,
        'Annualized Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio
    }

    return metrics


class Backtesting:
    def __init__(self, factor_data, start_date, end_date):
        """
        Initialize the Backtesting class with factor data and dates.

        Parameters:
        - factor_data: dict, contains multiple factor DataFrames with factor names as keys.
        - start_date: str, start date of the backtest.
        - end_date: str, end date of the backtest.
        """
        self.all_quantiles_input = input("Enter the total number of quantiles (e.g., 8, 10, 20, 40, etc.):")
        try:
            self.all_quantiles = int(self.all_quantiles_input)
            if self.all_quantiles <= 0:
                print("The total number of quantiles must be a positive integer.")
                exit(1)
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
            exit(1)

        self.factors = factor_data
        self.stock_code_sets = []
        for name, factor in self.factors.items():
            factor.index = factor.index.tz_localize(None)
            self.stock_code_sets.append(set(factor.columns.tolist()))
        self.start = start_date
        self.end = end_date
        self.stock_pool = list(set.union(*self.stock_code_sets))
        if len(self.stock_pool) == 0:
            print("Error: No common stocks among all factors. Exiting.")
            exit(1)
        for factor_name, factor_df in self.factors.items():
            self.factors[factor_name] = factor_df.shift(1).dropna()
            print(f"Factor data from {factor_name} shifted by one day to prevent look-ahead bias.")
        self.quantiles_to_hold_dict = {}
        for factor_name in self.factors.keys():
            quantiles_input = input(
                f"Enter the quantiles to hold for factor '{factor_name}' (1-{self.all_quantiles}), separated by commas, e.g., 1,5,{self.all_quantiles}:")
            try:
                quantiles_list = [int(q.strip()) for q in quantiles_input.split(',')]
                if not all(1 <= q <= self.all_quantiles for q in quantiles_list):
                    print(f"All quantiles must be between 1 and {self.all_quantiles}.")
                    exit(1)
                self.quantiles_to_hold_dict[factor_name] = quantiles_list
            except ValueError:
                print(f"Invalid input. Please enter numbers between 1 and {self.all_quantiles}, separated by commas.")
                exit(1)

    def get_stock_data(self, stock_pool):
        """
        Fetch adjusted close price data for stocks using yfinance.
        """
        print(f"Downloading data for {len(stock_pool)} stocks from {self.start} to {self.end}...")
        data = yf.download(stock_pool, start=self.start, end=self.end)['Adj Close']

        # If only one stock is downloaded, convert Series to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()

        # Remove timezone information
        data.index = data.index.tz_localize(None)

        # Check data completeness
        missing_stocks = [stock for stock in stock_pool if stock not in data.columns]
        if missing_stocks:
            print(f"Warning: The following stocks have no price data and will be excluded: {missing_stocks}")

        return data

    def back_test(self):
        """
        Perform backtesting based on factor data and stock prices.
        """
        price_data = self.get_stock_data(self.stock_pool)

        # Check data completeness
        price_data = price_data.dropna(axis=1, how='any')  # Remove columns with missing values
        common_stocks = price_data.columns.tolist()

        # Ensure factor data and price data have consistent stock codes
        for factor_name in self.factors.keys():
            self.factors[factor_name] = self.factors[factor_name][common_stocks]

        print(f"Number of stocks after aligning data: {len(common_stocks)}")

        if len(common_stocks) == 0:
            print("Error: No common stocks after aligning data. Exiting.")
            exit(1)

        # Define rebalancing frequencies: daily, weekly, and monthly
        rebalancing_frequencies = {
            'Daily': 'Daily',
            'Weekly': 'Weekly',
            'Monthly': 'Monthly'
        }

        # Fetch market index data (e.g., S&P 500)
        market_symbol = '^GSPC'
        market_data = self.get_stock_data([market_symbol])[market_symbol]

        # Calculate daily market returns
        market_ret_daily = market_data.pct_change().dropna()

        # =============== New Section: Calculate average return of the entire stock pool ===============
        pool_ret_daily = price_data.pct_change().mean(axis=1).dropna()

        # Initialize result dictionaries
        portfolio_ret_dict = {}
        performance_metrics = {}  # Store performance metrics
        market_ret_dict = {}  # Store market returns
        pool_ret_dict = {}  # Store pool returns (new)

        for freq_name, freq in rebalancing_frequencies.items():
            print(f"\nProcessing {freq_name} rebalancing...")
            # Generate rebalancing dates using one of the factors' date index
            sample_factor = next(iter(self.factors.values()))
            rebalance_dates = generate_rebalancing_dates(sample_factor, freq)
            print(f"Number of rebalance dates for {freq_name}: {len(rebalance_dates)}")

            # Calculate portfolio returns (daily settlement)
            portfolio_ret = backtesting_settlement(self.factors, price_data, self.quantiles_to_hold_dict,
                                                   rebalance_dates,
                                                   self.all_quantiles)

            # Align dates
            common_dates = portfolio_ret.index.intersection(market_ret_daily.index)
            portfolio_ret_aligned = portfolio_ret.loc[common_dates]
            market_ret_aligned = market_ret_daily.loc[common_dates]

            # Align pool returns
            pool_ret_aligned = pool_ret_daily.loc[common_dates]

            ret_series_resampled = portfolio_ret_aligned
            market_ret_resampled = market_ret_aligned
            pool_ret_resampled = pool_ret_aligned  # New

            # Store results
            portfolio_ret_dict[freq_name] = ret_series_resampled
            market_ret_dict[freq_name] = market_ret_resampled
            pool_ret_dict[freq_name] = pool_ret_resampled  # New

            # Calculate performance metrics
            portfolio_metrics = calculate_performance_metrics(ret_series_resampled, risk_free_rate=0.0)
            market_metrics = calculate_performance_metrics(market_ret_resampled, risk_free_rate=0.0)
            pool_metrics = calculate_performance_metrics(pool_ret_resampled, risk_free_rate=0.0)  # New

            performance_metrics[freq_name] = {
                'Portfolio': portfolio_metrics,
                'Market': market_metrics,
                'Pool': pool_metrics  # New
            }

            print(f"Completed {freq_name} rebalancing.")
        # Print performance metrics
        for freq, metrics in performance_metrics.items():
            print(f"\n=== Performance Metrics for {freq} Rebalancing ===")
            print("Portfolio:")
            print(f"  Annualized Return: {metrics['Portfolio']['Annualized Return']:.2%}")
            print(f"  Annualized Volatility: {metrics['Portfolio']['Annualized Volatility']:.2%}")
            print(f"  Sharpe Ratio: {metrics['Portfolio']['Sharpe Ratio']:.2f}")

            print("Market:")
            print(f"  Annualized Return: {metrics['Market']['Annualized Return']:.2%}")
            print(f"  Annualized Volatility: {metrics['Market']['Annualized Volatility']:.2%}")
            print(f"  Sharpe Ratio: {metrics['Market']['Sharpe Ratio']:.2f}")

            # Print metrics for the entire stock pool
            print("Pool (All Stocks):")
            print(f"  Annualized Return: {metrics['Pool']['Annualized Return']:.2%}")
            print(f"  Annualized Volatility: {metrics['Pool']['Annualized Volatility']:.2%}")
            print(f"  Sharpe Ratio: {metrics['Pool']['Sharpe Ratio']:.2f}")

        # Visualize cumulative returns
        for freq_name in rebalancing_frequencies.keys():
            plt.figure(figsize=(14, 8))

            # Define colors and line styles
            colors = {
                'Portfolio': 'blue',
                'Market': 'red',
                'Pool': 'green'  # Color for the entire stock pool
            }

            line_styles = {
                'Portfolio': '-',
                'Market': '--',
                'Pool': '-.'  # Line style for the entire stock pool
            }

            # Check if data exists
            if freq_name not in portfolio_ret_dict:
                print(f"Missing portfolio data for {freq_name} frequency, skipping plot.")
                continue

            # Get cumulative returns for the portfolio
            portfolio_cumret = (1 + portfolio_ret_dict[freq_name]).cumprod() - 1
            # Get cumulative returns for the market
            market_cumret_plot = (1 + market_ret_dict[freq_name]).cumprod() - 1
            # Get cumulative returns for the entire stock pool
            pool_cumret_plot = (1 + pool_ret_dict[freq_name]).cumprod() - 1

            # Plot cumulative returns for the portfolio
            plt.plot(portfolio_cumret.index, portfolio_cumret.values,
                     label=f"Portfolio ({freq_name} Rebalancing)", color=colors['Portfolio'],
                     linestyle=line_styles['Portfolio'])

            # Plot cumulative returns for the market
            plt.plot(market_cumret_plot.index, market_cumret_plot.values,
                     label="Market", color=colors['Market'], linestyle=line_styles['Market'])

            # Plot cumulative returns for the entire stock pool
            plt.plot(pool_cumret_plot.index, pool_cumret_plot.values,
                     label="Pool (All Stocks)", color=colors['Pool'], linestyle=line_styles['Pool'])

            plt.title(f"Market vs. Portfolio vs. Pool Cumulative Return for {freq_name} Rebalancing", fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Cumulative Return", fontsize=12)
            plt.legend()
            plt.grid(alpha=0.5)
            plt.tight_layout()
            plt.show()
