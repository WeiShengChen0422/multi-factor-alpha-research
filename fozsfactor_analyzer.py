import pandas as pd
import numpy as np
import fozsfactors
import statsmodels.api as sm
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
import scipy.stats as stats
from scipy.stats import t, skew, kurtosis

# Set font to SimHei to ensure Chinese support
rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei
rcParams['axes.unicode_minus'] = False  # Avoid issues with minus signs


class FactorAnalysisResult:
    def __init__(self, factor_data, start_date, end_date, industry_data=None, weights='Volume', quantiles=8,
                 periods=None):
        """
        Initialize the factor analysis result object

        Parameters:
        - factor_data: pd.Series, factor values, MultiIndex as date and stock tickers.
        - get_data: function, function to dynamically load data, takes stocks_name and dependencies as parameters.
        - industry_data: pd.Series, industry group information, MultiIndex consistent with factor values.
        - weights: pd.Series, weight information, MultiIndex consistent with factor values.
        - quantiles: int, number of quantiles, default is 8.
        - periods: list[int], list of holding periods (e.g., [1, 5, 21]).
        """
        if periods is None:
            periods = [1, 5, 21]
        self._factor_data = factor_data
        self._industry_data = industry_data
        self._weights = weights
        self._periods = periods
        self._quantiles = quantiles
        self.start = start_date
        self.end = end_date
        # Automatically load and calculate the following attributes
        self._price_data = self._extract_price_data()
        self._forward_returns = self._compute_forward_returns()
        self._clean_factor_data = self._compute_clean_factor_data()
        # self._return_by_quantile = self._compute_return_by_quantile()[0]
        self._mean_return_by_quantile = self._compute_return_by_quantile()
        self._market_weighted_returns = self._calc_market_returns()
        self._factor_weighted_returns = self._calculate_Factor_weighted_returns()
        self._factor_alpha_beta = self._calc_factor_alpha_beta()
        self._ic = self._calculate_daily_ic()
        self._ic_information_table = self._analyze_ic()

    def _extract_price_data(self):
        """Dynamically load price data based on factor stock column names"""
        factor_stocks = self._factor_data.columns.tolist()
        price_data = fozsfactors.get_data(stocks_name=factor_stocks, dependencies=['Close'], start_date=self.start,
                                          end_date=self.end)
        return price_data['Close']  # Return 'Close' data

    def _compute_forward_returns(self):
        """Calculate forward returns"""
        forward_returns = {}
        for period in self._periods:
            shifted_prices = self._price_data.shift(-period)
            returns = (shifted_prices / self._price_data - 1).stack()
            returns.name = f'return_{period}'
            forward_returns[f'return_{period}'] = returns
        forward_returns_df = pd.concat(forward_returns, axis=1)
        return forward_returns_df

    def _compute_clean_factor_data(self):
        """Remove NaN/Inf and organize factor data"""
        # Organize factor values and forward returns

        # combined_data.columns = ['factor_value'] + [f'return_{p}' for p in self._periods]

        # # Add industry grouping and weights
        # if self._industry_data is not None:
        #     combined_data['industry'] = self._industry_data

        # Market cap-weighted
        stocks = self._factor_data.columns.tolist()
        dependencies = fozsfactors.get_data(stocks, [self._weights], start_date=self.start, end_date=self.end)[
            self._weights]
        mcw = dependencies.div(dependencies.sum(axis=1), axis=0).reset_index().melt(id_vars='Date', var_name='Asset',
                                                                                    value_name='Weight')

        df1_melted = self._factor_data.reset_index().melt(id_vars='Date', var_name='Asset', value_name='Factor')
        df2_melted = self._forward_returns.reset_index()
        df2_melted.rename(columns={'level_1': 'Asset'}, inplace=True)

        # Merge
        combined_data = pd.merge(df1_melted, df2_melted, on=['Date', 'Asset'], how='inner')
        combined_data = pd.merge(combined_data, mcw, on=['Date', 'Asset'], how='inner')
        combined_data.sort_values('Date', inplace=True)
        combined_data.set_index(['Date', 'Asset'], inplace=True)
        # Remove NaN and Inf
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()

        # Calculate quantiles
        combined_data['Quantile'] = pd.qcut(combined_data['Factor'], q=self._quantiles, labels=False) + 1

        return combined_data

    def _compute_return_by_quantile(self):
        # Group by quantiles
        grouped = self._clean_factor_data.groupby('Quantile')

        # Calculate weighted average returns
        weighted_returns = pd.DataFrame({
            f'return_{period}': grouped.apply(lambda g: np.average(g[f'return_{period}'], weights=g['Weight']))
            for period in self._periods
        })
        # print(result)
        # Set quantile grouping as index name
        weighted_returns.index.name = 'Quantile'

        return weighted_returns

    def _calc_market_returns(self, weight_col='Weight', return_cols=None):
        """
        Calculate daily market portfolio returns.

        Returns:
        - pd.Series, daily market portfolio returns.
        """
        if return_cols is None:
            return_cols = [f'return_{period}' for period in self._periods]

        data = self._clean_factor_data.copy()
        data = data.dropna(subset=[weight_col] + return_cols)

        # Group by date
        grouped = data.groupby(level='Date')

        # Daily market portfolio returns weighted by weights
        market_returns = grouped.apply(
            lambda group: pd.Series({
                col: np.average(group[col], weights=np.abs(group[weight_col]))
                for col in return_cols
            })
        )

        return market_returns

    def _calculate_Factor_weighted_returns(self, factor_col='Factor', return_cols=None):
        """
        Calculate daily weighted returns weighted by factor values.

        Parameters:
        - data: pd.DataFrame, multi-index data containing factor values and returns.
        - factor_col: str, column name of factor values for weighting.
        - return_cols: list[str], columns of returns to calculate (e.g., ['return_1', 'return_5', 'return_21']).

        Returns:
        - pd.DataFrame, rows indexed by date, columns as weighted returns.
        """
        if return_cols is None:
            return_cols = [f'return_{period}' for period in self._periods]

        # Ensure factor values are non-null
        data = self._clean_factor_data.copy()
        data = data.dropna(subset=[factor_col] + return_cols)

        # Group by date
        grouped = data.groupby(level='Date')

        # Calculate weighted returns
        weighted_returns = grouped.apply(
            lambda group: pd.Series({
                col: np.average(group[col], weights=np.abs(group[factor_col]))
                for col in return_cols
            })
        )

        # Set column names
        weighted_returns.columns = return_cols

        return weighted_returns



    def _calc_factor_alpha_beta(self, demeaned=True, group_adjust=False):
        """
        Calculate alpha and beta for the factor.

        Parameters:
        - demeaned: bool, whether to demean.
        - group_adjust: bool, whether to adjust by industry groups.

        Returns:
        - pd.DataFrame, each row corresponds to a return column (e.g., return_1, return_5, return_10), each column is alpha or beta.
        """
        # Store results
        results = {}

        # Perform regression analysis for each column
        for period in self._factor_weighted_returns.columns:
            # Get factor returns and market returns for the current period
            factor_returns = self._factor_weighted_returns[period]
            market_returns = self._market_weighted_returns[period]

            # Regression analysis
            X = sm.add_constant(market_returns)
            model = sm.OLS(factor_returns, X, missing='drop').fit()

            # Extract regression coefficients
            alpha = model.params['const']
            beta = model.params[period]

            # Annualize alpha
            annualized_alpha = alpha * 252

            # Store results
            results[period] = {'alpha': annualized_alpha, 'beta': beta}

        # Convert to DataFrame
        return pd.DataFrame(results)

    def _calculate_daily_ic(self, factor_col='Factor', return_cols=None):
        """
        Calculate the daily information coefficient (IC) for the factor.

        Parameters:
        - data: pd.DataFrame, multi-index data containing factor values and target returns.
        - factor_col: str, column name for factor values.
        - return_cols: list[str], column names for target returns (e.g., ['return_1', 'return_5', 'return_10']).

        Returns:
        - pd.DataFrame, each row is a date, each column corresponds to the IC for a target return.
        """

        data = self._clean_factor_data

        if return_cols is None:
            return_cols = [f'return_{period}' for period in self._periods]

        # Initialize results storage
        ic_results = []

        # Group by date to calculate IC values
        for date, group in data.groupby(level='Date'):
            ic_values = {}
            for col in return_cols:
                # Calculate the rank correlation between factor values and target returns
                if group[factor_col].isnull().any() or group[col].isnull().any():
                    ic = None  # Skip groups with NaN
                else:
                    ic, _ = spearmanr(group[factor_col], group[col])
                ic_values[col] = ic
            ic_values['Date'] = date
            ic_results.append(ic_values)

        # Convert to DataFrame
        ic_df = pd.DataFrame(ic_results).set_index('Date')

        return ic_df

    def _analyze_ic(self):
        """
        Perform statistical analysis on IC data to calculate t-stat, p-value, Skew, Kurtosis.

        Parameters:
        - ic_data: pd.DataFrame, each row is a date, each column corresponds to IC for different return periods (e.g., return_1, return_5, return_21).

        Returns:
        - pd.DataFrame, each row corresponds to a return period, each column is a statistical metric.
        """
        results = {}

        for col in self._ic.columns:
            # Remove NaN values
            ic_series = self._ic[col].dropna()
            n = len(ic_series)

            # Calculate mean and standard deviation
            mean_ic = ic_series.mean()
            std_ic = ic_series.std()

            rolling_ir = ic_series.rolling(window=7).apply(lambda x: x.mean() / x.std(), raw=True).mean()

            # t-stat and p-value
            t_stat = mean_ic * np.sqrt(n) / std_ic if n > 1 else np.nan
            p_value = t.sf(np.abs(t_stat), df=n - 1) * 2 if n > 1 else np.nan

            # Skewness and Kurtosis
            skewness = skew(ic_series)
            kurt = kurtosis(ic_series)

            # Store results
            results[col] = {
                "IC mean": round(mean_ic, 8),
                "IC std": round(std_ic, 8),
                "IR": round(rolling_ir, 8),
                "t-stat": round(t_stat, 8),
                "p-value": round(p_value, 8),
                "Skew": round(skewness, 8),
                "Kurtosis": round(kurt, 8)
            }

        # Convert to DataFrame
        return pd.DataFrame(results)

    def _plot_ic_ts(self, window=30, title='IC Time Series', period_labels=None, sample_step=10):
        """
        Plot IC time series, including multiple IC data columns, moving averages, means, and variance annotations.

        Parameters:
        - window: int, moving average window size (default is 30 days).
        - title: str, chart title.
        - period_labels: list[str], labels for each IC data column, e.g., ['1 Day', '5 Days', '10 Days'].
        - sample_step: int, take every nth row to reduce density.
        """
        if period_labels is None:
            period_labels = self._ic.columns  # Use column names as labels

        # Iterate over each IC data column
        for i, col in enumerate(self._ic.columns):
            ic_data = self._ic[col]

            # Reduce density: take every nth row
            sampled_data = ic_data.iloc[::sample_step]
            moving_avg = sampled_data.rolling(window=window // sample_step).mean()

            # Calculate mean and variance
            mean_ic = ic_data.mean()
            std_ic = ic_data.var()

            # Plotting
            plt.figure(figsize=(12, 4))
            plt.plot(sampled_data.index, sampled_data, label='IC', alpha=0.6, color='blue', linewidth=1)  # Original IC
            plt.plot(moving_avg.index, moving_avg, label=f'{window} Day Moving Average', color='green', linewidth=2)  # Moving average

            # Add auxiliary lines
            plt.axhline(0, color='black', linestyle='--', linewidth=1)

            # Annotate mean and variance
            plt.text(x=sampled_data.index[int(len(sampled_data) * 0.1)], y=sampled_data.min() * 0.8,
                     s=f'Mean {mean_ic:.3f}\nVariance {std_ic:.3f}',
                     fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

            # Set title and labels
            plt.title(f'{period_labels[i]} IC', fontsize=12)
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('IC', fontsize=10)
            # Set x-axis ticks: one every 6 months
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # One major tick every 6 months

            # Legend
            plt.legend(fontsize=10, loc='upper left')

            # Beautify layout
            plt.grid(alpha=0.3)
            plt.tight_layout()

            # Show plot
            plt.show()

    def _plot_Returns_by_Quantil(self):
        # Plot bar chart
        ax = (self.mean_return_by_quantile * 10000).plot(kind='bar', figsize=(12, 4), width=0.5)
        plt.title("Returns by Quantile", fontsize=12)
        plt.xlabel("Quantile", fontsize=12)
        plt.ylabel("Mean Return(bps)", fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title="Periods", fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def _plot_ic_qq(self, group_adjust=False, method='rank', theoretical_dist='norm'):
        """
        Plot QQ plots for IC, arranged horizontally in one figure.

        Parameters:
        - group_adjust: bool, whether to perform industry-neutral adjustment.
        - method: str, IC calculation method, 'rank' or 'normal'.
        - theoretical_dist: str, theoretical distribution, default is 'norm' (normal distribution).
        """
        # Get IC data (assume self._ic is a DataFrame or Series)
        ic_data = self._ic
        num_cols = len(ic_data.columns)  # Get number of IC columns

        # Create horizontally arranged subplots
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

        # Iterate over IC columns, plotting each one in a subplot
        for i, col in enumerate(ic_data.columns):
            data = ic_data[col]
            ax = axes[i] if num_cols > 1 else axes  # Support for single-column case

            # Plot QQ plot
            stats.probplot(data, dist=theoretical_dist, plot=ax)
            ax.set_title(f'{col} IC QQ Plot ({method})', fontsize=12)
            ax.set_xlabel('Theoretical Quantiles', fontsize=10)
            ax.set_ylabel('Sample Quantiles', fontsize=10)

            # Set grid
            ax.grid(alpha=0.3)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def _plot_cumulative_returns_by_quantile(self, period=(1, 5, 10), demeaned=False, group_adjust=False):
        """
        Plot daily cumulative returns by quantile.

        Parameters:
        - period: tuple, rebalancing periods, e.g., (1, 5, 10).
        - demeaned: bool, whether to calculate cumulative returns using excess returns.
        - group_adjust: bool, whether to calculate cumulative returns using industry-neutralized returns.
        """
        # Get weighted average returns grouped by quantiles
        mean_return_by_quantile = self._return_by_quantile

        # Filter data by period
        cumulative_returns = {}
        for p in period:
            period_col = f'return_{p}'  # Column name
            if period_col in mean_return_by_quantile:
                # Calculate cumulative returns by quantiles
                cumulative_returns[p] = mean_return_by_quantile[period_col].cumsum()

        # Plot cumulative return curves
        plt.figure(figsize=(12, 6))
        for p, returns in cumulative_returns.items():
            plt.plot(returns.index, returns, label=f'Period: {p} days')

        # Legends and auxiliary lines
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title('Daily Cumulative Returns by Quantile', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Cumulative Returns', fontsize=14)
        plt.legend(title='Rebalancing Period', fontsize=12)

        # Beautify layout
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Show the plot
        plt.show()

    def create_summary_tear_sheet(self):
        print("Return Analysis: \n", self._factor_alpha_beta)
        # self._plot_cumulative_returns_by_quantile()
        self._plot_Returns_by_Quantil()
        print("IC Table: \n", self._ic_information_table)
        self._plot_ic_ts()
        self._plot_ic_qq()

    @property
    def price_data(self):
        """Access price data"""
        return self._price_data

    @property
    def forward_returns(self):
        """Access forward_returns"""
        return self._forward_returns

    @property
    def clean_factor_data(self):
        """Access clean_factor_data"""
        return self._clean_factor_data

    @property
    def mean_return_by_quantile(self):
        return self._mean_return_by_quantile

    @property
    def market_weighted_returns(self):
        return self._market_weighted_returns

    @property
    def factor_weighted_returns(self):
        return self._factor_weighted_returns

    @property
    def factor_alpha_beta(self):
        print("Return Analysis: \n")
        return self._factor_alpha_beta

    @property
    def ic_information_table(self):
        print("IC Information Table: \n")
        return self._ic_information_table
