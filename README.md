# multi-factor-alpha-research
# Multi-Factor Alpha Research

This project develops a systematic multi-factor stock selection model using U.S. equity market data and fundamental financial ratios. The goal is to identify predictive alpha signals, evaluate their robustness through factor analysis, and construct a portfolio that outperforms the S&P 500 benchmark.

## Overview

The project follows a full quantitative research pipeline:

1. Data collection and cleaning
2. Factor construction and preprocessing
3. Quantile portfolio analysis
4. Single-factor backtesting
5. Multi-factor portfolio construction
6. Benchmark comparison against the S&P 500

The final strategy combines three selected factors:

* TRIX
* Alpha 67
* Alpha 120

These factors were selected based on their predictive power, economic intuition, and backtesting performance.

## Data Sources

The project uses both market data and fundamental data.

**Market Data**

* Open
* High
* Low
* Close
* Adjusted Close
* Volume

Market data was collected using Yahoo Finance.

**Fundamental Data**

* Book-to-market ratio
* Price-to-earnings ratio
* Price-to-sales ratio
* Return on assets
* Return on equity
* Price-to-book ratio

Fundamental ratio data was obtained from WRDS.

## Methodology

### 1. Data Processing

The data processing pipeline includes:

* Collecting historical stock price and volume data
* Merging fundamental ratio data with market data
* Forward-filling monthly ratio data
* Filtering stocks based on data availability
* Handling missing values and inconsistent time ranges
* Preparing a clean stock universe for factor research and backtesting

### 2. Factor Engineering

More than 100 potential factors were explored using historical data from 2014 to 2020.

Before testing, factor values were processed using:

* **Neutralization** to reduce market capitalization effects
* **Standardization** to make factor values comparable across stocks
* **Winsorization** to reduce the impact of extreme outliers

### 3. Factor Evaluation

Each factor was evaluated through quantile portfolio analysis. Stocks were ranked by factor score and divided into 30 quantiles. Each quantile portfolio was equally weighted.

The main evaluation metrics include:

* Information Coefficient, or IC
* Information Ratio, or IR
* p-values
* Daily, weekly, and monthly portfolio returns
* Annualized return
* Annualized volatility
* Sharpe ratio

## Selected Factors

### TRIX

TRIX is a momentum-based technical indicator that applies triple exponential smoothing to closing prices. It helps filter out short-term noise and capture the underlying price trend.

In this project, TRIX showed mean-reversion behavior, so the strategy selected stocks with lower TRIX scores.

### Alpha 67

```text
EWMA(MAX(CLOSE - DELAY(CLOSE, 1), 0), 1/24)
/
EWMA(ABS(CLOSE - DELAY(CLOSE, 1)), 1/24) * 100
```

Alpha 67 measures the proportion of positive price movement relative to total price movement. The empirical results suggested a contrarian effect, where stocks with lower Alpha 67 values tended to outperform.

### Alpha 120

```text
RANK(VWAP - CLOSE) / RANK(VWAP + CLOSE)
```

Alpha 120 captures VWAP-based pricing dynamics and relative price positioning. In this project, stocks with higher Alpha 120 scores were selected.

## Backtesting

The backtest was conducted from 2021 to 2024 using three rebalancing frequencies:

* Daily rebalancing
* Weekly rebalancing
* Monthly rebalancing

The portfolio performance was compared against:

* S&P 500 benchmark
* Equal-weighted stock pool benchmark

## Single-Factor Results

### Alpha 120

| Rebalancing | Annualized Return | Annualized Volatility | Sharpe Ratio |
| ----------- | ----------------: | --------------------: | -----------: |
| Daily       |            26.05% |                23.64% |         1.10 |
| Weekly      |            23.97% |                23.26% |         1.03 |
| Monthly     |            22.66% |                23.02% |         0.98 |

### TRIX

| Rebalancing | Annualized Return | Annualized Volatility | Sharpe Ratio |
| ----------- | ----------------: | --------------------: | -----------: |
| Daily       |            31.75% |                26.86% |         1.18 |
| Weekly      |            29.08% |                26.47% |         1.10 |
| Monthly     |            27.22% |                26.62% |         1.02 |

### Alpha 67

| Rebalancing | Annualized Return | Annualized Volatility | Sharpe Ratio |
| ----------- | ----------------: | --------------------: | -----------: |
| Daily       |            19.64% |                21.68% |         0.91 |
| Weekly      |            19.09% |                20.74% |         0.92 |
| Monthly     |            21.70% |                19.89% |         1.09 |

## Multi-Factor Portfolio

The final portfolio combines stocks selected by TRIX, Alpha 67, and Alpha 120. At each rebalancing date, selected stocks are equally weighted to maintain simplicity and avoid over-concentration in any single factor.

### Multi-Factor Backtest Results

| Rebalancing | Portfolio Annualized Return | Portfolio Volatility | Portfolio Sharpe | S&P 500 Sharpe |
| ----------- | --------------------------: | -------------------: | ---------------: | -------------: |
| Daily       |                      24.12% |               21.99% |             1.10 |           0.77 |
| Weekly      |                      22.37% |               21.58% |             1.04 |           0.72 |
| Monthly     |                      23.31% |               21.42% |             1.09 |           0.77 |

## Key Findings

* The selected factors outperformed the S&P 500 benchmark in both return and Sharpe ratio.
* TRIX achieved the strongest single-factor performance.
* Alpha 120 showed consistent performance across different rebalancing frequencies.
* Alpha 67 performed best under monthly rebalancing.
* The multi-factor portfolio achieved a daily Sharpe ratio of 1.10, compared with 0.77 for the S&P 500.
* Combining multiple factors improved diversification across different alpha sources.

## Technologies Used

* Python
* pandas
* NumPy
* yfinance
* WRDS
* Matplotlib
* Factor modeling
* Portfolio backtesting

## Future Improvements

Potential extensions include:

* Incorporating transaction costs and slippage
* Adding turnover and maximum drawdown analysis
* Testing sector and industry neutralization
* Expanding the stock universe
* Applying walk-forward validation
* Building long-short factor portfolios
* Testing machine learning models for factor combination

## Conclusion

This project demonstrates how a systematic multi-factor framework can be used to identify predictive equity signals and construct portfolios with stronger risk-adjusted performance than the market benchmark. By combining momentum and mean-reversion factors, the final strategy achieved consistent outperformance across daily, weekly, and monthly rebalancing frequencies.
