import fozsfactors as fo
import numpy as np
import pandas as pd


class PriceMomentum(fo.Factor):
    name = 'price_momentum'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        close = data_['Close']
        return_12m = close.pct_change(periods=252)
        return_1m = close.pct_change(periods=21)
        momentum = return_12m - return_1m
        return momentum


class Alpha41(fo.Factor):
    name = 'alpha41'
    dependencies = ['High', 'Low', 'Close', 'Volume']

    def Calculation_Process(self, data_):
        typical_price = (data_['High'] + data_['Low'] + data_['Close']) / 3
        vwap = (typical_price * data_['Volume']).cumsum() / data_['Volume'].cumsum()
        delta_vwap = vwap.diff(3)
        max_delta = delta_vwap.rolling(window=5).max()
        rank = max_delta.rank(pct=True)
        alpha = rank * -1
        return alpha


class Alpha52(fo.Factor):
    name = 'alpha52'
    dependencies = ['High', 'Low', 'Close']

    def Calculation_Process(self, data_):
        hlc3 = (data_['High'] + data_['Low'] + data_['Close']) / 3
        delay_hlc3 = hlc3.shift(1)
        max_part1 = np.maximum(0, data_['High'] - delay_hlc3)
        sum_max1 = max_part1.rolling(window=26).sum()
        max_part2 = np.maximum(0, delay_hlc3 - data_['Low'])
        sum_max2 = max_part2.rolling(window=26).sum()
        alpha = (sum_max1 / (sum_max2 + 1e-10)) * 100
        return alpha


class Alpha53(fo.Factor):
    name = 'alpha53'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        condition = data_['Close'] > data_['Close'].shift(1)
        count = condition.rolling(window=12).sum()
        alpha = (count / 12) * 100
        return alpha


class Alpha65(fo.Factor):
    name = 'alpha65'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        mean_close_6 = data_['Close'].rolling(window=6).mean()
        alpha = mean_close_6 / data_['Close']
        return alpha


class Alpha66(fo.Factor):
    name = 'alpha66'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        mean_close_6 = data_['Close'].rolling(window=6).mean()
        alpha = (data_['Close'] - mean_close_6) / mean_close_6 * 100
        return alpha


class MA5(fo.Factor):
    name = 'ma5'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        return data_['Close'].rolling(5).mean()


class RUMI(fo.Factor):
    name = 'rumi'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        MA_short = data_['Close'].rolling(3).mean()
        MA_long = data_['Close'].rolling(50).mean()
        return (MA_short - MA_long).rolling(20).mean()


class Alpha1(fo.Factor):
    name = 'a1'
    dependencies = ['Close', 'Volume', 'Open']

    def Calculation_Process(self, data_):
        rank_delta_log_volume = np.log(data_['Volume']).diff(1).rank(pct=True)
        rank_returns = (data_['Close'] - data_['Open']) / data_['Open'].rank(pct=True)

        def rolling_corr(s1, s2, window):
            return s1.rolling(window).corr(s2)

        corr = rolling_corr(rank_delta_log_volume, rank_returns, 6)

        return -corr


class Alpha2(fo.Factor):
    name = 'a2'
    dependencies = ['Close', 'High', 'Low']

    def Calculation_Process(self, data_):
        return -1 * (((data_['Close'] - data_['Low']) - (data_['High'] - data_['Close'])) / (
                data_['High'] - data_['Low'])).diff(1)


class Alpha9(fo.Factor):
    name = 'a9'
    dependencies = ['Volume', 'High', 'Low']

    def Calculation_Process(self, data_):
        current_mid = (data_['High'] + data_['Low']) / 2

        previous_mid = (data_['High'].shift(1) + data_['Low'].shift(1)) / 2

        factor = (current_mid - previous_mid) * (data_['High'] - data_['Low']) / data_['Volume']

        def weighted_sma(series, window, weight):
            weights = [weight ** i for i in range(window)][::-1]
            sma = series.rolling(window).apply(lambda x: sum(w * val for w, val in zip(weights, x)) / sum(weights),
                                               raw=False)
            return sma

        return -weighted_sma(factor, 7, 2)


class Alpha13(fo.Factor):
    name = 'a13'
    dependencies = ['Volume', 'High', 'Low', 'Close']

    def Calculation_Process(self, data_):
        return (data_['High'] * data_['Low']) ** 0.5 - (data_['Close'] * data_['Volume']).cumsum() / data_[
            'Volume'].cumsum()


class Alpha34(fo.Factor):
    name = 'a34'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        return data_['Close'].rolling(12).mean() / data_['Close']


class TRIX(fo.Factor):
    name = 'trix'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        close_ema1 = data_['Close'].ewm(span=15, adjust=False).mean()
        close_ema2 = close_ema1.ewm(span=15, adjust=False).mean()
        close_ema3 = close_ema2.ewm(span=15, adjust=False).mean()
        trix = close_ema3.pct_change() * 100
        return trix


class Alpha67(fo.Factor):
    name = 'alpha67'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        diff_close = data_['Close'] - data_['Close'].shift(1)
        max_diff = np.maximum(diff_close, 0)
        abs_diff = abs(diff_close)
        sma_max = max_diff.ewm(alpha=1 / 24, adjust=False).mean()
        sma_abs = abs_diff.ewm(alpha=1 / 24, adjust=False).mean()
        alpha = (sma_max / sma_abs) * 100
        return alpha


class Alpha101(fo.Factor):
    name = 'Alpha101'
    dependencies = ['Close', 'Volume', 'High', 'Low']  # VWAP will be computed if missing

    def Calculation_Process(self, data_):
        # Combine dependencies into a single DataFrame
        data_combined = pd.concat(data_.values(), axis=1, keys=data_.keys())
        data_combined.columns = data_combined.columns.droplevel(0)  # Flatten MultiIndex

        # Compute VWAP if not present
        if 'VWAP' not in data_combined.columns:
            typical_price = (data_combined['High'] + data_combined['Low'] + data_combined['Close']) / 3
            price_volume = typical_price * data_combined['Volume']
            cumulative_price_volume = price_volume.cumsum()
            cumulative_volume = data_combined['Volume'].cumsum()
            data_combined['VWAP'] = cumulative_price_volume / cumulative_volume

        # Extract necessary data
        close = data_combined['Close']
        volume = data_combined['Volume']
        high = data_combined['High']
        vwap = data_combined['VWAP']

        # Step 1: Compute rolling mean and sum
        mean_volume_30 = volume.rolling(30).mean()
        sum_mean_volume_37 = mean_volume_30.rolling(37).sum()

        # Step 2: Compute 15-day rolling correlation
        corr1 = close.rolling(15).corr(sum_mean_volume_37)

        # Step 3: Weighted price and rankings
        weighted_price = (high * 0.1) + (vwap * 0.9)
        rank_weighted_price = weighted_price.rank(axis=1)
        rank_volume = volume.rank(axis=1)

        # Step 4: 11-day rolling correlation of ranked data
        corr2 = rank_weighted_price.rolling(11).corr(rank_volume)

        # Step 5: Rank correlations and calculate alpha signal
        rank_corr1 = corr1.rank(axis=1)
        rank_corr2 = corr2.rank(axis=1)
        alpha_signal = ((rank_corr1 < rank_corr2) * -1).fillna(0)

        return alpha_signal


class Alpha102(fo.Factor):
    name = 'Alpha102'
    dependencies = ['Volume']

    def Calculation_Process(self, data_):
        volume = data_['Volume']
        volume_diff = volume.diff(1)
        positive_diff = volume_diff.clip(lower=0)
        abs_diff = volume_diff.abs()

        sma_positive = positive_diff.rolling(window=6).mean()
        sma_abs = abs_diff.rolling(window=6).mean()

        return (sma_positive / sma_abs) * -100


class Alpha103(fo.Factor):
    name = 'Alpha103'
    dependencies = ['Low']

    def Calculation_Process(self, data_):
        low = data_['Low']
        lowday_20 = low.rolling(window=20).apply(lambda x: (len(x) - 1) - x[::-1].argmin(), raw=True)
        return ((20 - lowday_20) / 20) * 100


class Alpha104(fo.Factor):
    name = 'Alpha104'
    dependencies = ['High', 'Volume', 'Close']

    def Calculation_Process(self, data_):
        high = data_['High']
        volume = data_['Volume']
        close = data_['Close']
        corr_high_volume = high.rolling(window=5).corr(volume)

        # Step 2: Compute the 5-period change (delta) of the rolling correlation
        delta_corr = corr_high_volume.diff(periods=5)

        # Step 3: Calculate 20-period rolling standard deviation of CLOSE
        std_close = close.rolling(window=20).std()

        # Step 4: Rank the rolling standard deviation (normalize to [0, 1] rank)
        rank_std_close = std_close.rank(pct=True)

        # Step 5: Combine the components and apply the formula
        alpha = -1 * (delta_corr * rank_std_close)

        return alpha


class Alpha105(fo.Factor):
    name = 'Alpha105'
    dependencies = ['Open', 'Volume']

    def Calculation_Process(self, data_):
        open_prices = data_['Open']
        volume = data_['Volume']

        # Step 1: Rank the Open prices and Volume
        rank_open = open_prices.rank(axis=0, pct=True)  # Rank normalized to [0, 1]
        rank_volume = volume.rank(axis=0, pct=True)  # Rank normalized to [0, 1]

        # Step 2: Compute the rolling correlation between ranked Open and Volume over a 10-period window
        corr_rank = rank_open.rolling(window=10).corr(rank_volume)

        # Step 3: Apply the negative sign
        alpha = -1 * corr_rank

        return alpha


class Alpha106(fo.Factor):
    name = 'Alpha106'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        close = data_['Close']
        delayed_close = close.shift(20)
        return close - delayed_close


class Alpha107(fo.Factor):
    name = 'Alpha107'
    dependencies = ['Open', 'High', 'Close', 'Low']

    def Calculation_Process(self, data_):
        open_prices = data_['Open']
        high = data_['High']
        close = data_['Close']
        low = data_['Low']

        # Step 1: Calculate differences with delayed values
        diff_open_high = open_prices - high.shift(1)  # OPEN - DELAY(HIGH, 1)
        diff_open_close = open_prices - close.shift(1)  # OPEN - DELAY(CLOSE, 1)
        diff_open_low = open_prices - low.shift(1)  # OPEN - DELAY(LOW, 1)

        # Step 2: Rank each difference (normalized to [0, 1])
        rank_open_high = diff_open_high.rank(axis=0, pct=True)
        rank_open_close = diff_open_close.rank(axis=0, pct=True)
        rank_open_low = diff_open_low.rank(axis=0, pct=True)

        # Step 3: Multiply ranks with a negative sign
        alpha = (-1 * rank_open_high) * rank_open_close * rank_open_low

        return alpha


class Alpha108(fo.Factor):
    name = 'Alpha108'
    dependencies = ['High', 'Low', 'Close', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        close = data_['Close']
        volume = data_['Volume']

        # Step 1: Calculate VWAP manually
        price = (high + low + close) / 3  # Average price
        vwap = (price * volume).cumsum() / volume.cumsum()  # VWAP calculation

        # Step 2: Compute HIGH - MIN(HIGH, 2)
        high_min_diff = high - high.rolling(window=2).min()

        # Step 3: Rank HIGH - MIN(HIGH, 2)
        rank_high_min_diff = high_min_diff.rank(axis=0, pct=True)

        # Step 4: Compute the 120-period mean of Volume
        volume_mean_120 = volume.rolling(window=120).mean()

        # Step 5: Compute the 6-period correlation between VWAP and MEAN(VOLUME, 120)
        corr_vwap_volume = vwap.rolling(window=6).corr(volume_mean_120)

        # Step 6: Rank the correlation
        rank_corr_vwap_volume = corr_vwap_volume.rank(axis=0, pct=True)

        # Step 7: Raise the rank of the difference to the power of the rank of the correlation
        alpha = (rank_high_min_diff ** rank_corr_vwap_volume) * -1

        return alpha


class Alpha109(fo.Factor):
    name = 'Alpha109'
    dependencies = ['High', 'Low']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']

        # Step 1: Calculate HIGH - LOW
        high_low_diff = high - low

        # Step 2: Compute the 10-period SMA with smoothing factor 2 (using exponential smoothing)
        sma_high_low = high_low_diff.rolling(window=10).mean()

        # Step 3: Compute SMA of the previously computed SMA (10-period smoothing factor 2)
        double_sma_high_low = sma_high_low.rolling(window=10).mean()

        # Step 4: Calculate the ratio
        alpha = sma_high_low / double_sma_high_low

        return alpha


class Alpha110(fo.Factor):
    name = 'Alpha110'
    dependencies = ['High', 'Close', 'Low']

    def Calculation_Process(self, data_):
        high = data_['High']
        close = data_['Close']
        low = data_['Low']

        # Step 1: Calculate the delayed Close price
        delayed_close = close.shift(1)  # DELAY(CLOSE, 1)

        # Step 2: Compute MAX(0, HIGH - DELAY(CLOSE, 1))
        positive_diff_high = (high - delayed_close).clip(lower=0)

        # Step 3: Compute MAX(0, DELAY(CLOSE, 1) - LOW)
        positive_diff_low = (delayed_close - low).clip(lower=0)

        # Step 4: Calculate the 20-period rolling sum for both terms
        sum_positive_high = positive_diff_high.rolling(window=20).sum()
        sum_positive_low = positive_diff_low.rolling(window=20).sum()

        # Step 5: Compute the ratio and scale by 100
        alpha = (sum_positive_high / sum_positive_low) * 100

        return alpha


class Alpha111(fo.Factor):
    name = 'Alpha111'
    dependencies = ['Volume', 'Close', 'High', 'Low']

    def Calculation_Process(self, data_):
        volume = data_['Volume']
        close = data_['Close']
        high = data_['High']
        low = data_['Low']

        # Step 1: Calculate the core metric
        price_metric = volume * ((close - low) - (high - close)) / (high - low)

        # Step 2: Calculate the 11-period smoothed moving average (SMA with smoothing factor 2)
        sma_11 = price_metric.rolling(window=11).mean()

        # Step 3: Calculate the 4-period smoothed moving average (SMA with smoothing factor 2)
        sma_4 = price_metric.rolling(window=4).mean()

        # Step 4: Compute the difference between the two SMAs
        alpha = sma_11 - sma_4

        return alpha


class Alpha112(fo.Factor):
    name = 'Alpha112'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        close = data_['Close']

        # Step 1: Calculate DELAY(CLOSE, 1)
        delayed_close = close.shift(1)

        # Step 2: Calculate CLOSE - DELAY(CLOSE, 1)
        diff_close = close - delayed_close

        # Step 3: Calculate MAX(0, CLOSE - DELAY(CLOSE, 1)) (positive differences)
        positive_diff = diff_close.clip(lower=0)

        # Step 4: Calculate MAX(0, ABS(CLOSE - DELAY(CLOSE, 1)) * (CLOSE - DELAY(CLOSE, 1) < 0))
        negative_diff = (diff_close.where(diff_close < 0).abs()).fillna(0)

        # Step 5: Compute 12-period rolling sums
        sum_positive_diff = positive_diff.rolling(window=12).sum()
        sum_negative_diff = negative_diff.rolling(window=12).sum()

        # Step 6: Compute the final formula
        alpha = ((sum_positive_diff - sum_negative_diff) /
                 (sum_positive_diff + sum_negative_diff)) * 100

        return alpha


class Alpha113(fo.Factor):
    name = 'Alpha113'
    dependencies = ['Close', 'Volume']

    def Calculation_Process(self, data_):
        close = data_['Close']
        volume = data_['Volume']

        # Step 1: Compute DELAY(CLOSE, 5)
        delayed_close = close.shift(5)

        # Step 2: Compute SUM(DELAY(CLOSE, 5), 20) / 20
        sum_delayed_close_20 = delayed_close.rolling(window=20).sum()
        avg_delayed_close_20 = sum_delayed_close_20 / 20

        # Step 3: Rank the average
        rank_avg_delayed_close = avg_delayed_close_20.rank(axis=0, pct=True)

        # Step 4: Compute CORR(CLOSE, VOLUME, 2)
        corr_close_volume = close.rolling(window=2).corr(volume)

        # Step 5: Compute SUM(CLOSE, 5) and SUM(CLOSE, 20)
        sum_close_5 = close.rolling(window=5).sum()
        sum_close_20 = close.rolling(window=20).sum()

        # Step 6: Compute CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2)
        corr_sum_close = sum_close_5.rolling(window=2).corr(sum_close_20)

        # Step 7: Rank the correlation
        rank_corr_sum_close = corr_sum_close.rank(axis=0, pct=True)

        # Step 8: Combine the components
        alpha = -1 * (rank_avg_delayed_close * corr_close_volume * rank_corr_sum_close)

        return alpha


class Alpha114(fo.Factor):
    name = 'Alpha114'
    dependencies = ['High', 'Low', 'Close', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        close = data_['Close']
        volume = data_['Volume']

        price = (high + low + close) / 3  # Average price
        vwap = (price * volume).cumsum() / volume.cumsum()  # VWAP calculation

        # Step 1: Compute (HIGH - LOW) / (SUM(CLOSE, 5) / 5)
        avg_close_5 = close.rolling(window=5).mean()
        price_range_ratio = (high - low) / avg_close_5

        # Step 2: Delay the above ratio by 2 periods
        delayed_ratio = price_range_ratio.shift(2)

        # Step 3: Rank the delayed ratio
        rank_delayed_ratio = delayed_ratio.rank(axis=0, pct=True)

        # Step 4: Rank the rank of VOLUME
        rank_volume = volume.rank(axis=0, pct=True)
        rank_of_rank_volume = rank_volume.rank(axis=0, pct=True)

        # Step 5: Compute the denominator ((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)
        denominator = price_range_ratio / (vwap - close)

        # Step 6: Combine the components
        alpha = (rank_delayed_ratio * rank_of_rank_volume) / denominator

        return alpha


class Alpha115(fo.Factor):
    name = 'Alpha115'
    dependencies = ['High', 'Close', 'Low', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        close = data_['Close']
        low = data_['Low']
        volume = data_['Volume']

        # Step 1: Compute ((HIGH * 0.9) + (CLOSE * 0.1))
        weighted_price = (high * 0.9) + (close * 0.1)

        # Step 2: Compute MEAN(VOLUME, 30)
        mean_volume_30 = volume.rolling(window=30).mean()

        # Step 3: Compute CORR(weighted_price, mean_volume_30, 10)
        corr_weighted_mean_volume = weighted_price.rolling(window=10).corr(mean_volume_30)

        # Step 4: Rank the correlation
        rank_corr_weighted = corr_weighted_mean_volume.rank(axis=0, pct=True)

        # Step 5: Compute (HIGH + LOW) / 2
        average_price = (high + low) / 2

        # Step 6: Compute TSRANK of average_price and volume
        tsrank_average_price = average_price.rolling(window=4).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)
        tsrank_volume = volume.rolling(window=10).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)

        # Step 7: Compute CORR(TSRANK(average_price), TSRANK(volume), 7)
        corr_tsrank = tsrank_average_price.rolling(window=7).corr(tsrank_volume)

        # Step 8: Rank the correlation
        rank_corr_tsrank = corr_tsrank.rank(axis=0, pct=True)

        # Step 9: Combine the components
        alpha = rank_corr_weighted ** rank_corr_tsrank

        return alpha


class Alpha116(fo.Factor):
    name = 'Alpha116'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        close = data_['Close']

        # Step 1: Generate the sequence (1, 2, ..., 20)
        sequence = np.arange(1, 21)

        # Step 2: Define a function to compute the regression beta
        def compute_regbeta(y):
            if len(y) != len(sequence):
                return np.nan
            X = np.vstack([sequence, np.ones(len(sequence))]).T  # Add intercept
            beta = np.linalg.lstsq(X, y, rcond=None)[0][0]  # Linear regression, extract beta
            return beta

        # Step 3: Apply rolling regression to CLOSE with a 20-period window
        alpha = close.rolling(window=20).apply(compute_regbeta, raw=False)

        return alpha


class Alpha117(fo.Factor):
    name = 'Alpha117'
    dependencies = ['Volume', 'Close', 'High', 'Low']

    def Calculation_Process(self, data_):
        volume = data_['Volume']
        close = data_['Close']
        high = data_['High']
        low = data_['Low']

        # Step 1: Compute Return manually
        ret = (close / close.shift(1)) - 1  # DELAY(CLOSE, 1)

        # Step 2: Compute TSRANK for VOLUME over a 32-period window
        tsrank_volume = volume.rolling(window=32).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)

        # Step 3: Compute TSRANK for ((CLOSE + HIGH) - LOW) over a 16-period window
        price_diff = (close + high) - low
        tsrank_price_diff = price_diff.rolling(window=16).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)

        # Step 4: Compute TSRANK for RET over a 32-period window
        tsrank_ret = ret.rolling(window=32).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)

        # Step 5: Combine the components
        alpha = tsrank_volume * (1 - tsrank_price_diff) * (1 - tsrank_ret)

        return alpha


class Alpha118(fo.Factor):
    name = 'Alpha118'
    dependencies = ['High', 'Open', 'Low']

    def Calculation_Process(self, data_):
        high = data_['High']
        open_ = data_['Open']
        low = data_['Low']

        # Step 1: Compute HIGH - OPEN
        high_minus_open = high - open_

        # Step 2: Compute OPEN - LOW
        open_minus_low = open_ - low

        # Step 3: Calculate the 20-period rolling sums
        sum_high_minus_open = high_minus_open.rolling(window=20).sum()
        sum_open_minus_low = open_minus_low.rolling(window=20).sum()

        # Step 4: Compute the ratio and scale by 100
        alpha = (sum_high_minus_open / sum_open_minus_low) * 100

        return alpha


class Alpha119(fo.Factor):
    name = 'Alpha119'
    dependencies = ['High', 'Low', 'Close', 'Open', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        close = data_['Close']
        open_ = data_['Open']
        volume = data_['Volume']

        # Step 1: Calculate VWAP manually
        price = (high + low + close) / 3
        vwap = (price * volume).cumsum() / volume.cumsum()

        # Step 2: Compute MEAN(VOLUME, 5)
        mean_volume_5 = volume.rolling(window=5).mean()

        # Step 3: Compute SUM(MEAN(VOLUME, 5), 26)
        sum_mean_volume_5_26 = mean_volume_5.rolling(window=26).sum()

        # Step 4: Compute CORR(VWAP, SUM(MEAN(VOLUME, 5), 26), 5)
        corr_vwap_sum_volume = vwap.rolling(window=5).corr(sum_mean_volume_5_26)

        # Step 5: Apply DECAYLINEAR to CORR result over a 7-period window
        decay_corr = corr_vwap_sum_volume.rolling(window=7).apply(
            lambda x: np.dot(x[::-1], np.linspace(1, len(x), len(x))) / np.sum(np.linspace(1, len(x), len(x))),
            raw=True
        )

        # Step 6: Rank the DECAYLINEAR result
        rank_decay_corr = decay_corr.rank(axis=0, pct=True)

        # Step 7: Compute MEAN(VOLUME, 15)
        mean_volume_15 = volume.rolling(window=15).mean()

        # Step 8: Compute RANK(OPEN) and RANK(MEAN(VOLUME, 15))
        rank_open = open_.rank(axis=0, pct=True)
        rank_mean_volume_15 = mean_volume_15.rank(axis=0, pct=True)

        # Step 9: Compute CORR(RANK(OPEN), RANK(MEAN(VOLUME, 15)), 21)
        corr_rank_open_volume = rank_open.rolling(window=21).corr(rank_mean_volume_15)

        # Step 10: Compute MIN(CORR(...), 9)
        min_corr_9 = corr_rank_open_volume.rolling(window=9).min()

        # Step 11: Apply TSRANK to MIN(CORR(...), 9) over a 7-period window
        tsrank_min_corr = min_corr_9.rolling(window=7).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)

        # Step 12: Apply DECAYLINEAR to TSRANK result over an 8-period window
        decay_tsrank = tsrank_min_corr.rolling(window=8).apply(
            lambda x: np.dot(x[::-1], np.linspace(1, len(x), len(x))) / np.sum(np.linspace(1, len(x), len(x))),
            raw=True
        )

        # Step 13: Rank the DECAYLINEAR TSRANK result
        rank_decay_tsrank = decay_tsrank.rank(axis=0, pct=True)

        # Step 14: Combine the components
        alpha = rank_decay_corr - rank_decay_tsrank

        return alpha


class Alpha120(fo.Factor):
    name = 'Alpha120'
    dependencies = ['High', 'Low', 'Close', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        close = data_['Close']
        volume = data_['Volume']

        # Step 1: Calculate VWAP manually
        price = (high + low + close) / 3
        vwap = (price * volume).cumsum() / volume.cumsum()

        # Step 2: Compute (VWAP - CLOSE) and (VWAP + CLOSE)
        vwap_minus_close = vwap - close
        vwap_plus_close = vwap + close

        # Step 3: Create cumulative ranks for each date
        rank_vwap_minus_close = vwap_minus_close.expanding().apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        rank_vwap_plus_close = vwap_plus_close.expanding().apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        # Step 4: Compute the ratio
        alpha = rank_vwap_minus_close / rank_vwap_plus_close

        return alpha


class Alpha121(fo.Factor):
    name = 'Alpha121'
    dependencies = ['High', 'Low', 'Close', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        close = data_['Close']
        volume = data_['Volume']

        # Step 1: Calculate VWAP manually
        price = (high + low + close) / 3
        vwap = (price * volume).cumsum() / volume.cumsum()

        # Step 2: Compute (VWAP - MIN(VWAP, 12))
        min_vwap_12 = vwap.rolling(window=12).min()
        vwap_minus_min_vwap = vwap - min_vwap_12

        # Step 3: Rank (VWAP - MIN(VWAP, 12))
        rank_vwap_diff = vwap_minus_min_vwap.rank(axis=0, pct=True)

        # Step 4: Compute MEAN(VOLUME, 60)
        mean_volume_60 = volume.rolling(window=60).mean()

        # Step 5: Compute TSRANK(VWAP, 20) and TSRANK(MEAN(VOLUME, 60), 2)
        tsrank_vwap_20 = vwap.rolling(window=20).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)
        tsrank_mean_volume_2 = mean_volume_60.rolling(window=2).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)

        # Step 6: Compute CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME, 60), 2), 18)
        corr_tsrank = tsrank_vwap_20.rolling(window=18).corr(tsrank_mean_volume_2)

        # Step 7: Apply TSRANK to CORR result over a 3-period window
        tsrank_corr = corr_tsrank.rolling(window=3).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)

        # Step 8: Raise the rank to the power of TSRANK
        alpha = (rank_vwap_diff ** tsrank_corr) * -1

        return alpha


class Alpha122(fo.Factor):
    name = 'Alpha122'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        close = data_['Close']

        # Step 1: Compute the logarithm of CLOSE
        log_close = np.log(close)

        # Step 2: Apply the first SMA (13-period, smoothing factor 2)
        sma_1 = log_close.rolling(window=13).mean()

        # Step 3: Apply the second SMA
        sma_2 = sma_1.rolling(window=13).mean()

        # Step 4: Apply the third SMA
        sma_3 = sma_2.rolling(window=13).mean()

        # Step 5: Compute the delayed SMA (shifted by 1 period)
        delayed_sma = sma_3.shift(1)

        # Step 6: Compute the numerator: SMA - DELAY(SMA, 1)
        numerator = sma_3 - delayed_sma

        # Step 7: Compute the denominator: DELAY(SMA, 1)
        denominator = delayed_sma

        # Step 8: Compute the final Alpha122
        alpha = numerator / denominator

        return alpha


class Alpha123(fo.Factor):
    name = 'Alpha123'
    dependencies = ['High', 'Low', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        volume = data_['Volume']

        # Step 1: Compute the midpoint (HIGH + LOW) / 2
        midpoint = (high + low) / 2

        # Step 2: Compute the 20-period SUM of midpoint
        sum_midpoint_20 = midpoint.rolling(window=20).sum()

        # Step 3: Compute the 60-period MEAN of VOLUME
        mean_volume_60 = volume.rolling(window=60).mean()

        # Step 4: Compute the 20-period SUM of the MEAN VOLUME
        sum_mean_volume_20 = mean_volume_60.rolling(window=20).sum()

        # Step 5: Compute CORR(SUM(midpoint, 20), SUM(mean_volume, 20), 9)
        corr_sum_midpoint_volume = sum_midpoint_20.rolling(window=9).corr(sum_mean_volume_20)

        # Step 6: Rank the correlation result
        rank_corr_midpoint_volume = corr_sum_midpoint_volume.rank(axis=0, pct=True)

        # Step 7: Compute CORR(LOW, VOLUME, 6)
        corr_low_volume = low.rolling(window=6).corr(volume)

        # Step 8: Rank the correlation result
        rank_corr_low_volume = corr_low_volume.rank(axis=0, pct=True)

        # Step 9: Compare ranks and compute the final Alpha123
        alpha = (rank_corr_midpoint_volume < rank_corr_low_volume).astype(float) * -1

        return alpha


class Alpha124(fo.Factor):
    name = 'Alpha124'
    dependencies = ['High', 'Low', 'Close', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        close = data_['Close']
        volume = data_['Volume']

        # Step 1: Calculate VWAP manually
        price = (high + low + close) / 3
        vwap = (price * volume).cumsum() / volume.cumsum()

        # Step 2: Compute TSMAX(CLOSE, 30)
        tsmax_close_30 = close.rolling(window=30).max()

        # Step 3: Rank TSMAX(CLOSE, 30)
        rank_tsmax_close = tsmax_close_30.rank(axis=0, pct=True)

        # Step 4: Apply DECAYLINEAR with a 2-period window
        decay_linear = rank_tsmax_close.rolling(window=2).apply(
            lambda x: np.dot(x[::-1], np.linspace(1, len(x), len(x))) / np.sum(np.linspace(1, len(x), len(x))),
            raw=True
        )

        # Step 5: Compute the numerator: CLOSE - VWAP
        numerator = close - vwap

        # Step 6: Compute the denominator: DECAYLINEAR(...)
        denominator = decay_linear

        # Step 7: Compute Alpha124
        alpha = numerator / denominator

        return alpha


class Alpha125(fo.Factor):
    name = 'Alpha125'
    dependencies = ['High', 'Low', 'Close', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        close = data_['Close']
        volume = data_['Volume']

        # Step 1: Calculate VWAP manually
        price = (high + low + close) / 3
        vwap = (price * volume).cumsum() / volume.cumsum()

        # Step 2: Compute MEAN(VOLUME, 80)
        mean_volume_80 = volume.rolling(window=80).mean()

        # Step 3: Compute CORR(VWAP, MEAN(VOLUME, 80), 17)
        corr_vwap_volume = vwap.rolling(window=17).corr(mean_volume_80)

        # Step 4: Apply DECAYLINEAR to CORR result over a 20-period window
        decay_corr = corr_vwap_volume.rolling(window=20).apply(
            lambda x: np.dot(x[::-1], np.linspace(1, len(x), len(x))) / np.sum(np.linspace(1, len(x), len(x))),
            raw=True
        )

        # Step 5: Rank the DECAYLINEAR result
        rank_decay_corr = decay_corr.rank(axis=0, pct=True)

        # Step 6: Compute DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3)
        weighted_price = (close * 0.5) + (vwap * 0.5)
        delta_weighted_price = weighted_price.diff(periods=3)

        # Step 7: Apply DECAYLINEAR to DELTA result over a 16-period window
        decay_delta = delta_weighted_price.rolling(window=16).apply(
            lambda x: np.dot(x[::-1], np.linspace(1, len(x), len(x))) / np.sum(np.linspace(1, len(x), len(x))),
            raw=True
        )

        # Step 8: Rank the DECAYLINEAR DELTA result
        rank_decay_delta = decay_delta.rank(axis=0, pct=True)

        # Step 9: Compute the final Alpha125
        alpha = rank_decay_corr / rank_decay_delta

        return alpha


class Alpha126(fo.Factor):
    name = 'Alpha126'
    dependencies = ['Close', 'High', 'Low']

    def Calculation_Process(self, data_):
        close = data_['Close']
        high = data_['High']
        low = data_['Low']

        # Calculate the average of CLOSE, HIGH, and LOW
        alpha = (close + high + low) / 3

        return alpha


class Alpha127(fo.Factor):
    name = 'Alpha127'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        close = data_['Close']

        # Step 1: Compute MAX(CLOSE, 12)
        max_close_12 = close.rolling(window=12).max()

        # Step 2: Compute the squared term
        squared_term = (100 * (close - max_close_12) / max_close_12) ** 2

        # Step 3: Compute the rolling mean of the squared term
        mean_squared = squared_term.rolling(window=12).mean()

        # Step 4: Compute the square root of the mean
        alpha = mean_squared ** 0.5

        return alpha


class Alpha128(fo.Factor):
    name = 'Alpha128'
    dependencies = ['High', 'Low', 'Close', 'Volume']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        close = data_['Close']
        volume = data_['Volume']

        # Step 1: Compute PRICE = (HIGH + LOW + CLOSE) / 3
        price = (high + low + close) / 3

        # Step 2: Compute DELAY(PRICE, 1)
        delayed_price = price.shift(1)

        # Step 3: Compute SUM for upward and downward movements over a 14-period window
        upward_sum = ((price > delayed_price) * (price * volume)).rolling(window=14).sum()
        downward_sum = ((price < delayed_price) * (price * volume)).rolling(window=14).sum()

        # Step 4: Compute the ratio of upward to downward movements
        ratio = upward_sum / downward_sum

        # Step 5: Compute Alpha128
        alpha = 100 - (100 / (1 + ratio))

        return alpha


class Alpha129(fo.Factor):
    name = 'Alpha129'
    dependencies = ['Close']

    def Calculation_Process(self, data_):
        close = data_['Close']

        # Step 1: Compute DELAY(CLOSE, 1)
        delayed_close = close.shift(1)

        # Step 2: Compute the absolute difference where CLOSE - DELAY(CLOSE, 1) < 0
        negative_diff = (close - delayed_close).where(close - delayed_close < 0).abs().fillna(0)

        # Step 3: Compute the 12-period rolling sum of the negative differences
        alpha = negative_diff.rolling(window=12).sum()

        return alpha


class Alpha130(fo.Factor):
    name = 'Alpha130'
    dependencies = ['High', 'Low', 'Volume', 'Close']

    def Calculation_Process(self, data_):
        high = data_['High']
        low = data_['Low']
        close = data_['Close']
        volume = data_['Volume']

        # Step 1: Calculate VWAP manually
        price = (high + low + close) / 3
        vwap = (price * volume).cumsum() / volume.cumsum()

        # Step 2: Compute MEAN(VOLUME, 40)
        mean_volume_40 = volume.rolling(window=40).mean()

        # Step 3: Compute CORR((HIGH + LOW) / 2, MEAN(VOLUME, 40), 9)
        midpoint = (high + low) / 2
        corr_midpoint_volume = midpoint.rolling(window=9).corr(mean_volume_40)

        # Step 4: Apply DECAYLINEAR to CORR over a 10-period window
        decay_corr_midpoint = corr_midpoint_volume.rolling(window=10).apply(
            lambda x: np.dot(x[::-1], np.linspace(1, len(x), len(x))) / np.sum(np.linspace(1, len(x), len(x))),
            raw=True
        )

        # Step 5: Rank the DECAYLINEAR result
        rank_decay_midpoint = decay_corr_midpoint.rank(axis=0, pct=True)

        # Step 6: Compute RANK(VWAP) and RANK(VOLUME)
        rank_vwap = vwap.rank(axis=0, pct=True)
        rank_volume = volume.rank(axis=0, pct=True)

        # Step 7: Compute CORR(RANK(VWAP), RANK(VOLUME), 7)
        corr_rank_vwap_volume = rank_vwap.rolling(window=7).corr(rank_volume)

        # Step 8: Apply DECAYLINEAR to CORR over a 3-period window
        decay_corr_rank = corr_rank_vwap_volume.rolling(window=3).apply(
            lambda x: np.dot(x[::-1], np.linspace(1, len(x), len(x))) / np.sum(np.linspace(1, len(x), len(x))),
            raw=True
        )

        # Step 9: Rank the DECAYLINEAR result
        rank_decay_corr_rank = decay_corr_rank.rank(axis=0, pct=True)

        # Step 10: Compute the final Alpha130
        alpha = rank_decay_midpoint / rank_decay_corr_rank

        return alpha
