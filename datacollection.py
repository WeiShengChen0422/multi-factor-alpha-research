import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay

def get_historical_data(stock_list_file_path, output_dir, start_date, end_date):
    # verify the input date is correctly formated
    try:
        datetime.strptime(start_date,'%Y-%m-%d')
    except ValueError:
        print('Error: start_date is NOT in the correct format')
        return

    try:
        input_date = datetime.strptime(end_date, '%Y-%m-%d')
        yesterday = datetime.now() - timedelta(days=1)
        if input_date > yesterday:
            print('end_date is NOT in the range. Please ensure a correct input')
            return 
    except ValueError:
        print('Error: end_date is NOT in the correct format')
        return
            
    # extract and adjust ticker in compliance with yfinance API
    tickers_df = pd.read_csv(stock_list_file_path)
    if 'Symbol' in tickers_df.columns:
        tickers = tickers_df['Symbol'].dropna().tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
    else:
        print("The CSV file does not contain a 'Symbol' column.")
        tickers = []

    # batch download from yfinance
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        for ticker in batch:
            stock = yf.Ticker(ticker)
            try:
                # Try to download 10 years of data
                ticker_data = stock.history(start=start_date, end=end_date)
                if ticker_data.empty:
                    raise ValueError(f"No data available for {ticker} for 10 years.")
                print(f"Data for {ticker} downloaded for 10 years.")
            
            except Exception as e:
                # print(f"Error downloading 10 years of data for {ticker}: {e}. Trying to fetch the max available data...")
                # If downloading 10 years fails, fallback to 'max' period
                try:
                    ticker_data = stock.history(period='max')
                    if ticker_data.empty:
                        raise ValueError(f"No data available for {ticker} even for the maximum period.")
                    print(f"Data for {ticker} downloaded for the max available period.")
                except Exception as e:
                    print(f"Error downloading data for {ticker}: {e}")
                    continue  # Skip this ticker if both attempts fail
            
            ticker_data.index = ticker_data.index.date

            # ticker_data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            file_name = os.path.join(output_dir, f"{ticker}.csv")
            ticker_data.to_csv(file_name)
            print(f"Data for {ticker} saved to {file_name}")

    print("All downloads completed.")

def ratio_cleaning_and_merge(historical_data_dir, ratio_file_path, output_dir):
    # cleaning the raw ratio data
    ratio_df = pd.read_csv(ratio_file_path)
    ratio_df.drop(['permno', 'adate','qdate', 'divyield'], axis=1, inplace = True, errors='ignore')
    ratio_df.columns = ['Date','Bm','Pe','Ps','Pcf','Roa','Roe','Ptb','Ticker']
    ratio_df['Date'] = pd.to_datetime(ratio_df['Date'])
    ratio_df['Date'] = ratio_df['Date'] + BDay(0)
    ratio_df.set_index('Date', inplace = True)
    output_path = os.path.join(output_dir, 'cleaned_ratio.csv')
    ratio_df.to_csv(output_path)
    print ('Ratio cleaning completed')

    # Merge the historical data and ratio data
    for file_name in os.listdir(historical_data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(historical_data_dir, file_name)
            historical_df =pd.read_csv(file_path)
            historical_df.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
            historical_df.set_index('Date', inplace=True)

            ticker = os.path.splitext(file_name)[0]
            filtered_ratio_df = ratio_df[ratio_df['Ticker']==ticker]
            merged_df = historical_df.merge(filtered_ratio_df, left_index = True, right_index=True, how='left')
            merged_df.drop(columns=['Ticker'], inplace =True)
            ffill_columns = ['Bm','Pe','Ps','Pcf','Roa','Roe','Ptb']
            merged_df[ffill_columns] = merged_df[ffill_columns].ffill()
            merged_df.dropna(axis=0, how='any', inplace=True)

            output_file_path = os.path.join(output_dir, file_name)
            merged_df.to_csv(output_file_path, index=True)
    
    # remove the intermediate file: clean_ratio.csv
    try:
        os.remove(output_path)
        print('intermediate file deleted')
    except Exception as e:
        print('System does not permit to delete intermediate file. Please manually delete cleaned_ratio.csv under output_dir')

    print('Merge completed, check the output_dir')

# main program
stock_list_file_path = r'C:\Users\rayxu\OneDrive\Desktop\project\stocks_list.csv'
historical_data_dir = r'C:\Users\rayxu\OneDrive\Desktop\project\historical_data_10yrs\from_2024'
start_date = '2014-11-1'
end_date = '2024-11-22'
get_historical_data(stock_list_file_path, historical_data_dir, start_date, end_date)

ratio_file_path = r"C:\Users\rayxu\OneDrive\Desktop\project\ratios.csv"
output_dir = r"C:\Users\rayxu\OneDrive\Desktop\project\historical_data_with_ratios"
ratio_cleaning_and_merge(historical_data_dir, ratio_file_path, output_dir)
