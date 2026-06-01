import pandas as pd
import fozsfactors as fo
import fozsfactor_analyzer as foa
import backtesting as bt
import os
import all_factors as fac

if __name__ == '__main__':
    stocks_name = ['AAPL', 'AMGN', 'ALSN', 'ALLY', 'AMP', 'ANF', 'AOS', 'APA', 'APD', 'APH', 'APO', 'GPC', 'GPK', 'GPN',
                   'LEN', 'LOW', 'MET', 'NDAQ']
    stock100_symbols = [
        'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'NVDA', 'TSLA', 'PEP', 'COST',
        'AVGO', 'CSCO', 'ADBE', 'INTC', 'CMCSA', 'NFLX', 'TXN', 'HON', 'AMGN', 'QCOM',
        'SBUX', 'AMD', 'INTU', 'ISRG', 'MDLZ', 'MU', 'ADP', 'GILD', 'ADI', 'LRCX',
        'AMAT', 'BKNG', 'PANW', 'MAR', 'CSX', 'MRVL', 'KLAC', 'ORLY',
        'SNPS', 'MCHP', 'CDNS', 'FTNT', 'MNST', 'IDXX', 'WDAY', 'XEL', 'CHTR',
        'KDP', 'PAYX', 'VRTX', 'DXCM', 'LULU', 'EBAY', 'SWKS', 'EXC', 'SIRI',
        'ANSS', 'CTAS', 'MTCH', 'VRSK', 'AZN', 'BIIB', 'REGN', 'ALGN'
    ]

    # 文件夹路径（替换为你的文件夹路径）
    folder_path = 'D:/桌面/semester1/MF703 Programming/Project/stock_data'

    # 获取所有 CSV 文件的名字（不包含 .csv 扩展名）
    csv_files = [os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.endswith('.csv')]

    FOZS1 = fo.Calc_Factors(csv_files, [fac.TRIX()])
    factors_dict = fo.Processing_data(FOZS1)

    Far_FOZS1 = foa.FactorAnalysisResult(factors_dict['trix'],
                                         start_date='2015-01-01',
                                         end_date='2020-12-31',
                                         periods=[1, 5, 21],
                                         quantiles=30, weights='Volume')
    print(Far_FOZS1.create_summary_tear_sheet())
    print(Far_FOZS1.ic_information_table)

    FOZS_BC = bt.Backtesting(factors_dict, start_date="2021-01-01", end_date="2024-11-01")
    FOZS_BC.back_test()
