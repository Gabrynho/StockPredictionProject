import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class StockDataExtractor:
    def __init__(self):
        self.end_date = datetime.now()
        self.start_date1 = self.end_date - timedelta(days=729)
        self.start_date2 = self.end_date - timedelta(days=3650)
        self.data = None
        self.data2 = None

    def download_data(self, tickers=["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "ACN", "CSCO"]):
        self.data = yf.download(tickers, start=self.start_date1.strftime('%Y-%m-%d'), end=self.end_date.strftime('%Y-%m-%d'),
                                group_by="ticker", threads=True, interval="1h")
        self.data2 = yf.download(tickers, start=self.start_date2.strftime('%Y-%m-%d'), end=self.end_date.strftime('%Y-%m-%d'),
                                 group_by="ticker", threads=True, interval="1d")
    
    def load_data(self):
        self.data = pd.read_csv("data_2y_h.csv", header=[0, 1], index_col=0)
        self.data2 = pd.read_csv("data_10y_d.csv", header=[0, 1], index_col=0)

    def load_data_preprocessed(self):
        self.data = pd.read_csv("data/data_2y_h_new.csv", header=[0, 1], index_col=0)
        self.data2 = pd.read_csv("data/data_10y_d_new.csv", header=[0, 1], index_col=0)

    def preprocess_data(self, savedata = False):
        self.data.drop(['Adj Close', 'Volume'], axis=1, level=1, inplace=True)
        self.data2.drop(['Adj Close', 'Volume'], axis=1, level=1, inplace=True)

        for data in [self.data, self.data2]:
            for company in data.columns.levels[0]:
                # Calculate the fractional change, high, and low
                fracChange = (data[company]['Close'] - data[company]['Open']) / data[company]['Open']
                fracHigh = (data[company]['High'] - data[company]['Open']) / data[company]['Open']
                fracLow = (data[company]['Open'] - data[company]['Low']) / data[company]['Open']

                data.drop((company, 'Close'), axis=1, inplace=True)
                data.drop((company, 'High'), axis=1, inplace=True)
                data.drop((company, 'Low'), axis=1, inplace=True)

                data[(company, 'fracChange')] = fracChange
                data[(company, 'fracHigh')] = fracHigh
                data[(company, 'fracLow')] = fracLow
        
        if savedata == True:
            self.data.to_csv("../data/data_2y_h_new.csv")
            self.data2.to_csv("../data/data_10y_d_new.csv")

    def save_data_to_csv(self):
        self.data.to_csv("data_2y_h.csv")
        self.data2.to_csv("data_10y_d.csv")
        

if __name__ == "__main__":
    tickers=["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "ACN", "CSCO"] # Best 10 performing tech companies, but you can modify this and choose your own companies
    dataset = StockDataExtractor()
    dataset.download_data(tickers)
    dataset.preprocess_data()
    dataset.save_data_to_csv()