import yfinance as yf
import pandas as pd
import time
# Define the ticker for Zen Technologies (NSE)
ticker = "ZENTEC.NS"

print("zentec data fetched from yahoo finance")
time.sleep(2)
# this is important as many times it is not able to fetch the data 
# this acts as a reminder
# Download data
data = yf.download(ticker, start="2000-01-01", end="2025-07-15")
time.sleep(2)
print("data downloaded")
# Save the data to CSV file locally
data.to_csv("zentec_stock_data.csv")
print("data converted to the csv form")
time.sleep(2)
print("✅ Data downloaded and saved as 'zentec_stock_data.csv'")
