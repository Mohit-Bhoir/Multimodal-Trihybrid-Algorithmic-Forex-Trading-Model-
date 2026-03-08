import pandas as pd
import sys
from pandas import api
import yaml
import os 
import tpqoa as tpqoa
import time
import datetime


# end = datetime.datetime.now() - datetime.timedelta(days=1)
# start = end - datetime.timedelta(days=60)

#Load params from params.yaml
params = yaml.safe_load(open("params.yaml"))['fetch']

def get_fetch_window():
    now = datetime.datetime.utcnow().replace(second=0, microsecond=0)

    # Align to the last completed 15-minute candle
    end = now - datetime.timedelta(minutes=now.minute % 15)

    # Roll weekends back to Friday
    if end.weekday() == 5:  # Saturday
        end = (end - datetime.timedelta(days=1)).replace(hour=21, minute=45)
    elif end.weekday() == 6:  # Sunday
        end = (end - datetime.timedelta(days=2)).replace(hour=21, minute=45)

    # Exact 2 calendar months back
    start = (pd.Timestamp(end) - pd.DateOffset(months=2)).to_pydatetime()
    start = start.replace(second=0, microsecond=0)

    return start, end

def fetch_data(output_path):
    start, end = get_fetch_window()

    api = tpqoa.tpqoa("src/oanda.cfg")
    mid_EURUSD = api.get_history(
        instrument="EUR_USD",
        start=start,
        end=end,
        granularity="M15",
        price="M",
    )
    bid_EURUSD = api.get_history(
        instrument="EUR_USD",
        start=start,
        end=end,
        granularity="M15",
        price="B",
    )
    ask_EURUSD = api.get_history(
        instrument="EUR_USD",
        start=start,
        end=end,
        granularity="M15",
        price="A",
    )
    
    cols  = ['time', 'mid', 'bid', 'ask']
    raw = pd.DataFrame({
        'price': mid_EURUSD['c'],
        'spread': ask_EURUSD['c'] - bid_EURUSD['c'],
    })
    
    raw.dropna(inplace=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw.to_csv(output_path)
    print(f"Fetching EUR_USD from {start} to {end} at M15")
    print(f"Data saved to {output_path}")
    
if __name__ == "__main__":
    fetch_data(params['output_path'])