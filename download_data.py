from config import APIKEY
from alpha_vantage.timeseries import TimeSeries 
import numpy as np

def download_data(config):
    ts = TimeSeries(key=APIKEY) 
    data, meta_data = ts.get_daily(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "From " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number of data points: " + str(num_data_points) + "\n" + str(display_date_range))

    return data_date, data_close_price, num_data_points, display_date_range