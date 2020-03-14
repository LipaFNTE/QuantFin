import pandas as pd
import numpy as np
from Data import CryptoData


def prepare_duration_orderbook(ob):
    ob['spread'] = ob['ask0'] - ob['bid0']
    ob['mid_price'] = (ob['ask0'] + ob['bid0'])/2
    ob['lr'] = np.insert(np.diff(np.log(ob['mid_price'])))
    return ob


def get_duration_data(pair, interval, timestamp_int: bool, extended: bool, ob_db):
    tr = CryptoData.get_trades(pair, interval, timestamp_int)
    if extended:
        ob = CryptoData.get_orderbook(pair, interval, ob_db, timestamp_int)
        ob = prepare_duration_orderbook(ob)




class ACD:
    def __init__(self):
        pass

    def estimate_ARMA_duration(self):
        pass

    def get_ACD_model(self):
        pass

    class EACD:
        pass