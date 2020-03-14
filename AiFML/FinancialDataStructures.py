import pandas as pd
import numpy as np
import enum


class VolumeStep(enum.Enum):
    OB_TOP = 1,
    QSUM = 2


def calc_ob_top_volume(ob):
    return np.mean(ob['askv0'] + ob['bidv0'])


def calc_qsum(trades):
    return np.percentile(trades.amount[trades.side == 0], 90) + np.percentile(trades.amount[trades.side == 1], 90)


def get_sampling_vol(trades, ob, vs: VolumeStep):
    if vs == VolumeStep.OB_TOP:
        return calc_ob_top_volume(ob)
    if vs == VolumeStep.QSUM:
        return calc_qsum(trades)


def get_sampled_trades(trades, ob, volume_step):
    trades_sampled = []
    vol_sampled = get_sampling_vol(trades, ob, volume_step)
    s = volume_step
    for i, r in trades.iterrows():
        d = {}
        if r.csum_amount > s:
            d.update(r)
            trades_sampled.append(d)
            s = r.csum_amount + vol_sampled
    return pd.DataFrame(trades_sampled)


def volume_bars(trades: pd.DataFrame, volume_step: VolumeStep, ob: pd.DataFrame, prepare_data: bool):
    if prepare_data:
        trades['timestamp_index'] = trades.index
        trades['csum_amount'] = trades.amount.cumsum()
        ob['timestamp_index'] = trades.index
    t = get_sampled_trades(trades, ob, volume_step)
    if ob is None:
        return t
    else:
        return pd.merge_asof(t, ob, on='timestamp_index')