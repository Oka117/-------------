"""EXP-03 causal features, fitted on training-only raw-feature importance."""
import numpy as np
import pandas as pd
from baseline import fit_model

WINDOWS = {'mean_deviation':[3,18,72], 'lag_difference':[1,3,18],
           'history_std':[3,18,72], 'standardized_deviation':[18,72]}


def fit_recipe(raw, y, kind, rounds=300, threads=4, seed=42):
    if kind not in WINDOWS:
        raise ValueError(f'Unknown feature kind: {kind}')
    model, _ = fit_model(raw,y,rounds,threads,seed)
    if model is None:
        # Predeclared single-class fallback: input schema order; no validation data.
        selected = list(raw.columns[:30])
        source = 'first_30_schema_columns_single_class'
    else:
        order = np.argsort(-model.feature_importance('gain'),kind='stable')[:30]
        selected = raw.columns[order].tolist()
        source = 'training_only_raw_lightgbm_gain_top30'
    std = raw[selected].std(ddof=0).fillna(0).to_numpy(dtype=float)
    epsilon = np.maximum(std*1e-3,1e-6)
    return dict(version=2,kind=kind,raw_features=list(raw.columns),selected=selected,
                selection_source=source,windows=WINDOWS[kind],std_ddof=0,
                min_periods='full_window',epsilon=dict(zip(selected,epsilon.tolist())),
                epsilon_rule='max(training_raw_population_std * 0.001, 0.000001)',
                gap_policy='reset_history_on_non_20_minute_gap',history_points=72)


def window_stat(selected, window, statistic):
    # Recompute each window independently. Incremental rolling variance can retain
    # roundoff after a constant stretch, changing tree splits across chunks.
    result = {}
    for col in selected:
        source = selected[col].to_numpy()
        values = np.full(len(source), np.nan)
        if len(source) > window:
            views = np.lib.stride_tricks.sliding_window_view(source[:-1], window)
            values[window:] = np.std(views, axis=1, ddof=0) if statistic == "std" else np.mean(views, axis=1)
        result[col] = values
    return pd.DataFrame(result, index=selected.index)


def build_features(raw, recipe):
    if list(raw.columns) != recipe['raw_features']:
        raise ValueError('Raw feature schema/order differs from recipe')
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError('Raw NaN/inf is forbidden; derived warmup NaNs are allowed')
    selected=raw[recipe['selected']].astype('float64')
    lagged=selected.shift(1)
    derived={}
    for window in recipe['windows']:
        kind=recipe['kind']
        if kind=='lag_difference':
            values=selected-selected.shift(window)
        else:
            rolling=lagged.rolling(window,min_periods=window)
            if kind=='mean_deviation':
                values=selected-rolling.mean()
            elif kind=='history_std':
                values=window_stat(selected,window,"std")
            elif kind=='standardized_deviation':
                denom=window_stat(selected,window,"std").clip(lower=pd.Series(recipe['epsilon']),axis=1)
                values=(selected-window_stat(selected,window,"mean"))/denom
            else:
                raise ValueError('Unknown feature kind')
        for col in recipe['selected']:
            derived[f'ts_{kind}_{window}_{col}']=values[col].astype('float32')
    output=pd.concat([raw,pd.DataFrame(derived,index=raw.index)],axis=1)
    if np.isinf(output.to_numpy()).any():
        raise ValueError('Nonfinite derived feature overflow')
    return output


def inference_features(test, context, recipe):
    """Use persisted last 72 training rows; reset if test does not immediately follow."""
    cols=recipe['raw_features']
    context=context.tail(recipe['history_points'])
    if not context.empty:
        delta=pd.Timestamp(test.timestamp.iloc[0])-pd.Timestamp(context.timestamp.iloc[-1])
        if delta<=pd.Timedelta(0):
            raise ValueError('Test overlaps saved history')
        if delta!=pd.Timedelta(minutes=20):
            context=context.iloc[:0]
    combined=pd.concat([context,test],ignore_index=True)
    return build_features(combined[cols],recipe).iloc[len(context):].reset_index(drop=True)
