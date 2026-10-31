"""Deep learning and explainability: an LSTM forecaster, explained with SHAP on a surrogate tree.

STATUS: written, not executed in the build sandbox. PyTorch and SHAP could not be installed there
(the package index was blocked). To run it locally:
    pip install torch shap
    python -m src.deep_learning
Results and figures will then appear in outputs/. Until that run is done, treat the deep-learning
line of the report as unverified.

Design choices:
- Forecasting, not classification: the LSTM takes the previous 24 hours and predicts the next hour.
- Windows never cross a data gap (the Sep 2014 - May 2015 hole), so every sequence is contiguous.
- Chronological split: train windows end in 2017, test windows fall in 2018.
- Baseline: persistence (the next hour equals the current hour).
- SHAP is applied to a gradient-boosted surrogate, which the brief permits when the network is hard to explain.
"""
import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

LOOKBACK = 24
SEQ_FEATURES = ["traffic_volume", "temp", "rain_1h", "clouds_all", "hour_sin", "hour_cos", "is_weekend"]


def contiguous_segments(df: pd.DataFrame) -> list:
    """Return (start, end) index pairs for runs of consecutive hours with no gap."""
    gaps = df["date_time"].diff() != pd.Timedelta(hours=1)
    seg_id = gaps.cumsum()
    return [(g.index.min(), g.index.max()) for _, g in df.groupby(seg_id)]


def make_windows(df: pd.DataFrame, values: np.ndarray):
    X, y, idx = [], [], []
    for start, end in contiguous_segments(df):
        for t in range(start + LOOKBACK, end + 1):
            if t - LOOKBACK < start:
                continue
            X.append(values[t - LOOKBACK:t])
            y.append(values[t, 0])  # traffic_volume is column 0
            idx.append(t)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(idx)


def train_lstm(X_tr, y_tr, X_te, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    class Forecaster(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    model = Forecaster(X_tr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    for epoch in range(5):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
        logger.info("LSTM epoch %d training MSE %.4f", epoch + 1, total / len(ds))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_te)).numpy()
    return model, pred


def run() -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler

    from src.features import build_frame
    from src.data_prep import load_clean

    frame = build_frame(load_clean()).reset_index(drop=True)
    years = frame["date_time"].dt.year
    scaler = StandardScaler().fit(frame.loc[years <= config.TRAIN_END_YEAR, SEQ_FEATURES])
    values = scaler.transform(frame[SEQ_FEATURES]).astype(np.float32)
    X, y, idx = make_windows(frame, values)
    target_years = frame.loc[idx, "date_time"].dt.year.to_numpy()
    tr, te = target_years <= config.TRAIN_END_YEAR, target_years == config.TEST_YEAR
    logger.info("LSTM windows: %d train, %d test (contiguous 24-hour sequences only)", tr.sum(), te.sum())

    mu, sd = scaler.mean_[0], scaler.scale_[0]  # traffic_volume is column 0
    model, pred_s = train_lstm(X[tr], y[tr], X[te], config.SEED)
    actual = frame.loc[idx[te], "traffic_volume"].to_numpy()
    pred = pred_s * sd + mu
    persistence = frame.loc[idx[te] - 1, "traffic_volume"].to_numpy()
    lstm_mae = mean_absolute_error(actual, pred)
    naive_mae = mean_absolute_error(actual, persistence)
    logger.info("LSTM 2018 MAE %.1f vs persistence %.1f", lstm_mae, naive_mae)

    # Surrogate: a gradient-boosted tree that mimics the LSTM, so SHAP can explain it.
    summary = lambda w: np.column_stack([w[:, -1, :], w[:, :, 0].mean(1), w[:, :, 0].std(1), w[:, :, 0].min(1), w[:, :, 0].max(1)])
    names = [f"last_{f}" for f in SEQ_FEATURES] + ["mean_traffic_24h", "std_traffic_24h", "min_traffic_24h", "max_traffic_24h"]
    S_tr, S_te = summary(X[tr]), summary(X[te])
    surrogate = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=config.SEED)
    surrogate.fit(S_tr, model_predict_train(model, X[tr]))
    fidelity = r2_score(pred_s, surrogate.predict(S_te))
    logger.info("Surrogate fidelity on 2018 (R2 against the LSTM): %.3f", fidelity)

    try:
        import shap
        explainer = shap.TreeExplainer(surrogate)
        sv = explainer.shap_values(S_te[:2000])
        importance = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
        method = "SHAP (TreeExplainer on surrogate)"
    except ImportError:
        logger.warning("shap not installed; falling back to impurity-based importance on the surrogate")
        importance = pd.Series(surrogate.feature_importances_, index=names).sort_values(ascending=False)
        method = "impurity importance on surrogate (SHAP not installed)"

    importance.to_csv(config.METRIC_DIR / "lstm_surrogate_importance.csv", header=["mean_abs_importance"])
    pd.DataFrame([{"model": "lstm", "mae_2018": lstm_mae, "fidelity_r2_surrogate": fidelity,
                   "persistence_mae_2018": naive_mae, "explain_method": method}]
                 ).to_csv(config.METRIC_DIR / "lstm_metrics.csv", index=False)
    return {"lstm_mae": lstm_mae, "persistence_mae": naive_mae, "fidelity": fidelity, "importance": importance}


def model_predict_train(model, X_tr):
    import torch
    with torch.no_grad():
        return model(torch.from_numpy(X_tr)).numpy()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    run()
