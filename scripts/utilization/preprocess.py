"""Preprocess logic for utilization."""

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, RAW_COLUMNS, TARGET_COL


def load_data(path):
    return pd.read_csv(path)


def preprocess(raw):
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing utilization columns: {sorted(missing)}")
    df = raw.drop_duplicates().copy()
    if (
        df.empty
        or df.MachineId.isna().any()
        or df.MachineId.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Utilization requires nonempty data and MachineId values")
    for col in ["StartedAt", "EndedAt"]:
        original = df[col]
        df[col] = pd.to_datetime(original, format="mixed", errors="coerce")
        if (original.notna() & df[col].isna()).any():
            raise ValueError(f"Invalid {col} timestamps")
    if df.StartedAt.isna().any():
        raise ValueError("StartedAt must be present for every session")
    duration = pd.to_numeric(df.SessionMinutes, errors="coerce")
    if (
        df.SessionMinutes.notna()
        & (duration.isna() | ~np.isfinite(duration) | duration.lt(0))
    ).any():
        raise ValueError("SessionMinutes must be finite and nonnegative")
    df["EndedAt"] = df.EndedAt.fillna(
        df.StartedAt + pd.to_timedelta(duration, unit="m")
    )
    if (df.EndedAt < df.StartedAt).any():
        raise ValueError("Session ends before it starts")
    watermark = max(df.StartedAt.max(), df.EndedAt.max())
    stop = watermark.normalize()
    rows = []
    for machine, sessions in df.groupby("MachineId", sort=True):
        first = sessions.StartedAt.min().normalize()
        intervals = []
        for start, end in (
            sessions[["StartedAt", "EndedAt"]]
            .dropna()
            .sort_values("StartedAt")
            .itertuples(index=False, name=None)
        ):
            if intervals and start <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
            else:
                intervals.append((start, end))
        minutes = {}
        for start, end in intervals:
            end = min(end, stop)
            while start < end:
                day = start.normalize()
                boundary = min(day + pd.Timedelta(days=1), end)
                minutes[day] = (
                    minutes.get(day, 0.0) + (boundary - start).total_seconds() / 60
                )
                start = boundary
        unknown = set(sessions.loc[sessions.EndedAt.isna(), "StartedAt"].dt.normalize())
        for day in pd.date_range(first, stop - pd.Timedelta(days=1), freq="D"):
            occupied = np.nan if day in unknown else minutes.get(day, 0.0)
            rows.append(
                {
                    "MachineId": str(machine),
                    "Date": day,
                    "OccupiedMinutes": occupied,
                    "UtilizationPct": occupied / 1440 * 100,
                    "UnknownDuration": int(day in unknown),
                }
            )
    if not rows:
        raise ValueError("No complete calendar days available")
    return pd.DataFrame(rows).sort_values(["MachineId", "Date"]).reset_index(drop=True)


def add_features(base):
    df = base.sort_values(["MachineId", "Date"]).copy()
    g = df.groupby("MachineId")["UtilizationPct"]
    df["Lag1"] = g.shift(1)
    df["Lag7"] = g.shift(7)
    for days in [3, 7]:
        df[f"Mean{days}"] = g.transform(lambda s: s.rolling(days, min_periods=1).mean())
    df["HistoryDays"] = df.groupby("MachineId").cumcount() + 1
    df["TargetDate"] = df.Date + pd.Timedelta(days=1)
    df["NextDayOfWeek"] = df.TargetDate.dt.dayofweek
    df["NextIsWeekend"] = df.NextDayOfWeek.isin([5, 6]).astype(int)
    df[TARGET_COL] = g.shift(-1)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df


def engineer_features(base):
    return add_features(base).dropna(subset=[TARGET_COL]).reset_index(drop=True)


def chronological_split(df, train_frac=0.75):
    dates = sorted(df.TargetDate.unique())
    if len(dates) < 4 or not 0 < train_frac < 1:
        raise ValueError(
            "Need at least four target dates and a split fraction between 0 and 1"
        )
    cutoff = pd.Timestamp(
        dates[min(len(dates) - 2, max(0, int(len(dates) * train_frac) - 1))]
    )
    train, test = (
        df[df.TargetDate <= cutoff].copy(),
        df[df.TargetDate > cutoff].copy(),
    )
    return (train, test, cutoff)
