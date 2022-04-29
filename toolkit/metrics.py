import pandas as pd
from sklearn.metrics import f1_score


def macro_disparate_impact(df_to_compute: pd.DataFrame) -> float:
    """
    Compute the macro disparate impact

    Args:
        df_to_compute (pd.DataFrame): DataFrame to compute the score on.

    Returns:
        float: the macro disparate impact score.
    """
    counts = df_to_compute.groupby(["job", "gender"]).size().unstack("gender")
    counts["disparate_impact"] = counts[["M", "F"]].max(axis="columns") / counts[
        ["M", "F"]
    ].min(axis="columns")
    return counts["disparate_impact"].mean()


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")
