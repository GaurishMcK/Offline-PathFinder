import pandas as pd


def build_activity_blocks(transcript_df: pd.DataFrame, cls_df: pd.DataFrame, sel_enum=None) -> pd.DataFrame:
    # Merge classification into transcript
    joined = transcript_df.merge(cls_df[["phrase_rank", "activity"]], on="phrase_rank", how="left")

    # Compute duration_sec if not already present
    if "duration_sec" not in joined.columns:
        if {"START_TIME_MS", "END_TIME_MS"}.issubset(joined.columns):
            joined["duration_sec"] = (joined["END_TIME_MS"] - joined["START_TIME_MS"]) / 1000.0
        else:
            # Fallback: if timings are missing, assume 0 duration
            joined["duration_sec"] = 0.0

    joined["speaker_text"] = "[" + joined["speaker"].astype(str) + "]: " + joined["text"].astype(str)

    # Aggregate blocks by activity
    blocks = (
        joined.groupby("activity", dropna=False)
        .agg(
            duration_sec=("duration_sec", "sum"),
            present=("phrase_rank", "size"),
        )
        .reset_index()
    )

    # Collect exchanges per activity
    exchanges = (
        joined.groupby("activity", dropna=False)["speaker_text"]
        .apply(lambda s: "\n".join(s.tolist()))
        .reset_index(name="exchange")
    )

    out = blocks.merge(exchanges, on="activity", how="left")

    # present should be boolean in your UI later
    out["present"] = out["present"] > 0

    return out
