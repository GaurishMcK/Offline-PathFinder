import os
import pandas as pd
import streamlit as st


def ms_to_mmss(ms) -> str:
    """Convert milliseconds to mm:ss for UI display."""
    try:
        ms_i = int(ms) if ms is not None else 0
    except Exception:
        ms_i = 0
    s = ms_i // 1000
    return f"{s//60:02d}:{s%60:02d}"


def get_index_dir(index_path: str) -> str:
    index_abs = os.path.abspath(index_path)
    return os.path.dirname(index_abs)


@st.cache_data
def load_index(index_path: str) -> pd.DataFrame:
    if index_path.lower().endswith(".parquet"):
        df = pd.read_parquet(index_path)
    else:
        df = pd.read_csv(index_path)

    req = {"conversation_id", "transcript_id", "emp_id", "call_type", "transcript_path"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Index missing columns: {sorted(missing)}")

    return df


@st.cache_data
def load_transcript(transcript_path: str) -> pd.DataFrame:
    if transcript_path.lower().endswith(".parquet"):
        df = pd.read_parquet(transcript_path)
    else:
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                return pd.read_csv(transcript_path, encoding=enc)
            except UnicodeDecodeError:
                pass

    req = {"phrase_rank", "speaker", "START_TIME_MS", "END_TIME_MS", "text"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Transcript missing columns: {sorted(missing)}")

    df = df.sort_values("phrase_rank").copy()
    df["duration_sec"] = (df["END_TIME_MS"] - df["START_TIME_MS"]) / 1000.0
    return df


def resolve_data_path(index_path: str, maybe_relative_path: str) -> str:
    """Resolve transcript path relative to index location for portability."""
    if not isinstance(maybe_relative_path, str):
        maybe_relative_path = str(maybe_relative_path)

    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path

    index_dir = get_index_dir(index_path)
    candidate = os.path.join(index_dir, maybe_relative_path)
    if os.path.exists(candidate):
        return candidate

    return os.path.abspath(maybe_relative_path)
