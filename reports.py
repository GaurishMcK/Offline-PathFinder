import pandas as pd


def build_activity_report(eval_df: pd.DataFrame) -> pd.DataFrame:
    rep = (
        eval_df.groupby("activity", as_index=False)
        .agg(
            avg_score=("efficacy_score", "mean"),
            total_duration_sec=("duration_sec", "sum"),
            n_blocks=("present", "size"),
        )
    )
    rep["avg_score"] = rep["avg_score"].astype(float).round(1)
    rep["total_duration_sec"] = rep["total_duration_sec"].astype(float).round(1)
    return rep


def build_agent_coaching_summary(eval_df: pd.DataFrame) -> str:
    df = eval_df.copy()
    df["efficacy_score"] = df["efficacy_score"].astype(float)
    worst = df.sort_values("efficacy_score", ascending=True).head(3)
    lines = ["Top coaching priorities (lowest scoring activities):"]
    for _, r in worst.iterrows():
        lines.append(f"- {r['activity']}: {float(r['efficacy_score']):.1f}%")
    return "\n".join(lines)
