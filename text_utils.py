import unicodedata
import pandas as pd


def sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return unicodedata.normalize("NFKC", s)


def segments_to_plaintext(df: pd.DataFrame) -> str:
    lines: list[str] = []
    for r in df.sort_values("phrase_rank").to_dict("records"):
        lines.append(f"#{r['phrase_rank']} | speaker:{r['speaker']} | {r['START_TIME_MS']}->{r['END_TIME_MS']} ms")
        lines.append(str(r["text"]).strip())
        lines.append("")
    return "\n".join(lines).strip()
