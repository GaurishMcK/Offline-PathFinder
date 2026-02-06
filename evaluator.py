import json
import pandas as pd
import streamlit as st
from llm_client import get_openai
from text_utils import sanitize_text


def build_eval_messages(rubric: dict, activity: str, exchange: str, metadata: dict):
    # (kept for reference; not used in the new bulk path)
    system = {
        "role": "system",
        "content": (
            "You are an expert QA evaluator for telecommunications calls. "
            "Score the agent's performance for the given activity using the rubric. "
            "Provide evidence-based observations and tactical coaching."
        ),
    }
    user = {
        "role": "user",
        "content": f"""
<RULES>
1) Use only the exchange and rubric provided.
2) Be objective; do not be overly critical.
3) No nulls.
4) Return efficacy_score as a percentage from 0 to 100 (not 0–1).
</RULES>

<METADATA>
{json.dumps(metadata)}
</METADATA>

<ACTIVITY>
{activity}
</ACTIVITY>

<RUBRIC>
{json.dumps(rubric)}
</RUBRIC>

<EXCHANGE>
{exchange}
</EXCHANGE>

<OUTPUT>
Return JSON:
{{
  "activity": "string",
  "efficacy_score": "number",
  "detailed_observations": "string",
  "tactical_feedback": "string"
}}
</OUTPUT>
""".strip(),
    }
    return [system, user]


def _not_present_row(activity: str, duration_sec: float, rubric: dict) -> dict:
    expected = rubric.get(activity, {}) if isinstance(rubric, dict) else {}
    exp_good = expected.get("good", "") if isinstance(expected, dict) else ""
    exp_bad = expected.get("bad", "") if isinstance(expected, dict) else ""

    obs = "Activity not present in the call transcript for this interaction."
    fb_parts = []
    if exp_good:
        fb_parts.append(f"Expected: {exp_good}")
    if exp_bad:
        fb_parts.append(f"Avoid: {exp_bad}")
    fb = " ".join(fb_parts) if fb_parts else "Ensure this step is completed when applicable."

    return {
        "activity": activity,
        "present": False,
        "duration_sec": float(duration_sec or 0.0),
        "efficacy_score": 0.0,
        "detailed_observations": obs,
        "tactical_feedback": fb,
    }


def build_bulk_eval_messages(items: list[dict], metadata: dict) -> list[dict]:
    """One-shot evaluation for all present activities to avoid N network round-trips."""
    system = {
        "role": "system",
        "content": (
            "You are an expert QA evaluator for telecommunications calls. "
            "For EACH item, score the agent's performance for that activity using the provided rubric snippet. "
            "Provide evidence-based observations and tactical coaching. Return strictly valid JSON."
        ),
    }

    user = {
        "role": "user",
        "content": f"""
<RULES>
1) Use ONLY the exchange + rubric_snippet given per item.
2) Be objective; do not be overly critical.
3) No nulls.
4) efficacy_score is a percentage from 0 to 100 (not 0–1).
5) Return exactly one result per input item.
</RULES>

<METADATA>
{json.dumps(metadata)}
</METADATA>

<ITEMS>
Each item contains:
- activity
- rubric_snippet (good/bad expectations)
- exchange (agent+customer lines)

{json.dumps(items)}
</ITEMS>

<OUTPUT>
Return JSON:
{{
  "results": [
    {{
      "activity": "string",
      "efficacy_score": "number",
      "detailed_observations": "string",
      "tactical_feedback": "string"
    }}
  ]
}}
</OUTPUT>
""".strip(),
    }

    return [system, user]


@st.cache_data(show_spinner=False)
def _evaluate_present_items_cached(items_json: str, metadata_json: str) -> dict:
    """Cache the LLM result so Streamlit reruns don't repay the LLM tax."""
    client, model = get_openai()

    items = json.loads(items_json)
    metadata = json.loads(metadata_json)

    schema = {
        "name": "bulk_activity_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "activity": {"type": "string"},
                            "efficacy_score": {"type": "number", "minimum": 0, "maximum": 100},
                            "detailed_observations": {"type": "string"},
                            "tactical_feedback": {"type": "string"},
                        },
                        "required": ["activity", "efficacy_score", "detailed_observations", "tactical_feedback"],
                    },
                }
            },
            "required": ["results"],
        },
    }

    resp = client.chat.completions.create(
        model=model,
        messages=build_bulk_eval_messages(items, metadata),
        response_format={"type": "json_schema", "json_schema": schema},
    )
    return json.loads(resp.choices[0].message.content)


def evaluate_blocks(blocks_df: pd.DataFrame, rubric: dict, metadata: dict) -> pd.DataFrame:
    """Evaluate all present activities in ONE LLM call (instead of 1 call per activity)."""
    activity_order = blocks_df["activity"].tolist()

    present_rows = []
    rows_out = []

    for _, r in blocks_df.iterrows():
        activity = str(r["activity"])
        present = bool(r.get("present", True))
        exchange = sanitize_text(r.get("exchange", ""))
        duration_sec = float(r.get("duration_sec", 0.0))

        if (not present) or (not exchange.strip()):
            rows_out.append(_not_present_row(activity, duration_sec, rubric))
            continue

        present_rows.append(
            {
                "activity": activity,
                "duration_sec": duration_sec,
                "rubric_snippet": rubric.get(activity, {}),
                "exchange": exchange,
            }
        )

    if not present_rows:
        out = pd.DataFrame(rows_out)
        out["activity_order"] = pd.Categorical(out["activity"], categories=activity_order, ordered=True)
        return out.sort_values("activity_order").drop(columns=["activity_order"]).reset_index(drop=True)

    items_json = json.dumps(present_rows, ensure_ascii=False, sort_keys=True)
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    parsed = _evaluate_present_items_cached(items_json, metadata_json)

    by_activity = {
        str(x.get("activity", "")).strip(): x
        for x in parsed.get("results", [])
        if isinstance(x, dict)
    }

    for item in present_rows:
        activity = item["activity"]
        duration_sec = item["duration_sec"]

        x = by_activity.get(activity)
        if not x:
            rows_out.append(_not_present_row(activity, duration_sec, rubric))
            continue

        score = float(x.get("efficacy_score", 0.0))
        if 0 <= score <= 1:
            score *= 100.0

        rows_out.append(
            {
                "activity": activity,
                "present": True,
                "duration_sec": float(duration_sec or 0.0),
                "efficacy_score": score,
                "detailed_observations": str(x.get("detailed_observations", "")),
                "tactical_feedback": str(x.get("tactical_feedback", "")),
            }
        )

    out = pd.DataFrame(rows_out)
    out["activity_order"] = pd.Categorical(out["activity"], categories=activity_order, ordered=True)
    out = out.sort_values("activity_order").drop(columns=["activity_order"]).reset_index(drop=True)
    return out
