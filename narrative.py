import json
import pandas as pd
from llm_client import get_openai


def build_narrative_messages(enum: list[str], rubric: dict, eval_df: pd.DataFrame, metadata: dict):
    system = {
        "role": "system",
        "content": (
            "You are an expert call QA analyst writing succinct, executive-friendly insights. "
            "Use ONLY the provided evaluation content; do not invent facts."
        ),
    }

    payload = eval_df[
        ["activity", "duration_sec", "efficacy_score", "detailed_observations", "tactical_feedback"]
    ].to_dict("records")

    user = {
        "role": "user",
        "content": f"""
<RULES>
1) Use ONLY the evaluation data provided in <EVAL_DATA>. Do not add new facts.
2) Be concise, structured, and specific.
3) No nulls.
</RULES>

<METADATA>
{json.dumps(metadata)}
</METADATA>

<ENUM_ORDER>
{json.dumps(enum)}
</ENUM_ORDER>

<RUBRIC>
{json.dumps(rubric)}
</RUBRIC>

<EVAL_DATA>
{json.dumps(payload)}
</EVAL_DATA>

<OUTPUT>
Return JSON:
{{
  "activity_narrative": [
    {{
      "activity": "string",
      "themes": ["string","string","string"],
      "supporting_evidence": "string",
      "recommendations": "string"
    }}
  ],
  "agent_narrative": {{
    "observations_gaps": "string",
    "tactical_feedback": "string"
  }}
}}
</OUTPUT>
""".strip(),
    }
    return [system, user]


def generate_narrative(enum: list[str], rubric: dict, eval_df: pd.DataFrame, metadata: dict) -> dict:
    client, model = get_openai()

    schema = {
        "name": "narrative_insights",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "activity_narrative": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "activity": {"type": "string", "enum": enum},
                            "themes": {"type": "array", "items": {"type": "string"}},
                            "supporting_evidence": {"type": "string"},
                            "recommendations": {"type": "string"},
                        },
                        "required": ["activity", "themes", "supporting_evidence", "recommendations"],
                    },
                },
                "agent_narrative": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "observations_gaps": {"type": "string"},
                        "tactical_feedback": {"type": "string"},
                    },
                    "required": ["observations_gaps", "tactical_feedback"],
                },
            },
            "required": ["activity_narrative", "agent_narrative"],
        },
    }

    resp = client.chat.completions.create(
        model=model,
        messages=build_narrative_messages(enum, rubric, eval_df, metadata),
        response_format={"type": "json_schema", "json_schema": schema},
    )
    return json.loads(resp.choices[0].message.content)
