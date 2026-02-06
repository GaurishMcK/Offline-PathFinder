import json
import pandas as pd
from llm_client import get_openai
from text_utils import sanitize_text, segments_to_plaintext


def build_cls_messages(ctx: dict, segments_text: str, metadata: dict, extra_context: dict | None = None):
    system = {
        "role": "system",
        "content": (
            "You are an expert telco call-center activity classifier. "
            "Classify each segment into exactly one activity from the call flow. "
            "Use only transcript content and call flow context. No nulls."
        ),
    }

    extra_context = extra_context or {}

    user = {
        "role": "user",
        "content": f"""
<RULES>
1) Use ONLY transcript content + call flow context.
2) If ambiguous, maintain prior activity.
3) No nulls; choose best-fit activity.
</RULES>

<METADATA>
{json.dumps(metadata)}
</METADATA>

<CALL_FLOW_CONTEXT>
{json.dumps(ctx)}
</CALL_FLOW_CONTEXT>

<ADDITIONAL_CONTEXT>
{json.dumps(extra_context)}
</ADDITIONAL_CONTEXT>

<SEGMENTS>
{segments_text}
</SEGMENTS>
""".strip(),
    }
    return [system, user]


def classify_call(
    transcript_df: pd.DataFrame,
    enum: list[str],
    ctx: dict,
    metadata: dict,
    extra_context: dict | None = None,
) -> pd.DataFrame:
    client, model = get_openai()

    df = transcript_df.copy()
    df["text"] = df["text"].apply(sanitize_text)
    segments_text = segments_to_plaintext(df)

    schema = {
        "name": "activity_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "conversation_id": {"type": "string"},
                "transcript_id": {"type": "string"},
                "emp_id": {"type": "integer"},
                "per_segment": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "phrase_rank": {"type": "integer"},
                            "activity": {"type": "string", "enum": enum},
                        },
                        "required": ["phrase_rank", "activity"],
                    },
                },
            },
            "required": ["conversation_id", "transcript_id", "emp_id", "per_segment"],
        },
    }

    resp = client.chat.completions.create(
        model=model,
        messages=build_cls_messages(ctx, segments_text, metadata, extra_context=extra_context),
        response_format={"type": "json_schema", "json_schema": schema},
    )
    parsed = json.loads(resp.choices[0].message.content)

    out = pd.DataFrame(parsed["per_segment"])
    out["conversation_id"] = metadata["conversation_id"]
    out["transcript_id"] = metadata["transcript_id"]
    out["emp_id"] = metadata["emp_id"]
    return out
