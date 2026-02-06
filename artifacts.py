import json
import streamlit as st


@st.cache_data
def load_artifacts(call_type: str) -> tuple[list[str], dict, dict]:
    with open(f"{call_type}/Artifacts/enum.json", "r") as f:
        enum = json.load(f)["activities"]
    with open(f"{call_type}/Artifacts/call_flow_context.json", "r") as f:
        ctx = json.load(f)["call_flow_context"]
    with open(f"{call_type}/Artifacts/call_flow_rubric.json", "r") as f:
        rubric = json.load(f)["call_flow_rubric"]
    return enum, ctx, rubric
