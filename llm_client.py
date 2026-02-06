"""LLM client abstraction.

This repo originally used OpenAI's Chat Completions API. For demo / offline use-cases,
we ship a tiny in-process "OfflineClient" that returns deterministic, pre-baked JSON
responses that satisfy the app's json_schema contracts.

Goal: ZERO network / API calls.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import streamlit as st


# ---------------------------
# Tiny response objects (OpenAI-like)
# ---------------------------

@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Resp:
    choices: List[_Choice]


class OfflineClient:
    """Drop-in-ish replacement for OpenAI() with just what this app needs."""

    class chat:  # noqa: N801
        class completions:  # noqa: N801
            @staticmethod
            def create(model: str, messages: List[Dict[str, str]], response_format: Dict[str, Any]):
                schema = (response_format or {}).get("json_schema", {})
                name = schema.get("name", "")
                # User prompt is always the last message in this codebase
                user_text = (messages[-1] or {}).get("content", "") if messages else ""

                if name == "activity_classification":
                    payload = OfflineClient._handle_activity_classification(schema, user_text)
                elif name == "bulk_activity_evaluation":
                    payload = OfflineClient._handle_bulk_activity_evaluation(user_text)
                elif name == "narrative_insights":
                    payload = OfflineClient._handle_narrative(user_text)
                elif name == "intent_tree_discovery":
                    payload = OfflineClient._handle_intent_tree(user_text)
                elif name in {"intent_mapping", "intent_mapping_bulk"}:
                    payload = OfflineClient._handle_intent_mapping(name, user_text)
                else:
                    # Safe fallback: return an empty object that won't crash json.loads.
                    payload = {}

                return _Resp(choices=[_Choice(message=_Msg(content=json.dumps(payload)))])
    # ---------------------------
    # Handlers
    # ---------------------------

    @staticmethod
    def _extract_json_block(tag: str, text: str) -> Any | None:
        """Extract JSON between <TAG>...</TAG>. Returns parsed object or None."""
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL)
        if not m:
            return None
        blob = m.group(1).strip()
        try:
            return json.loads(blob)
        except Exception:
            # Sometimes there is extra prose before the JSON; try last JSON object/array
            m2 = re.search(r"(\{.*\}|\[.*\])\s*$", blob, flags=re.DOTALL)
            if m2:
                try:
                    return json.loads(m2.group(1))
                except Exception:
                    return None
        return None

    @staticmethod
    def _handle_activity_classification(schema: Dict[str, Any], user_text: str) -> Dict[str, Any]:
        # Get enum list from schema
        try:
            enum = schema["schema"]["properties"]["per_segment"]["items"]["properties"]["activity"]["enum"]
        except Exception:
            enum = []

        # Parse segment headers like: "#12 | speaker:AGENT | 123->456 ms"
        ranks = [int(x) for x in re.findall(r"^#(\d+)\s*\|", user_text, flags=re.MULTILINE)]
        if not ranks:
            # reasonable default to avoid crashes
            ranks = list(range(1, 9))

        n = len(ranks)
        m = max(1, len(enum))

        # Keyword nudges (very light): useful for "looks real" demos
        keyword_to_activity = {
            "greet": "GREETING",
            "thank you for calling": "GREETING",
            "verify": "AUTHENTICATION",
            "authentication": "AUTHENTICATION",
            "address": "COLLECT_NEW_ADDRESS",
            "move": "INTENT_CONFIRMATION",
            "transfer": "SCHEDULE_TRANSFER",
            "schedule": "SCHEDULE_TRANSFER",
            "date": "CONFIRM_MOVE_DATE",
            "resolved": "RESOLUTION",
            "anything else": "CALL_CLOSING",
            "have a great": "CALL_CLOSING",
        }

        # Grab text per segment
        seg_texts: Dict[int, str] = {}
        parts = re.split(r"^#(\d+)\s*\|.*$", user_text, flags=re.MULTILINE)
        # parts pattern: [preamble, rank1, body1, rank2, body2, ...]
        for i in range(1, len(parts), 2):
            try:
                r = int(parts[i])
                body = parts[i + 1].strip()
                seg_texts[r] = body.lower()
            except Exception:
                continue

        # Baseline: chunk into ordered activities
        chunk = max(1, round(n / m))
        baseline = {}
        for idx, r in enumerate(ranks):
            a = enum[min(m - 1, idx // chunk)] if enum else "UNKNOWN"
            baseline[r] = a

        # Apply keyword overrides where the activity exists in enum
        out = []
        for r in ranks:
            text_l = seg_texts.get(r, "")
            chosen = baseline[r]
            for kw, act in keyword_to_activity.items():
                if kw in text_l and (not enum or act in enum):
                    chosen = act
                    break
            out.append({"phrase_rank": r, "activity": chosen})

        # Metadata placeholders (the caller overwrites with real metadata anyway)
        return {
            "conversation_id": "offline",
            "transcript_id": "offline",
            "emp_id": 0,
            "per_segment": out,
        }

    @staticmethod
    def _handle_bulk_activity_evaluation(user_text: str) -> Dict[str, Any]:
        items = OfflineClient._extract_json_block("ITEMS", user_text) or []
        results = []
        for it in items:
            act = str(it.get("activity", "")).strip() or "UNKNOWN"
            exchange = str(it.get("exchange", ""))
            # Super deterministic: longer exchanges look "better"
            score = 88 if len(exchange) > 300 else 78 if len(exchange) > 120 else 70
            results.append(
                {
                    "activity": act,
                    "efficacy_score": float(score),
                    "detailed_observations": (
                        f"Offline demo score for {act}. "
                        "This build uses deterministic, pre-defined text (no model inference). "
                        "Observations are based only on exchange length, not semantics."
                    ),
                    "tactical_feedback": (
                        f"For {act}: keep it crisp, confirm the customer's intent, and avoid dead air. "
                        "(Static demo coaching — replace with real QA logic when re-enabling LLMs.)"
                    ),
                }
            )
        return {"results": results}

    @staticmethod
    def _handle_narrative(user_text: str) -> Dict[str, Any]:
        eval_data = OfflineClient._extract_json_block("EVAL_DATA", user_text) or []
        enum_order = OfflineClient._extract_json_block("ENUM_ORDER", user_text) or []
        # Preserve enum order if possible
        by_act = {d.get("activity"): d for d in eval_data if isinstance(d, dict)}
        ordered = [by_act[a] for a in enum_order if a in by_act] + [d for d in eval_data if d.get("activity") not in enum_order]

        activity_narr = []
        for d in ordered:
            act = d.get("activity", "UNKNOWN")
            score = d.get("efficacy_score", 0)
            activity_narr.append(
                {
                    "activity": act,
                    "themes": [
                        "Static offline insights",
                        "Deterministic scoring",
                        "Template-based coaching",
                    ],
                    "supporting_evidence": f"Efficacy score recorded: {score} (offline demo).", 
                    "recommendations": "Use this as a UI demo. For real analysis, wire back a model or a rule engine.",
                }
            )

        return {
            "activity_narrative": activity_narr,
            "agent_narrative": {
                "observations_gaps": "This is an offline/static build; narratives are template-generated.",
                "tactical_feedback": "If you want 'better-than-static' without APIs, add a ruleset + keyword features.",
            },
        }

    @staticmethod
    def _handle_intent_tree(user_text: str) -> Dict[str, Any]:
        # Fixed tree that works for typical telco demos
        tree = [
            {"L1": "Billing", "L2": ["Charges", "Payment", "Other"]},
            {"L1": "Cancellation", "L2": ["Service Cancel", "Retention", "Other"]},
            {"L1": "Move Request", "L2": ["New Address", "Transfer Date", "Other"]},
            {"L1": "Tech Support", "L2": ["Internet", "Equipment", "Other"]},
        ]
        return {"intent_tree": tree}

    @staticmethod
    def _handle_intent_mapping(name: str, user_text: str) -> Dict[str, Any]:
        # Bulk mapping expects an array of results; single expects one
        def classify(text: str) -> Tuple[str, str]:
            t = (text or "").lower()
            if "cancel" in t or "disconnect" in t:
                return "Cancellation", "Service Cancel"
            if "bill" in t or "charge" in t or "payment" in t:
                return "Billing", "Charges"
            if "move" in t or "address" in t:
                return "Move Request", "New Address"
            if "internet" in t or "router" in t or "wifi" in t:
                return "Tech Support", "Internet"
            return "Billing", "Other"

        if name == "intent_mapping":
            transcript = OfflineClient._extract_json_block("TRANSCRIPT", user_text)
            # In this codebase the transcript is not tagged; fall back to searching whole prompt
            l1, l2 = classify(user_text if transcript is None else json.dumps(transcript))
            return {"L1": l1, "L2": l2, "sentiment": "Neutral"}

        # intent_mapping_bulk: find transcripts payload (a JSON list)
        payload = OfflineClient._extract_json_block("TRANSCRIPTS", user_text)
        if payload is None:
            # fall back: parse last JSON list
            m = re.search(r"(\[\s*\{.*\}\s*\])\s*</", user_text, flags=re.DOTALL)
            payload = json.loads(m.group(1)) if m else []

        results = []
        for row in payload or []:
            tid = row.get("transcript_id") or row.get("id") or row.get("sample_id") or ""
            text = row.get("transcript") or row.get("text") or ""
            l1, l2 = classify(text)
            results.append({"transcript_id": tid, "L1": l1, "L2": l2, "sentiment": "Neutral"})
        return {"results": results}


# ---------------------------
# Public factory used by the rest of the app
# ---------------------------

@st.cache_resource
def get_openai() -> Tuple[OfflineClient, str]:
    # Default to offline unless explicitly disabled.
    offline = os.environ.get("PATHFINDER_OFFLINE", "1").strip().lower() not in {"0", "false", "no"}
    if offline:
        return OfflineClient(), "offline"

    # Hard stop: this repo version is meant to be offline-only.
    raise RuntimeError(
        "Online/OpenAI mode is disabled in this build. Set PATHFINDER_OFFLINE=1 (default). "
        "If you want to re-enable API calls, restore the original llm_client.py."
    )
