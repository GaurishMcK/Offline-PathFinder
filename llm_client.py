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
        """Offline intent mapping.
    
        Must satisfy intent_batch.py schemas:
          - intent_mapping: {"L1": str, "L2": str, "Sentiment": "Positive|Negative|Neutral"}
          - intent_mapping_bulk: {"results": [{"i": int, "L1": str, "L2": str, "Sentiment": ...}, ...]}
        """
        import json, re
        from typing import Any, Dict, List, Tuple
    
        # ---- Helpers ---------------------------------------------------------
        def safe_first_l1_l2(tree: Dict[str, List[str]]) -> Tuple[str, str]:
            if not tree:
                return ("Other", "Other")
            l1 = sorted(tree.keys())[0]
            l2s = tree.get(l1) or ["Other"]
            l2 = sorted(set(l2s))[0]
            return (l1, l2)
    
        def classify_sentiment(text: str) -> str:
            t = (text or "").lower()
            if any(k in t for k in ["angry", "upset", "frustrat", "not happy", "terrible", "worst", "unacceptable"]):
                return "Negative"
            if any(k in t for k in ["thank", "thanks", "appreciate", "great", "awesome", "perfect", "love"]):
                return "Positive"
            return "Neutral"
    
        def map_to_tree(text: str, tree: Dict[str, List[str]]) -> Tuple[str, str]:
            """
            Keyword-ish mapping, BUT it only returns values that exist in the provided intent_tree.
            If it can't find a good match, it returns the first (L1,L2) from the tree.
            """
            t = (text or "").lower()
            default_l1, default_l2 = safe_first_l1_l2(tree)
    
            # Build a quick searchable list of (l1,l2) pairs that are valid.
            pairs: List[Tuple[str, str]] = []
            for l1, l2s in (tree or {}).items():
                for l2 in (l2s or []):
                    pairs.append((str(l1), str(l2)))
    
            if not pairs:
                return (default_l1, default_l2)
    
            # If any L1/L2 strings literally appear in text, pick that.
            for l1, l2 in pairs:
                if l2 and l2.lower() in t:
                    return (l1, l2)
            for l1, _ in pairs:
                if l1 and l1.lower() in t:
                    # pick first L2 under that L1
                    l2s = tree.get(l1) or ["Other"]
                    return (l1, sorted(set(l2s))[0])
    
            # Lightweight generic keywords -> try to land in a matching part of the tree
            keyword_buckets = [
                (["cancel", "disconnect", "terminate", "close account"], ["cancel", "disconnect", "terminate"]),
                (["bill", "billing", "charge", "charged", "payment", "refund"], ["bill", "charge", "payment", "refund"]),
                (["move", "moving", "new address", "transfer service", "relocation"], ["move", "address", "transfer"]),
                (["internet", "router", "wifi", "wi-fi", "modem", "no connection", "down"], ["internet", "wifi", "router", "modem"]),
            ]
    
            for text_keys, intent_keys in keyword_buckets:
                if any(k in t for k in text_keys):
                    # find a pair whose L1 or L2 contains any of the intent keys
                    for l1, l2 in pairs:
                        l1l = l1.lower()
                        l2l = l2.lower()
                        if any(ik in l1l or ik in l2l for ik in intent_keys):
                            return (l1, l2)
    
            # fallback: guaranteed valid for BOTH open-ended and close-ended trees
            return (default_l1, default_l2)
    
        # ---- Extract the intent tree from the prompt (works for both modes) ---
        intent_tree = OfflineClient._extract_json_block("INTENT_TREE", user_text)
        if not isinstance(intent_tree, dict):
            intent_tree = {}
    
        # ---- Single mapping ---------------------------------------------------
        if name == "intent_mapping":
            # sometimes prompt may include <TRANSCRIPT> JSON; if present, use it
            transcript = OfflineClient._extract_json_block("TRANSCRIPT", user_text)
            text = user_text if transcript is None else json.dumps(transcript)
            l1, l2 = map_to_tree(text, intent_tree)
            return {"L1": l1, "L2": l2, "Sentiment": classify_sentiment(text)}
    
        # ---- Bulk mapping (intent_mapping_bulk) -------------------------------
        payload = OfflineClient._extract_json_block("BATCH", user_text)
    
        if payload is None:
            payload = OfflineClient._extract_json_block("TRANSCRIPTS", user_text)
    
        if payload is None:
            # last-resort: grab a JSON list in the prompt
            m = re.search(r"(\[\s*\{.*\}\s*\])", user_text, flags=re.DOTALL)
            payload = json.loads(m.group(1)) if m else []
    
        results: List[Dict[str, Any]] = []
        for row in (payload or []):
            # IMPORTANT: i MUST match the 1-based S No. used in pathfinder_v4.py
            i = int(row.get("i") or row.get("id") or row.get("sample_id") or 0)
            text = row.get("transcript") or row.get("text") or ""
            l1, l2 = map_to_tree(text, intent_tree)
            results.append({"i": i, "L1": l1, "L2": l2, "Sentiment": classify_sentiment(text)})
    
        results.sort(key=lambda r: int(r.get("i", 0)))
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
