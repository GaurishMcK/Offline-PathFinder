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
