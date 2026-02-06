Multi-call-type demo bundle

Call types included:
- cancellations (2 calls: stellar and poor)
- billing_inquiry (2 calls)
- move_request (1 call)
- internet_troubleshooting (2 calls)

Key reference paths (bundle-root-relative):
- CALL INDEX: data/calls_index.csv
- TRANSCRIPTS: data/*.csv
- ARTIFACTS: <call_type>/Artifacts/{enum.json, call_flow_context.json, call_flow_rubric.json}

Portability note:
- transcript_path values in calls_index.csv are RELATIVE (data/<file>.csv).
- Your app should resolve transcript_path relative to the bundle root folder (same folder that contains data/).

Run (from this folder):
  pip install -r requirements.txt
  $env:CALL_INDEX_PATH="data/calls_index.csv"   # PowerShell
  streamlit run app.py

OFFLINE MODE:
- This build is offline-only (no API calls).
- It returns deterministic, template-based outputs that satisfy the app UI.
- Optional: set PATHFINDER_OFFLINE=1 (default).
