"""TATM — Temporal Arbitration via Time-State Mediation.

Diagnostic framework for analyzing temporal knowledge conflicts in LLMs.
Part of the three-phase pipeline:

    Phase 1  RevisionReplayQA  (data foundation — see fact_timeline/)
    Phase 2  TATM              (mechanistic diagnosis — this package)
    Phase 3  Align-then-Answer (inference-time intervention — TBD)

Submodules
----------
model           Model loading, prompt/tokenization utilities (incl. get_first_answer_token)
hooks           TransformerLens hook operations (attention extraction, knockout)
sat_probe       SAT Probe for F1 diagnosis (attention → logistic regression)
logit_lens      Layer-by-layer vocabulary projection (shared by F2-b / F3-a)
f2_diagnosis    F2 diagnosis: STR patching, RouteScore, B5-vs-B6 analysis
f3_diagnosis    F3 diagnosis: Logit-Lens trajectory (F3-a), attribution patching
                bridge (F3-0.5), routing-causality ablation (F3-b), and 2x2
                Override / Chain / Content (F3-c) with title-resolution policy.
"""
