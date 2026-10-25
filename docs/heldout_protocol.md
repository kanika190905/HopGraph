# Held-out confirmation protocol (written before any of these runs)

*Committed 2026-10-25, before the train confirmation and the held-out evaluation were run.*

## Step 1: v2 confirmation on train

- Data: `configs/musique_tune500_tuned.yaml` (500 MuSiQue **train** questions, tuned v1 settings).
- **v1 reference:** the frozen tuned configuration.
- **v2 candidate:** v1 + `linking.use_ngram=false` + `graph.mention_weight=count`. These two changes each improved graph R@5 in the single-change ablations (`docs/error_analysis.md` §5); their combination is untested.
- Systems: hybrid, graph, graph_ce (cross-encoder depth 50, as tuned).
- **Adopt v2 if and only if both hold:**
  1. Δ graph R@5 (v2 − v1) has a 95% paired-bootstrap CI with lower bound > 0, **and**
  2. Δ graph_ce R@5 (v2 − v1) ≥ 0 (point estimate), so the reranked headline system does not get worse.
  
  Otherwise keep v1. No other variant is considered.

## Step 2: single held-out evaluation

- **Questions:** the MuSiQue-Ans dev questions at positions 1,000–2,416 of the same seed-0 permutation. These are the 1,417 questions **not** in the dev-1,000 sample that was evaluated twice. Their candidates form their own merged corpus. The build must confirm that no question ID overlaps with dev-1,000.
- **Configuration:** the one selected in Step 1, frozen, with no further changes.
- **Systems:** bm25, dense, hybrid, hybrid_ce, graph, graph_only, graph_ce. Latency is measured on the first 200 questions (5 repeats, 10 warm-up).
- **Primary comparisons** (Holm-adjusted, R@5): graph_ce − hybrid_ce (headline), graph − hybrid, graph_only − hybrid.
- **Pre-registered secondary check:** graph − hybrid_ce (R@5). This was exploratory on dev-1,000 (+2.5 [+1.1, +3.9]); here it is tested as a stated hypothesis, unadjusted.
- The evaluation runs **once**. Whatever it shows is reported, including a smaller or absent effect. Nothing is tuned afterwards on these questions.
