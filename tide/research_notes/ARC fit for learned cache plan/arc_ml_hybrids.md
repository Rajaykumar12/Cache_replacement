# ARC + Machine Learning Hybrids: Concrete Integration Options for the Option-B Learned Cache (as of Sept 2026)

Scope note: ARC base mechanics and benchmark standing are covered by another researcher. These notes cover only ARC/ARC-style hybrids with learning. "Preprint" = arXiv only, not peer reviewed as of 2026-09-27. Items marked **[INFERENCE]** have no paper directly behind them.

## Q1. Learned or adaptive control of ARC's p (RL, bandits, gradient, LAH-style); do "learned ARC" papers exist?

### Takeaway
I found **no peer-reviewed or arXiv paper that learns or RL-controls ARC's p directly**. Searches for "ML-ARC", "DeepARC", "RL-ARC" and "RL controlling ARC's T1/T2 target" found nothing relevant; the only "ARC-RL" hit is an unrelated robotics paper. The closest work either replaces ARC's p rule with regret minimization over experts (LeCaR/CACHEUS, Q2), or learns cache-level parameters of a *different* heuristic (LAH / S4-FIFO learns S3-FIFO's parameters, not ARC's). LAH explicitly names ARC's recency/frequency balance as a parameter the same approach could learn, and argues that ARC's per-miss p update is fragile.

### Cited Findings
- ARC already has a built-in online learner: the ARC paper calls the p increments on B1/B2 ghost hits "learning rates," and their size depends on the sizes of the ghost lists (δ1 = max(|B2|/|B1|, 1) on a B1 hit, δ2 = max(|B1|/|B2|, 1) on a B2 hit). — [Megiddo & Modha, ARC, FAST '03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
- **LAH / S4-FIFO** (Xia, Nixon, Marthen, Bhandari, Juncheng Yang; arXiv 2608.27975, Aug 2026, **preprint**):
  - Architecture: the data plane runs a deterministic S3-FIFO variant. The control plane asynchronously runs a pre-trained GBDT (20 trees, depth 9) that picks one of 18 representative parameter configurations. Tuned parameters: small-queue ratio in [0.05, 0.90], ghost ratio in [0.90, 6.00], small-skip ratio, and promotion thresholds.
  - Features: 73 cache-level features, mostly hit-position histograms per queue, plus one-hit ratio, scan intensity, ghost pressure and similar.
  - Prediction timing: one prediction after a 20% observation window; periodic re-prediction is possible.
  - Cost: inference under 2 ms, about 73 KB of state, and no per-object metadata (LRB needs 200+ bytes per object).
  - — [arXiv 2608.27975](https://arxiv.org/html/2608.27975)
- LAH classifies ARC as a "cache-level, per-miss adaptive" algorithm and criticizes it on two points. First, "The default move one slot per ghost hit approach in ARC has pathological failure cases." Second, its Figure 3 argues that "choosing the right parameter is much more important than per-miss adaptivity." LAH says "ARC exposes the balance between its recency and frequency regions" as a parameter the LAH principle generalizes to, but it **only instantiates S3-FIFO**. It does not build a learned-ARC. — [arXiv 2608.27975](https://arxiv.org/html/2608.27975)
- LAH vs ARC on production traces (trained on 4,140 traces, evaluated on 1,035):
  - At the large cache size, S4-FIFO gets "nearly double ARC's improvement over FIFO."
  - Overall it is +26% mean efficiency over S3-FIFO and +8% over 3L-Cache.
  - Worst case, it raises FIFO's miss ratio by only 0.8% (large cache) or 0.2% (small cache). For comparison, LRB raises it by 20–72% and 3L-Cache by 8.8%.
  - — [arXiv 2608.27975](https://arxiv.org/html/2608.27975)
- The only recent ARC-with-adaptive-split paper I found is **not ML**. It ports plain ARC into vLLM's KV cache (Shen, Madhyastha, Underwood, Nicolae, Burns; arXiv 2606.21238, **preprint**). Against LRU it gains up to 10.8% hit rate and 12.6% lower TTFT on synthetic QA, and 2.1% hit rate / 2.0% TTFT on real conversation traces. Against DBL (a fixed 1:1 split) it shows that adaptive p beats a static split, especially for large caches and long workload shifts. — [arXiv 2606.21238](https://arxiv.org/html/2606.21238v1)
- The search for an RL/DRL controller over ARC partitions returned an ARC-named RL paper about robotics (ARC Raiders, arXiv 2605.19503), which is unrelated. It also returned a student GitHub project that benchmarks an online RL policy *against* ARC (not a hybrid, not a paper). — [arXiv 2605.19503](https://arxiv.org/abs/2605.19503); [estersassis/adaptive-rl-cache](https://github.com/estersassis/adaptive-rl-cache)

### Inferences
- **[INFERENCE]** A "learned-ARC" in the LAH style is straightforward but unpublished. You could keep ARC's data plane and have a slow control plane either (a) clamp p to a learned band [p_lo, p_hi], or (b) replace the δ step size with a learned per-regime step. Features would be ARC-native: T1/T2 hit-position histograms, B1/B2 ghost-hit rates, one-hit ratio. This mirrors the LAH evidence that picking the right regime beats per-miss adaptation, but nobody has measured it on ARC.
- **[INFERENCE]** For Option B this is a *fallback-track* idea, not the main learner. Its value is that the controller is interpretable (one scalar) and it inherits ARC's robustness.

### Gaps
- No paper found that tunes ARC's p (or δ) with RL, bandits or gradient methods. A search snippet claimed a "DQN/PPO extension ... 1.55–3.29× reward over greedy" for partition control, but I could not tie it to a verifiable source, so it is excluded.
- LAH's "pathological failure cases" of ARC are asserted in the text. I did not extract the specific trace or pattern.

## Q2. LeCaR, CACHEUS, ALeCaR/OLeCaR and the H-MC linear-regret critique

### Takeaway
LeCaR and CACHEUS are the canonical "ARC-inspired ML" line. They replace ARC's deterministic p rule with regret-minimizing weights over two experts (LRU/LFU, or SR-LRU/CR-LFU). Penalties come from history lists that act exactly like ARC's ghost lists.

- **Gains over ARC:** large at tiny caches, smaller or neutral at large caches.
- **Theory is now contested.** A 2020 JMLR paper claims vanishing regret for an adapted LeCaR (OLeCaR). A Sept 2026 preprint proves LeCaR has *linear* regret against min(LRU, LFU) and proposes H-MC (Hedge + maximal coupling) with O(√T) regret. H-MC has no experiments.

### Cited Findings
- **LeCaR** (Vietri et al., HotStorage 2018):
  - Mechanics: on each eviction it picks LRU or LFU at random, with probability proportional to weights. Each expert has a history list H_LRU / H_LFU. A miss found in an expert's history counts as that expert's "poor" decision and lowers its weight exponentially. Total history size equals cache size "(as with ARC)."
  - — [LeCaR, HotStorage '18](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf)
- LeCaR's framing relative to ARC: ARC "reduced eviction to a choice between LRU and a version of LFU that does not differentiate between entries that are accessed more than twice." LeCaR also notes that "replacing LRU with ARC also lowers the efficacy of LeCaR," which it attributes to lost orthogonality between experts. — [LeCaR, HotStorage '18](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf)
- LeCaR results vs ARC on 8 FIU production traces: up to 18× higher hit rate when the cache is 0.1% of the workload, −4% to +30.5% at 1%, and within 0.33% at larger caches. LeCaR's metadata overhead is 3× ARC's when implemented naively and 2× when implemented carefully. Giving ARC 2× metadata does not help it. — [LeCaR, HotStorage '18](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf)
- **CACHEUS** (Rodriguez et al., FAST '21):
  - Changes from LeCaR: drops LeCaR's discount rate and adapts the learning rate by gradient-based stochastic hill-climbing on windowed hit rate (reset after 10 consecutive degrading windows, initialized randomly in [10⁻³, 1]).
  - Experts: exactly two. More than two experts "was significantly worse."
  - SR-LRU is scan-resistant, with an ARC-like adaptive partition (SR/R) and history. CR-LFU is LFU with MRU tie-breaking to resist churn.
  - — [CACHEUS, FAST '21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- CACHEUS's critique of ARC's adaptation: when a scan phase is followed by a churn phase, "ARC continues to evict from T1 and behaves similar to LRU." It also says ARC's LRU-ordered T2 cannot capture the full frequency distribution. — [CACHEUS, FAST '21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- CACHEUS results (329 traces, cache sizes from 0.05% to 10% of footprint), for variant C3 = CACHEUS(SR-LRU, CR-LFU):
  - It is best or indistinguishable from best in 87% of workload–cache combinations.
  - It is distinctly best in 47%.
  - The best case is +38.32% hit rate vs ARC (CloudPhysics, 10% cache). The worst case is −15.12% vs DLIRS (MSR, 5% cache).
  - On webmail day 16 (10% cache), hit rates are 30.08% for ARC, 40.71% for LIRS, 42.08% for LeCaR and 43.95% for C3. That is +46.11% relative to ARC.
  - The variant with ARC as one expert, C1 = CACHEUS(ARC, LFU), was worse than the best state of the art in 39% of combinations.
  - — [CACHEUS, FAST '21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- CACHEUS overhead: about 2N metadata (N resident plus N history), which it says is equivalent to ARC and LIRS. Compute is bounded by LFU's O(log N). — [CACHEUS, FAST '21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- **OLeCaR / EXP4-DFDC** (Yusuf, Stebliankin, Vietri, Narasimhan, JMLR 21, 2020): formalizes LeCaR as a multi-armed bandit with delayed feedback and decaying costs. Feedback arrives when an evicted page is re-requested while still in history, and is ignored beyond the history length. The paper derives an optimal learning rate and claims "LeCaR and OLeCaR are theoretically guaranteed to have vanishing regret over time." — [JMLR / arXiv 2009.11330](https://arxiv.org/abs/2009.11330)
- **ALeCaR** exists only as a code repo ("Adaptive Learning Cache Replacement (Yusuf et al, 2021)", variants ALeCaR0–8). I could not locate an associated paper or any ARC comparison. — [GitHub movingpictures83/ALeCaR](https://github.com/movingpictures83/ALeCaR)
- **H-MC critique** (Ben Mazziane & Zou; arXiv 2609.07566, Sept 2026, **preprint, theory only, no experiments**):
  - Negative result (Thm 3.1): for cache size C=2 and any history length k≥1, LeCaR suffers Ω(T) regret against min(LRU, LFU). The adversarial construction uses periodic phases in which LRU and LFU miss equally, but LeCaR oscillates between them.
  - H-MC: treats LRU and LFU as experts that recommend *full cache states*, uses Hedge for marginals and maximal coupling for switches, and achieves O(√T) regret with O(√T) switching cost.
  - Cost: it has to simulate both virtual caches.
  - ARC is only cited, not analyzed.
  - — [arXiv 2609.07566](https://arxiv.org/html/2609.07566)
- Conflict: the JMLR 2020 vanishing-regret claim and the 2609.07566 linear-regret claim appear to measure against different benchmarks. — [JMLR 2020](https://arxiv.org/abs/2009.11330) vs [arXiv 2609.07566](https://arxiv.org/html/2609.07566)

### Inferences
- **[INFERENCE]** The two regret results probably do not formally contradict each other. EXP4-DFDC bounds regret on its per-eviction delayed-feedback loss, while H-MC measures regret on the misses of the full LRU/LFU trajectories, where history-based penalties can mis-credit experts. The report should present this as "contested / benchmark-dependent," not as settled.
- **[INFERENCE]** For Option B, LeCaR/CACHEUS are more useful as a *baseline to beat* and as a *design template for ghost-list credit assignment* than as a component. They show that a two-expert, ghost-driven learner reliably beats ARC at small cache/working-set ratios, which is where a DQN-replacement project will likely be evaluated.

### Gaps
- No head-to-head of H-MC vs ARC or CACHEUS exists (the paper has no experiments).
- No independent 2024–2026 reproduction of CACHEUS vs ARC turned up in this search. LAH evaluates LeCaR but not CACHEUS, and I did not extract its LeCaR numbers.

## Q3. RL/bandit as a controller over heuristic structures vs per-object value learning

### Takeaway
Evidence consistently favors **controllers over heuristic structure** over per-object RL:

- Low-dimensional controllers work: expert weights (LeCaR/CACHEUS), cache-level parameters (LAH), or re-ranking a small heuristic candidate set (HALP, Cold-RL).
- They are cheaper and have better worst cases than per-object learners like LRB.
- No paper directly benchmarks "RL setting ARC's p" against per-object RL. The evidence is indirect.

### Cited Findings
- LAH argues that object-level learners suffer "objective mismatch, where improvements in prediction metrics do not translate into fewer misses." Cache-level learning "directly optimizes for the miss ratio," uses features "less susceptible to perturbations and noise," and selects "from a validated set of safe configurations." — [arXiv 2608.27975](https://arxiv.org/html/2608.27975) (preprint)
- The same paper's robustness data, measured as the worst-case trace increase in FIFO miss ratio: LRB (per-object learned) +20–72%, 3L-Cache +8.8%, S4-FIFO (cache-level learned) +0.8% / +0.2%. — [arXiv 2608.27975](https://arxiv.org/html/2608.27975)
- LeCaR/CACHEUS use a 2-arm online learner over heuristic experts and beat ARC at small caches (numbers in Q2). — [LeCaR](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf); [CACHEUS](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- **Cold-RL** (Gupta & Bhayani; arXiv 2508.12485, **preprint**):
  - Design: a dueling DQN trained *offline* re-ranks the K least-recently-used objects, using 6 features (age, size, hit count, inter-arrival, TTL, origin RTT). It has a 500 µs timeout that falls back to LRU, and adds under 2% CPU.
  - Hit ratio vs ARC:

    | Cache size | ARC | Cold-RL |
    |---|---|---|
    | 25 MB | 0.144 | 0.354 |
    | 100 MB | 0.753 | 0.868 |
    | 400 MB | 0.919 | 0.918 |
    | "trap" workload | 0.134 | 0.421 |

  - Caveat: the workloads look synthetic, and ARC is nearly identical to the reported LRU numbers (0.1436 at 25 MB). Treat as weak evidence.
  - — [arXiv 2508.12485](https://arxiv.org/html/2508.12485v1)
- **HALP** (Song et al., NSDI '23): a heuristic proposes 4 candidates and a learned pairwise scorer picks the victim. It has run in the YouTube CDN since 2022, with an average 9.1% byte-miss reduction at 1.8% CPU overhead. The paper estimates about 19.2% extra CPU for a naive learned approach. — [HALP, NSDI '23](https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu)

### Inferences
- **[INFERENCE]** For this project, the practical ranking is:
  1. Candidate re-ranking by a supervised scorer (Option B as planned) — per-decision, but bounded to K candidates.
  2. A cache-level controller (LAH-style, or a bandit over ARC/LRU/LFU experts).
  3. Per-object RL over the whole cache (the current DQN). This is the least supported.

  Setting ARC's p with RL fits category 2. It is plausible but unvalidated.
- **[INFERENCE]** Policy-gradient or actor-critic methods add little over simple bandits/Hedge when the action is "which expert or list evicts." The published successes are all regret-minimization or GBDT classification, not deep RL.

### Gaps
- There is no paper comparing a per-object RL cache against a structure-level RL controller on the same traces with the same features.
- There is no published actor-critic or policy-gradient controller for ARC's p.

## Q4. Using ARC's structure inside a learned policy: candidates, ghost labels, features, ARC as the safe fallback

### Takeaway
Each piece has published precedent in a *neighboring* system, but **no paper combines them on ARC**:

- **HALP** explicitly allows "LRU, LFU, or other heuristic policies" as candidate generators.
- **LeCaR/CACHEUS/OLeCaR** show that ghost/history hits are usable delayed training labels.
- **Robust-caching theory** (Wei 2020, Chłędowski et al. ICML 2021) uses LRU or MARKER as the fallback. None uses ARC.
- **Guard** (NeurIPS 2025) is not a fallback combiner at all.

ARC's proven O(N) competitive ratio (4N, preprint result) means an ARC fallback gives the same order of worst-case protection as LRU. This is a sound but unpublished substitution.

### Cited Findings
- HALP's candidate selection: "This heuristic algorithm can be selected as LRU, LFU, or other heuristic policies. We find in practice LRU policy is sufficient." Its rationale is that the heuristic "also provides a lower limit on decision quality." Four candidates is the chosen trade-off. Candidates not chosen are re-inserted at the LRU head. HALP labels comparisons after the fact (post-hoc) from future re-accesses. — [HALP, NSDI '23](https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu)
- Ghost/history lists as delayed labels: in EXP4-DFDC, "feedback on the eviction comes in the form of a 'miss', but at an indeterminate time," and it is ignored beyond a threshold equal to the history size. This is exactly ARC's B1/B2 horizon. — [JMLR / arXiv 2009.11330](https://arxiv.org/abs/2009.11330). LeCaR labels each history entry with the policy that evicted it. — [LeCaR](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf)
- **Wei 2020** ("Better and Simpler Learning-Augmented Online Caching," APPROX 2020): combines BlindOracle with **LRU** and shows this gives the best result among deterministic algorithms. — [arXiv 2005.13716](https://arxiv.org/abs/2005.13716)
- **Chłędowski, Polak, Szabucki, Żołna, ICML 2021**:
  - Main finding: "blindly following either a predictor or a classical robust algorithm, and switching whenever one becomes worse than the other" has low overhead and acts as "cheap worst-case insurance."
  - Fallbacks studied: MARKER or LRU, with deterministic or randomized (Blum–Burch) combiners.
  - On LRU as fallback, footnote 3 says: "combining with LRU does not yield robustness, since LRU is not O(log k)-competitive ... in practice, it makes sense to use LRU as a fallback."
  - ARC is not used.
  - — [PMLR v139](http://proceedings.mlr.press/v139/chledowski21a.html); [arXiv 2106.14693](https://arxiv.org/abs/2106.14693)
- **Guard** (Chen, Zhao, Zhang, Tang, Wang, Deng; NeurIPS 2025; arXiv 2507.16242):
  - Mechanism: phase-based marking, not a black-box fallback. It "guards" pages evicted and then re-missed within a phase, and randomly evicts an unguarded old page.
  - Guarantees: preserves 1-consistency and gives (2H_{k−1}+2)-robustness at O(1) extra per request.
  - It wraps BlindOracle, **LRB (LightGBM Belady labels)** and Parrot, and does not mention ARC.
  - — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/116615); [arXiv 2507.16242](https://arxiv.org/html/2507.16242v1)
- ARC's competitive ratio: upper bound 4N, versus N for LRU (CAR: 18N upper, N+1 lower). The authors state no pathological sequences degrade ARC or CAR "by more than a constant factor as compared to LRU." — [Consuegra et al., arXiv 1503.07624](https://arxiv.org/abs/1503.07624) (preprint)

### Inferences
Integration designs, all **[INFERENCE]** with no paper behind the ARC-specific versions:
- **(a) ARC tails as the candidate generator.** Take the 2 LRU-end entries of T1 plus the 2 LRU-end entries of T2 (4 candidates, matching HALP's K=4). Or take K from whichever list ARC's REPLACE would pick. This gives the scorer both "one-hit" and "frequent" candidates, where an LRU tail gives only recency-ordered ones. Candidates the scorer does not evict go back to the MRU end of their own list, as HALP does for LRU.
- **(b) B1/B2 as a free label source.** A request that hits B1 or B2 within the ghost horizon is a positive "would have been reused" label for the evicted key, with a known reuse gap. A ghost-list expiry is a censored negative. This is essentially the plan's ghost table. ARC's ghosts cover about c entries total, so the label horizon H is roughly one cache-size of evictions.
  - Keep a separate, possibly longer, ghost table with the features captured at eviction time. ARC's own B1/B2 hold only keys.
- **(c) List membership as a feature.** Features would be: in T1 vs T2, position within the list, current p/c, and recent B1-vs-B2 hit rates. These are cheap and ARC-native, and LAH shows hit-position histograms carry about 75% of its feature importance (at cache level).
- **(d) ARC as the shadow/fallback instead of LRU.** The Chłędowski deterministic combiner works with any fallback whose cost can be simulated, and ARC's metadata (2c keys) is only about 2× an LRU shadow. The worst-case bound order stays O(k), like LRU (4N vs N). Empirically ARC is usually ≥ LRU. So an ARC shadow should raise the floor without weakening the theory.
  - Caveat: CACHEUS shows ARC underperforming on scan-then-churn phases.
  - Guard-style O(log k) robustness would need marking, which is independent of the ARC choice.
- **(e) Cheaper variant.** Run the learned scorer *inside* ARC's REPLACE. ARC decides which list loses a slot (keeping p adaptation) and the model picks which of that list's K tail items to evict. This keeps ARC's recency/frequency balancing as a structural prior and gives two levels of safety: an ARC floor on list choice, plus abstention (evict the plain LRU tail of the chosen list).

### Gaps
- No empirical study uses ARC as the robust partner in a learning-augmented combiner.
- No paper uses ARC's T1/T2 tails as ML candidates. HALP only says "other heuristic policies" and uses LRU in practice.
- The 4N ARC bound is from a preprint. I did not verify later peer-reviewed confirmation.

## Q5. Adaptation speed and drift: ARC's p vs online retraining; drift-detector combinations

### Takeaway
ARC adapts on every ghost hit with a step size that grows with ghost-list imbalance, so it reacts within about a cache-size of requests, at zero training cost. But it overreacts or misreacts in some phase patterns:

- LAH describes pathological failure cases.
- CACHEUS shows ARC stuck on T1 eviction after scan→churn.
- LeCaR: its learning rate needed tuning, and different rates were optimal at different times.

**No paper combines ARC with a formal drift detector (ADWIN, etc.).**

### Cited Findings
- ARC's δ "learning rates" scale with |B2|/|B1| (and the inverse), so adaptation speeds up when one ghost list dominates. — [ARC, FAST '03](https://www.usenix.org/legacy/events/fast03/tech/full_papers/megiddo/megiddo.pdf)
- CACHEUS found that not only do workload characteristics change over time, "the velocity and magnitude of these changes also varied significantly over time." This is why it adapts its learning rate online, including a reset after 10 consecutive degrading windows (a crude drift trigger). — [CACHEUS, FAST '21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- CACHEUS Figure 10 plots ARC's normalized p over time on webmail and shows ARC failing to leave its scan setting when churn arrives. — [CACHEUS, FAST '21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- LAH argues that per-miss adaptation (ARC-style) is less important than choosing the right parameter regime, and uses one prediction after 20% of the trace. It reports cross-dataset generalization (CDN-trained model applied to Twitter KV) without retraining. — [arXiv 2608.27975](https://arxiv.org/html/2608.27975) (preprint)
- HALP retrains continuously from post-hoc labels as part of normal cache operation. — [HALP, NSDI '23](https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu)

### Inferences
- **[INFERENCE]** ARC's p and the online model operate on different timescales. p reacts within about c misses. A LightGBM retrain on delayed labels needs at least H requests for labels to resolve, plus a batch. Using ARC as the shadow/list-chooser (Q4 d/e) therefore gives fast structural adaptation while the model lags. ADWIN on the model's windowed miss rate relative to the ARC shadow is a natural switch trigger.
- **[INFERENCE]** ARC's p trajectory is itself a cheap drift signal. A sudden sustained swing in p, or in the B1/B2 hit mix, could trigger model retraining or abstention. This is unpublished.
- **[INFERENCE]** Label-horizon interaction: after a regime shift, ghost-derived labels describe the *old* regime for up to one ghost horizon. A drift detector should discard or down-weight labels created before the change.

### Gaps
- No study measures ARC's recovery time against online-retrained learned caches after controlled workload shifts.
- No ARC + ADWIN (or other formal detector) paper was found.

## Q6. Practical cost: latency and metadata of hybrids vs ARC alone vs HALP

### Takeaway
ARC alone costs about 2c keys of metadata and O(1) work per request. Adding a HALP-style scorer over ARC tails adds per-eviction inference over K≈4 candidates plus a feature-snapshot ghost table. Expert-mixing hybrids (LeCaR/CACHEUS) cost about 2N metadata. H-MC needs two full virtual caches. LAH-style parameter control is the cheapest learned option (no per-object state).

### Cited Findings
- CACHEUS: about 2N metadata, "equivalent to ... ARC and LIRS"; O(log N) per operation because of its LFU expert. — [CACHEUS, FAST '21](https://www.usenix.org/system/files/fast21-rodriguez.pdf)
- LeCaR: 2–3× ARC's metadata. — [LeCaR, HotStorage '18](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf)
- H-MC: simulates both LRU and LFU caches (about double the metadata); no measured costs. — [arXiv 2609.07566](https://arxiv.org/html/2609.07566)
- LAH/S4-FIFO: about 73 KB total, under 2 ms asynchronous inference, O(1) histogram updates per access. Throughput is "comparable to conventional heuristics" with a modest slowdown. LRB needs 200+ bytes per object. — [arXiv 2608.27975](https://arxiv.org/html/2608.27975)
- HALP: 1.8% CPU overhead in production with 4 candidates, vs an estimated about 19.2% for an unconstrained learned design. — [HALP, NSDI '23](https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu)
- Cold-RL: DQN over the K LRU tail, with p95 eviction latency within a 500 µs budget and under 2% CPU. It falls back to LRU on timeout. — [arXiv 2508.12485](https://arxiv.org/html/2508.12485v1) (preprint)
- Guard adds O(1) per request on top of the base learned algorithm. — [arXiv 2507.16242](https://arxiv.org/html/2507.16242v1)

### Inferences
- **[INFERENCE]** In a Python simulator with uniform sizes, the absolute costs matter less than the cost *ratios*. The relevant extra cost of "ARC shadow instead of LRU shadow" is about 2× shadow metadata (T1, T2, B1, B2 vs one list) and a few more branches per request. Scoring 4 ARC-tail candidates costs the same as scoring 4 LRU-tail candidates.
- **[INFERENCE]** The ensemble-for-abstention in Option B multiplies inference cost by the ensemble size. HALP's 1.8% CPU was for a single scorer, and I found no published cost for ensembles in caching.

### Gaps
- There are no published latency or metadata numbers for any hybrid that uses ARC as its structure (candidates, shadow or features).
- There is no measured cost of ensemble-based abstention in a learned cache.
