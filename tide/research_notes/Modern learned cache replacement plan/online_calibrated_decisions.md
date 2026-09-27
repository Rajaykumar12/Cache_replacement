# Online Learning, Calibrated Decisions, and Robust Learning-Augmented Eviction (as of Sept 2026)

Scope: how an eviction policy can (a) keep learning online so hit rate improves while it runs, (b) know when it is uncertain, and (c) fall back safely to a classical policy. The target is a Python/NumPy simulator: a small value-MLP scores the 16 LRU-tail candidates, and today it is trained once offline with a naive DQN TD loss on an LRU trajectory.

Source-verification legend. "(verified)" means I fetched the primary page or PDF in this session. "(background)" means a well-known result cited from my prior knowledge with its canonical link, which I did not re-fetch here. The report writer should treat (background) numbers as needing a spot check before they are quoted exactly. "(preprint)" marks an arXiv paper that has not been peer reviewed.

---

## Q1. Online / adaptive cache policies: LeCaR, CACHEUS, ARC, bandits, LRB / HALP-style continual training. How quickly do they adapt?

### Takeaway
Two proven families adapt online. The first is **expert mixing with ghost-list regret** (LeCaR/CACHEUS): it is cheap, has no model, and adapts on every miss. The second is **continual supervised retraining on delayed, self-generated labels** (LRB: a GBM retrained every 128K labels; HALP: a 2-layer MLP trained fully online from pairwise "who was re-referenced first" labels). For this project the highest-leverage move is to keep the MLP and swap the offline DQN for HALP/LRB-style online supervised updates, because future-reuse labels come for free from the request stream. Then wrap it in a LeCaR/CACHEUS-style mixer with LRU so the system can never drift far below LRU.

### Cited Findings
**LeCaR (Vietri et al., HotStorage 2018)**
- Each eviction picks an expert (LRU or LFU) at random, with probability proportional to its weight. When a missed item is found in an expert's eviction-history list, LeCaR penalizes that expert, because its eviction caused the miss. It has two hand-set parameters, a learning rate and a discount rate. (verified) — [Rodriguez et al., "Learning Cache Replacement with CACHEUS", USENIX FAST 2021, §3.1](https://www.usenix.org/system/files/fast21-rodriguez.pdf); original: [Vietri et al., "Driving Cache Replacement with ML-based LeCaR", USENIX HotStorage 2018](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf)
- LeCaR "was the first to demonstrate an ability to adapt its behavior based on the available cache size, independent of the ability to adapt to the dynamics of the workload". It has "drawbacks relating to adaptiveness, overhead, and churn-friendliness". (verified) — [CACHEUS, FAST 2021, §2.2](https://www.usenix.org/system/files/fast21-rodriguez.pdf)

**CACHEUS (Rodriguez et al., FAST 2021)**, taken from Algorithms 1–3 of the paper (verified — [PDF](https://www.usenix.org/system/files/fast21-rodriguez.pdf), [code](https://github.com/sylab/cacheus)):
- State: two experts A and B, each with an eviction-history LRU list H_A and H_B of size **N/2** (N = cache size), plus weights w_A = w_B = 0.5 at the start.
- On a miss for q: if q ∈ H_A then `w_A *= exp(-λ)`, else if q ∈ H_B then `w_B *= exp(-λ)`. Then renormalize. If both experts would evict the same item, evict it and skip the history update. Otherwise sample an action with probability (w_A, w_B), evict that expert's victim, and push the victim to that expert's history.
- The **discount rate was removed**: "eliminating the discount rate altogether did not affect LeCaR's performance appreciably".
- **Adaptive learning rate**: at the end of every window of N requests (N = cache size), the method computes the hit-rate change δHR and the λ change δλ over the previous two windows. If δHR/δλ > 0 it keeps moving λ in the same direction, otherwise it reverses: `λ = max(λ + sign·|λ·δλ|, 1e-3)`. After **10 consecutive windows** of degrading or zero hit rate, it resets λ to a uniform random value in [1e-3, 1]. This is gradient-based stochastic hill climbing with random restart.
- Motivation: "different static values of the learning rate were found to be optimal for different workloads … the velocity and magnitude of these changes varied significantly over time".
- Experts are SR-LRU (scan-resistant LRU) and CR-LFU (churn-resistant LFU). The authors found more than two experts "significantly worse", because overlapping experts corrupt the learning signal. Evaluation covered 329 traces from 5 production collections, with cache sizes from 0.05% to 10% of the footprint. Figure 3 shows that no prior algorithm (ARC, LIRS, DLIRS, LeCaR) wins everywhere.
- **Adaptation speed, in concrete terms**: weights react on every miss that hits a history list, by a factor of e^{-λ}. The learning rate itself reacts once per N requests. Regret is only detectable for items still in a history of size N/2. Reuse beyond roughly one cache-length of misses is therefore invisible to the mixer.

**Theory caveat (Sept 2026, preprint)**
- Ben Mazziane & Zou show that LeCaR "suffers linear regret against an oblivious adversary, even with unbounded history". They propose **H-MC**, a Hedge-based mixture of *virtual* LRU and LFU caches. It keeps Hedge's selection-probability guarantees and achieves optimal switching cost. (verified abstract, preprint) — [arXiv:2609.07566, Sept 2026](https://arxiv.org/abs/2609.07566)
- An earlier preprint frames cache replacement as a multi-armed bandit with delayed feedback and decaying costs. (verified title only, preprint) — [arXiv:2009.11330](https://arxiv.org/pdf/2009.11330)

**ARC (Megiddo & Modha, FAST 2003)** (background) — [USENIX FAST 2003](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache)
- ARC keeps two LRU lists, T1 (seen once) and T2 (seen at least twice), and two ghost lists B1 and B2 of evicted keys. A hit in B1 grows the T1 target size p, and a hit in B2 shrinks it. This is the same "ghost hit = regret signal" idea, applied to a single continuous parameter. CACHEUS notes that ARC has trouble with churn and with LFU-friendly workloads. (verified) — [CACHEUS §2.2](https://www.usenix.org/system/files/fast21-rodriguez.pdf)

**LRB (Song, Berger, Li, Lloyd, NSDI 2020)**, continual retraining (verified, [PDF §4](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)):
- Keeps features only for objects whose last request falls within a **sliding memory window**. The window size is a hyperparameter tuned on a validation prefix of each trace. For large caches it is extrapolated with a least-squares fit.
- **Label generation is decoupled from features**. LRB periodically samples random objects from the memory window, sampling over objects rather than requests to avoid popularity bias. It snapshots their features and labels each one either (1) when the object is requested again, or (2) when it falls out of the window, with label = 2× the window size. This second rule relies on the "Belady boundary": every object not reused within the boundary is an equally good eviction.
- Target: **log(time-to-next-request)** regression with L2 loss, using LightGBM GBM. GBM beat logistic regression, linear regression, SVM and a 2-layer/16-unit NN on the "good decision ratio". Training takes about 300 ms. Prediction on 64 candidates takes about 30 µs.
- Retrains **each time 128K labeled samples accumulate**, then discards the dataset and swaps in the new model. Larger datasets showed diminishing returns.
- Picks **k = 64 random eviction candidates** and evicts the one with the farthest predicted next request.
- **Cold start: "our implementation uses LRU as a fallback until sufficient training data is available."**

**HALP (Song et al., NSDI 2023; YouTube CDN)**:
- A heuristic (for example LRU) proposes a small candidate set. A **two-layer MLP reward model** reranks it, and the eviction is the best-of-n. Randomization "approximates exact priority queues". (verified) — [Google Research blog, 2023](https://research.google/blog/preference-learning-with-automated-feedback-for-cache-eviction/); [NSDI'23 paper](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf)
- **Automated pairwise preference feedback**: at each eviction it creates pairwise queries between candidates and puts them in a pending buffer. When one item of a pair is re-accessed first, the pair is labeled, and it is labeled "incrementally after the current request". The MLP trains "fully online starting from a random weight initialization" from a transient buffer, which gives per-server specialization. Features include time since last access and average time between accesses. (verified, blog)
- Production: running as the YouTube CDN DRAM eviction policy since early 2022. It reduced peak byte miss by an average of 9.1% at 1.8% CPU overhead. (verified) — [USENIX NSDI'23 page](https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu). The blog also reports a 6% memory hit-rate improvement and a 12% increase in the egress/ingress ratio. (verified, blog)

### Inferences
- **HALP is almost exactly the project's architecture**: an LRU tail of candidates plus a tiny MLP. The difference is the training signal. HALP learns online from *which candidate is re-referenced first*, which is a supervised, action-independent label. The current setup learns from a DQN TD loss on an LRU trajectory. Switching to HALP-style pairwise labels, or to LRB-style reuse-time labels, is the single most direct way to "improve live". Implementation in NumPy: a pending dict `{key: (feature_snapshot, t_snapshot)}`, resolved on the next access or on timeout, feeding minibatch SGD every M requests.
- Why the DQN is the wrong tool: in a trace-driven simulator, the future request stream does not depend on the eviction decision. The ground-truth "next reuse time" of every candidate, including evicted ones, is therefore observable after a delay. That makes this a **full-information, delayed-label supervised problem**, not a bandit or RL problem. A TD loss on an LRU trajectory also bootstraps values *for LRU's* behavior, not for optimal behavior.
- A robust shell: run CACHEUS(A = LRU, B = MLP-policy). Keep history lists of the victims each expert chose, apply the exp(-λ) penalty on history hits, and use CACHEUS's λ hill-climbing. If the MLP is bad, w_MLP collapses and the cache behaves like LRU within a few windows. Caveat: this inherits LeCaR's lack of worst-case regret guarantees (arXiv:2609.07566). H-MC or the switching combiners in Q2 are the principled alternatives.
- Expected adaptation latency for the project: an online MLP updated every ~1K requests from labels with horizon H will lag a workload shift by about H + 1K requests. A CACHEUS mixer reacts within about one cache-size window of requests.

### Gaps
- I did not extract quantitative "time to recover after a shift" curves for CACHEUS, LRB or HALP. None of the fetched text gives a number in requests.
- HALP's exact candidate count, training cadence and learning rate are in the NSDI PDF, but the blog did not state them.
- The H-MC preprint's experimental numbers were not available in the abstract.

---

## Q2. Learning-augmented (algorithms-with-predictions) caching: consistency/robustness and how to combine a predictor with a robust fallback

### Takeaway
Theory gives a clear recipe: **follow the predictor, but keep a classical robust algorithm (Marker or LRU) running in shadow, and switch to whichever is currently doing better.** This bounds the damage to O(log k) × OPT, or k × OPT with LRU, when predictions are garbage. It stays near 1-consistent when predictions are good. Chledowski et al. showed empirically that this simple switch costs little on real traces with a good predictor. It is also the cheapest thing to implement.

### Cited Findings
- **Lykouris & Vassilvitskii**, "Competitive Caching with Machine Learned Advice" (ICML 2018; journal version JACM 68(4), 2021). *Predictive Marker* is a Marker algorithm that, within a phase, evicts the unmarked page with the farthest *predicted* next arrival. Where the prediction contradicts the phase structure it falls back to random-marker eviction along "clean chains". Its competitive ratio degrades gracefully with prediction error and "is capped at O(log k)" when predictions are useless. The framework treats the oracle as a black box. (verified abstract) — [arXiv:1802.05399](https://arxiv.org/abs/1802.05399); JACM version: [doi:10.1145/3447579](https://doi.org/10.1145/3447579). The exact bound O(1 + min(√(η/OPT), log k)), with η = ℓ1 error of predicted next-arrival times, is (background).
- **Rohatgi**, "Near-Optimal Bounds for Online Caching with Machine Learned Advice" (SODA 2020). Improves the dependence on η to roughly O(1 + min(log k, (log k / k)·η/OPT)) and gives a near-matching lower bound. (background; exact form to verify) — [SIAM SODA 2020](https://epubs.siam.org/doi/10.1137/1.9781611975994.110)
- **Wei**, "Better and Simpler Learning-Augmented Online Caching" (APPROX/RANDOM 2020). *BlindOracle* evicts the page with the farthest predicted next arrival, with no marking. It achieves competitive ratio **min(1 + 2η/OPT, 2 + 4/(k−1)·η/OPT)**. Combining BlindOracle with a k-competitive deterministic algorithm such as **LRU**, through a combiner, is "the best one could hope to do among deterministic algorithms". (verified via search snippet of paper) — [arXiv:2005.13716](https://arxiv.org/pdf/2005.13716)
- **Antoniadis, Coester, Eliáš, Polak, Simon**, "Online Metric Algorithms with Untrusted Predictions" (ICML 2020; TALG 2023). Predictions are of the *cache state* rather than next-arrival times. "Follow the Prediction" (FtP) is combined with a robust algorithm through a deterministic or randomized combiner (background) — [arXiv:1904.03936](https://arxiv.org/abs/1904.03936)
- **Chledowski, Polak, Szabucki, Zolna**, "Robust Learning-Augmented Caching: An Experimental Study" (ICML 2021). This is the first comprehensive evaluation on real caching traces with state-of-the-art learned predictors. "Blindly following either a predictor or a classical robust algorithm, and switching whenever one becomes worse than the other — has only a low overhead over a well-performing predictor, while competing with classical methods when the coupled predictor fails, thus providing a cheap worst-case insurance." (verified) — [arXiv:2106.14693](https://arxiv.org/abs/2106.14693), [PMLR v139](http://proceedings.mlr.press/v139/chledowski21a.html), [datasets](https://github.com/chledowski/Robust-Learning-Augmented-Caching-An-Experimental-Study-Datasets)
- **Chen, Zhao, Zhang, Tang, Wang, Deng**, "Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency" (**NeurIPS 2025**). The *Guard* framework raises robustness of a broad class of learning-augmented caching algorithms to **2H_{k−1} + 2** while *preserving 1-consistency*, with **O(1) extra per-request overhead**. The authors describe this as the best-known consistency/robustness trade-off. (verified) — [arXiv:2507.16242](https://arxiv.org/abs/2507.16242)
- Other lines, verified by title or search hit only:
  - Parsimonious learning-augmented caching (few predictor queries): [Im, Kumar, Petety, Purohit, ICML 2022, arXiv:2202.04262](https://arxiv.org/pdf/2202.04262)
  - New upper bounds: [arXiv:2410.01760 (preprint)](https://arxiv.org/abs/2410.01760)
  - A 2026 survey preprint, "Learning-Augmented Algorithms: Guarantees, Construction Mechanisms, and System-Level Implications": [arXiv:2609.04787 (preprint)](https://arxiv.org/pdf/2609.04787)
  - The community paper list: [algorithms-with-predictions.github.io](https://algorithms-with-predictions.github.io/)
- **Paging with succinct predictions** (Antoniadis, Boyar, Eliáš, Favrholdt, Hoeksma, Larsen, Polak, Simon, ICML 2023). Uses **one bit per request**, for example "will this page be evicted by Belady?" or "requested again soon?", and gives consistency/robustness bounds for such binary predictions. (background) — [arXiv:2210.02775](https://arxiv.org/abs/2210.02775). This fits a calibrated P(reuse within horizon) classifier directly.

### Inferences
- A concrete, implementable robust wrapper in the style of Chledowski/Wei:
  1. Simulate a **shadow LRU cache**, which holds only keys, alongside the real ML-driven cache on the same request stream.
  2. Keep miss counters for both over a sliding window, or cumulatively with a margin.
  3. When the ML policy's misses exceed the shadow's by more than a threshold, switch the real cache to following LRU. Switch back when the gap reverses.
  4. Reconciling cache contents after a switch costs at most k extra misses, and that cost is what the theory accounts for.
  
  This is about 40 lines of Python and gives a real, explainable "never much worse than LRU" guarantee.
- The project's "bottom-16 LRU candidates" restriction is itself a mild robustness device. HALP uses the same pattern. Because the model can only choose among old items, the worst case is bounded by a scan-like failure mode, not arbitrary.
- Predictive Marker, BlindOracle and Guard all need a **next-arrival-time** prediction or an eviction bit. The MLP should therefore output an interpretable quantity (log reuse time, or P(reuse ≤ H)) rather than a Q-value. That is also what makes calibration possible (Q3).

### Gaps
- I did not fetch the exact theorem statements for Lykouris–Vassilvitskii (JACM) or Rohatgi. The bounds above are marked (background).
- Guard's empirical overhead and hit-rate numbers are not in the abstract.

---

## Q3. Calibration and uncertainty: conformal prediction, temperature/isotonic calibration, ensembles/MC dropout, abstention, and "RL for calibrated decisions" (RLCR). Does it transfer to eviction?

### Takeaway
RLCR's core insight is that **rewarding with a bounded proper scoring rule (Brier) makes the model's confidence calibrated without hurting accuracy**. This transfers to eviction, but the transfer is simpler than "use RL". In eviction the outcome (was the item reused within H requests?) is observed for every candidate after a delay, whatever action was taken. You can therefore minimize the Brier score or log loss *directly* as a supervised loss. That is exactly the objective RLCR's reward induces, and it avoids RL variance. Then evict by **expected cost**, arg-min p̂_i·cost_i. Abstain to LRU when the ensemble disagrees or the top candidates' calibrated probabilities are within the noise band. Online conformal methods (ACI) can keep an abstention threshold valid under drift.

### Cited Findings
**RLCR (Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas)**, "Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty", arXiv:2507.16806. Submitted July 2025, revised May 2026. **Preprint.** (verified) — [arXiv:2507.16806](https://arxiv.org/abs/2507.16806)
- The reward combines binary correctness with a **Brier score** on the model's verbalized confidence q. The usual form is R = 1[correct] − (q − 1[correct])².
- The authors prove that "this reward function (or any reward function that uses a bounded, proper scoring rule) yields models whose predictions are both accurate and well-calibrated."
- "RLCR substantially improves calibration with no loss in accuracy, on both in-domain and out-of-domain evaluations". Ordinary RL *degrades* calibration. The method outperforms post-hoc confidence classifiers and enables confidence-weighted test-time scaling.

**Algorithms with calibrated predictions (Shen, Vitercik, Wikum, ICML 2025, PMLR 267:54476–54498)**. Proposes **calibration** as the bridge between per-prediction uncertainty and online algorithms with predictions, replacing a single user-specified global trust parameter. Case studies are ski rental and online job scheduling. For ski rental, the algorithm "achieves near-optimal prediction-dependent performance". In high-variance settings, "calibrated advice offers more effective guidance than alternative methods for uncertainty quantification" (conformal intervals). (verified) — [PMLR](https://proceedings.mlr.press/v267/shen25f.html), [arXiv:2502.02861](https://arxiv.org/abs/2502.02861)

**Online Algorithms with Uncertainty-Quantified Predictions (Sun, Huang, Christianson, Hajiesmaili, Wierman, Boutaba)**. Predictions come with UQ, meaning the likelihood that ground truth lies in a range, as conformal gives. Standard algorithms must be modified to exploit UQ. The paper also offers an online-learning framework that learns *how* to use UQ across repeated instances. Problems studied are ski rental and online search, not caching. (verified; the arXiv page lists no venue, and I believe it appeared at ICML 2024 (background)) — [arXiv:2310.11558](https://arxiv.org/abs/2310.11558)

**Online conformal under drift, ACI (Gibbs & Candès, NeurIPS 2021)**:
- Achieves long-run target coverage "irrespective of the true data generating process" (no exchangeability). (verified) — [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html), [arXiv:2106.00170](https://arxiv.org/abs/2106.00170)
- Update rule: α_{t+1} = α_t + γ(α − err_t), where err_t = 1 if the truth fell outside the set. This is online subgradient descent on the pinball loss. (background; a search snippet shows the equivalent θ-parameterization)
- A Sept 2026 preprint studies ACI with **delayed feedback**, which matches caching's delayed reuse labels. (title only, preprint) — [arXiv:2609.07251](https://arxiv.org/pdf/2609.07251)
- Strongly adaptive online conformal variant: [Bhatnagar et al., arXiv:2302.07869](https://arxiv.org/abs/2302.07869)

**Standard calibration / uncertainty tools** (all background):
- Temperature scaling, a single scalar T fit on held-out logits: [Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks", ICML 2017](https://arxiv.org/abs/1706.04599)
- Isotonic regression calibration: [Zadrozny & Elkan, KDD 2002](https://dl.acm.org/doi/10.1145/775047.775151)
- Deep ensembles: [Lakshminarayanan, Pritzel, Blundell, NeurIPS 2017](https://arxiv.org/abs/1612.01474)
- MC dropout: [Gal & Ghahramani, ICML 2016](https://arxiv.org/abs/1506.02142)
- Selective classification with a guaranteed-risk abstention threshold: [Geifman & El-Yaniv, NeurIPS 2017](https://arxiv.org/abs/1705.08500)
- Brier score: [Brier, Monthly Weather Review 1950](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)

**LRB already uses its regression output "also as a confidence measure for distance to the Belady boundary"**. Mispredicting 100K versus 2M matters more than 2M versus 3M, hence the log target. (verified) — [LRB NSDI'20 §4.3.3](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)

### Inferences
How RLCR's idea transfers to eviction, as a concrete design for the NumPy simulator:
1. **Target.** For candidate i at time t, let y_i = 1 if key i is requested within the next H requests, with H ≈ cache size or tuned like LRB's memory window. Snapshot the features when the key is a candidate. Resolve y when the key is next requested (y = 1 if within H), or after H requests (y = 0). Evicted keys need a ghost/metadata table so their labels still resolve. This mirrors LRB's labeling and HALP's pending buffer.
2. **Loss.** The MLP outputs p̂_i = σ(f(x_i)). Train online by minibatch SGD on **log loss or Brier**, (p̂ − y)². Both are strictly proper, so the minimizer is the true conditional probability. That is RLCR's calibration argument without RL. RL-with-Brier-reward would only be needed if the label depended on the action, and in trace-driven caching it does not. Report reliability diagrams and ECE on resolved labels as the calibration metric.
3. **Decision.** Evict arg-min_i p̂_i · c_i, the expected miss cost. Use c_i = 1 for the hit rate, c_i = size_i or fetch cost for the byte-hit rate, or evict arg-min p̂_i/size_i. This decision is only meaningful if p̂ is calibrated. With c_i = 1 the ranking is invariant to monotone recalibration, but calibration still matters for (a) size- and cost-aware eviction, (b) the abstention threshold, and (c) admission ("don't cache if p̂ < τ").
4. **Uncertainty.**
   - **Epistemic**: an ensemble of 3–5 tiny MLPs with different seeds or bootstrap minibatches. This is cheap in NumPy, where it amounts to stacking weight matrices. Use σ_i = std across members.
   - **Aleatoric**: comes from p̂ itself.
   - **Abstain rule**: fall back to the LRU victim, the tail of the list, when (i) max_i σ_i > τ_σ, or (ii) the calibrated margin between the chosen victim and the LRU victim is smaller than the ensemble's noise, meaning the model is not confidently better than LRU. Pick τ with a selective-risk rule on recent resolved decisions, or adapt it online with ACI: raise τ when abstained-from decisions turned out wrong at more than rate α.
5. **Recalibration.** Keep an isotonic regression (`sklearn.isotonic.IsotonicRegression`, if scikit-learn is allowed) or a single temperature T. Refit on the last N resolved (p̂, y) pairs. Temperature scaling is about 5 lines of NumPy: 1-D Newton or grid search on log loss.
6. **Messaging for judges.** "RL for calibrated decisions" can be honestly described as "trained with a proper-scoring-rule (Brier) objective, following RLCR's result that proper-scoring-rule rewards yield calibrated confidence, plus expected-cost decisions and abstention to LRU". Claiming it is RL would be misleading once the loss is supervised.

### Gaps
- I found **no paper applying conformal prediction or RLCR-style calibration rewards specifically to cache eviction**. The closest are Shen et al. (calibrated predictions for ski rental and scheduling), Sun et al. (UQ for ski rental and search), and the binary-prediction paging line (Antoniadis et al. 2023). The transfer above is my inference.
- I did not confirm RLCR's exact reward formula from the paper body. The abstract confirms "binary correctness + Brier".

---

## Q4. Drift detection (ADWIN, Page-Hinkley) and retraining triggers

### Takeaway
Feed a streaming change detector a per-request **loss signal**: the ML policy's miss minus the shadow LRU's miss, or the Brier loss of resolved predictions. On an alarm, (a) route decisions to LRU, (b) raise the learning rate or reset the training buffer to recent data, (c) resume the model once its shadow performance recovers. The `river` library provides ADWIN and PageHinkley with a two-line API. Both are also easy to hand-roll.

### Cited Findings
- `river.drift.ADWIN` implements ADWIN2. `update(value)` adds a value and updates the window statistics. The `drift_detected` property flags change after the last update. `river.drift.PageHinkley` implements a CUSUM-type, two-sided Page-Hinkley test on the mean. (verified) — [River ADWIN docs](https://riverml.xyz/dev/api/drift/ADWIN/), [River PageHinkley docs](https://riverml.xyz/dev/api/drift/PageHinkley/), [River concept-drift example](https://riverml.xyz/0.8.0/examples/concept-drift-detection/)
- ADWIN keeps a variable-length window. It drops the older sub-window whenever two sub-windows' means differ by more than a Hoeffding-style bound parameterized by a confidence δ, which gives rigorous false-positive and false-negative bounds. (background) — [Bifet & Gavaldà, "Learning from Time-Changing Data with Adaptive Windowing", SDM 2007](https://doi.org/10.1137/1.9781611972771.42)
- Page-Hinkley test origin: [Page, "Continuous Inspection Schemes", Biometrika 1954](https://doi.org/10.1093/biomet/41.1-2.100) (background)
- CACHEUS has its own built-in drift response: it resets λ to a random value after 10 consecutive degrading windows. (verified) — [CACHEUS Alg. 3](https://www.usenix.org/system/files/fast21-rodriguez.pdf). LRB's implicit drift response is periodic retraining on the most recent 128K labels. (verified) — [LRB §4.3.5](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)

### Inferences
Suggested simulator wiring:
```
d = drift.ADWIN(delta=0.002)
d.update(ml_miss - lru_shadow_miss)
if d.drift_detected:
    fallback = True
    lr *= 4
    buffer.clear_older_than(t - W)
```
Clear fallback when the model's shadow miss rate over the last W requests is at or below LRU's.

A second detector on the Brier loss of resolved predictions catches *calibration* drift before hit-rate drift. Evaluate it with synthetic phase-change traces, for example Zipf α 0.8 → 1.2, working-set swap, or scan injection. Measure detection delay in requests.

### Gaps
- I found no caching-specific evaluation of ADWIN or Page-Hinkley for triggering eviction-model retraining. The thresholds above would need tuning in the simulator.

---

## Q5. Off-policy evaluation / safe policy improvement: shadow caches, ghost caches, counterfactual evaluation

### Takeaway
Caching is unusually friendly to counterfactual evaluation. The request trace is exogenous, so any candidate policy can be **simulated exactly in a shadow cache** on the live stream. Only keys are needed; values are not. Deploy a new model only after its shadow beats the incumbent over a window: a "champion/challenger" gate. Miniature spatially-sampled caches (SHARDS) make this cheap in production. Ghost lists (ARC, LeCaR/CACHEUS) are the lightweight version that attributes regret to specific decisions.

### Cited Findings
- **Miniature simulation / SHARDS** (Waldspurger, Saemundsson, Ahmad, Park, USENIX ATC 2017). Emulates a given cache size by running a *miniature cache on a small spatially-hashed sample* of requests. It is the first detailed study showing this works for non-LRU policies (ARC, LIRS, OPT) and is used to optimize cache parameters online. (verified) — [USENIX ATC'17](https://www.usenix.org/conference/atc17/technical-sessions/presentation/waldspurger), [PDF](https://www.usenix.org/system/files/conference/atc17/atc17-waldspurger.pdf)
- **Ghost/history lists as counterfactual attribution.** CACHEUS/LeCaR keep per-expert histories of the victims each expert evicted. A later miss on a key in H_A is direct evidence that expert A's decision cost a miss. (verified) — [CACHEUS Alg. 1–2](https://www.usenix.org/system/files/fast21-rodriguez.pdf). ARC's B1/B2 ghost lists play the same role. (background) — [Megiddo & Modha, FAST 2003](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache)
- **Switching on shadow performance** is exactly the robust combiner that Chledowski et al. found to be cheap insurance. (verified) — [arXiv:2106.14693](https://arxiv.org/abs/2106.14693)
- HALP lists "measuring impact under production noise" as one of three deployment challenges. (verified) — [NSDI'23 page](https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu). The blog did not describe the A/B methodology.
- **Belady/OPT as an offline counterfactual oracle.** Hawkeye's OPTgen reconstructs what OPT would have done on past accesses in order to label training data. (background) — [Jain & Lin, "Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement", ISCA 2016](https://doi.org/10.1109/ISCA.2016.17). LRB's "relaxed Belady" boundary labeling is the online analogue. (verified) — [LRB §4.2](https://www.usenix.org/system/files/nsdi20-paper-song.pdf). Imitation of Belady for an offline warm start is described in [Liu et al., "An Imitation Learning Approach for Cache Replacement" (Parrot), ICML 2020](https://arxiv.org/abs/2006.16239) (background).

### Inferences
**Champion/challenger in the simulator:**
- Maintain `live_policy` (champion) and `challenger_model`. The challenger is the online-trained MLP snapshot from the last K updates.
- Run it in a key-only shadow cache of the same capacity, or a 1–10% SHARDS-sampled miniature scaled down, on the live stream.
- Promote when the challenger's windowed hit rate exceeds the champion's by more than 2σ. Estimate σ with a paired bootstrap over request blocks. Demote on the reverse, or on an ADWIN alarm.
- This is **exact** off-policy evaluation, with no importance weighting, because actions do not affect future requests. Inverse-propensity or doubly-robust OPE from RL is **unnecessary for eviction-only decisions**. It would matter only if the policy also influenced the request stream, for example through prefetching or latency feedback on user behavior. That point is my inference.
- Cost: one extra dict + OrderedDict per shadow policy, O(1) per request for LRU shadows. A shadow MLP policy costs one extra forward pass per miss.

**Offline safety pre-check.** Before going live, compare the policy on held-out traces against LRU, ARC and Belady (OPT hit rate). Report the "fraction of OPT gap closed", not raw hit rate.

**Minimal implementation stack, ordered by effort:**
1. Shadow LRU plus switch-on-worse (Chledowski/Wei robustness).
2. Replace the DQN with online supervised P(reuse ≤ H) using Brier or log loss and LRB/HALP delayed labels.
3. Expected-cost eviction.
4. A 3–5 member ensemble with abstention to LRU.
5. ADWIN on the (ML − LRU) miss difference for drift.
6. A CACHEUS mixer or champion/challenger gate.
7. Optional: an ACI-tuned abstention threshold, and Belady-imitation warm start (Parrot-style) instead of the LRU-trajectory DQN.

Production-grade equivalents: HALP (YouTube), LRB (Apache Traffic Server prototype), SHARDS miniature caches, and the Guard wrapper (NeurIPS 2025).

### Gaps
- I found no published paper that formally frames cache-policy deployment as "safe policy improvement" with statistical guarantees. The champion/challenger gate above is a standard engineering practice, not a cited method.
- HALP's production A/B/noise methodology is in the NSDI paper but was not in the fetched blog.
