# Deployment and Training of Learned Cache Eviction Policies (as of Sept 2026)

Scope: (a) how production-grade learned caches keep eviction fast, and (b) how they build training data and retrain. The notes are written for a Python KV-cache simulator that currently runs a 64-64 NumPy MLP over the 16 LRU-tail candidates on every miss, trained with offline DQN on about 20k synthetic Zipf requests.

Primary sources I read in full text: LRB (NSDI'20), HALP (NSDI'23), Parrot (ICML'20), S3-FIFO (SOSP'23). The rest (GL-Cache, 3L-Cache, SIEVE, Cold-RL, libCacheSim, lleaves, river) come from abstract pages, READMEs or HTML versions, and are flagged where relevant.

---

## Latency techniques: how learned caches keep eviction fast, and what overheads are reported

### Takeaway
Every deployed learned cache does three things. It runs the model only on eviction (a miss that needs space), over a small candidate set (LRB: 64 random samples; HALP: 4 LRU-tail candidates). It uses a tiny model (GBDT, or an MLP with one hidden layer). It runs that model in compiled code (C++/XLA), not an interpreter. The reported costs are about 30 µs per eviction for 64 LightGBM predictions (LRB) and 720 ns per pairwise comparison (HALP). The newer designs (GL-Cache, 3L-Cache) mainly attack the remaining CPU overhead by learning less often or scoring groups instead of objects.

### Cited Findings

**LRB (Learning Relaxed Belady), NSDI 2020, peer-reviewed**
- On each eviction, LRB randomly samples k = 64 cached objects as eviction candidates, runs one batch GBM prediction over them, and evicts the candidate with the farthest predicted next request. The authors picked 64 with their "good decision ratio" metric: it is "past the point of diminishing returns", and "prediction on 64 samples takes LRB only 30 µs." — [Song et al., LRB, NSDI'20 (PDF)](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)
- They chose GBM over linear regression, logistic regression, SVM, and "a shallow neural network with 2 layers and 16 hidden nodes". Reasons given: GBM had the best good decision ratio, needs no feature normalization, handles missing values natively (objects have varying numbers of deltas), and is fast. "We can train our model in 300 ms. And, we can run prediction on 64 eviction candidates in 30 µs." — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)
- Implementation: about 1,400 lines of C++, shared between the simulator and the Apache Traffic Server prototype. ATS was changed "to make eviction decisions asynchronously by scheduling cache admissions in a lock-free queue." — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)
- Feature-update tricks: deltas are defined between consecutive requests, so only delta1 changes over time and the rest are time-invariant (a new request just shifts delta_n to delta_n+1). EDC decay uses "a lookup table with precomputed decay rates" (about 1 MB for a 256M-request window). Features are updated "only at the times when it is requested and when it is sampled as an eviction candidate." Features are compressed by treating one-hit wonders separately. — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)
- Measured prototype overhead (Wikipedia trace, the pessimistic case): throughput 11.66 Gbps for ATS vs 11.72 Gbps for LRB, i.e. no measurable throughput loss. Peak CPU went from 9% (ATS) to 16% (LRB), with 12% for B-LRU. Peak memory was 39 GB vs 36 GB. P90 latency fell from 110 ms to 72 ms and object misses from 5.7% to 2.6%. Metadata stays under 3% of cache size. — [LRB PDF, Table 5](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)
- Artifact/simulator repo (the HALP paper cites it as its baseline): [github.com/sunnyszy/lrb](https://github.com/sunnyszy/lrb)

**HALP (Heuristic Aided Learned Preference), YouTube CDN, NSDI 2023, peer-reviewed**
- HALP's authors judge LRB as run at their scale: "for each eviction it needs to run predictions for 64 objects, which makes deployment cost-prohibitive (≈ 19.2% additional CPU overhead)". — [Song et al., HALP, NSDI'23 (PDF)](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf)
- Design: a heuristic (LRU) proposes candidates, and a learned scorer picks among them. "Selecting four candidates achieves a good balance." The final victim comes from 3 pairwise comparisons in tournament style, and deselected candidates are re-inserted into the heuristic, which "provides a lower limit on decision quality." — [HALP PDF](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf)
- Candidate-count ablation: going from 2 to 4 candidates cut P95 normalized byte miss ratio from 60.4% to 59.3%. Going from 4 to 8 raises comparisons from 3 to 7 for a gain under 1% relative, so they kept 4. — [HALP PDF §5](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf)
- Model and serving: "a simple neural network model with one hidden layer" (more layers did not help), binary cross-entropy on "which of a pair gets re-accessed first". "Pairwise prediction in 720 ns, and each training in several ms." It is written in XLA and hand-tuned C++ with per-CPU user-space spinlocks and RCU, on top of Google's SmartChoices. — [HALP PDF](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf)
- Production result: deployed as YouTube's DRAM-level eviction since early 2022, it "reduced the byte miss during peak by an average of 9.1% while expending a modest CPU overhead of 1.8%." — [HALP PDF](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf)

**GL-Cache (group-level learning), FAST 2023, peer-reviewed**
- It clusters similar objects into groups and learns and evicts at group level. Reported gains: "improves throughput by 228× and hit ratio by 7% on average" vs LRB, and "64% higher throughput" than the best other learned design, across 118 production block I/O and CDN traces. — [Yang et al., GL-Cache, FAST'23](https://www.usenix.org/conference/fast23/presentation/yang-juncheng)

**3L-Cache, FAST 2025, peer-reviewed**
- Cuts overhead with (1) a training-data collection scheme that "filters out unnecessary historical cache requests and dynamically adjusts the training frequency", and (2) "a low-overhead eviction method that integrates a bidirectional sampling policy". CPU overhead is 60.9% lower than HALP and 94.9% lower than LRB, but still 6.4× LRU for small caches and 3.4× for large ones. Across 4,855 traces it had the best byte or object miss ratio among 12 policies. — [Zhou et al., 3L-Cache, FAST'25](https://www.usenix.org/conference/fast25/presentation/zhou-wenbin). It is available in libCacheSim behind a build flag. — [libCacheSim README](https://github.com/1a1a11a/libCacheSim)

**Glider, MICRO 2019 (hardware CPU caches), peer-reviewed**
- Method: train an offline attention-LSTM on Belady labels, analyze it, then replace it with a simple online model. The analysis found an unordered set of recent PCs is enough, so the online model is an Integer SVM (per-PC table of 16 weights, 4-bit hashes). Single-core miss-rate reduction over LRU: 8.9%, vs 7.1% (Hawkeye), 6.5% (MPPPB), 7.5% (SHiP++). — [Shi et al., Glider, MICRO'19 (PDF)](https://www.cs.utexas.edu/~akanksha/micro19glider.pdf). These numbers were taken from a search snippet of the paper; the PDF fetch failed.

**Heuristic baselines to beat, in throughput and in miss ratio**
- S3-FIFO: three static FIFO queues (small queue at 10% of space, main at 90%, ghost queue with as many entries as main). Evaluated on 6,594 traces from 14 datasets, it has the lowest mean miss ratio on 10 of 14 datasets and "6× higher throughput compared to optimized LRU at 16 threads." — [Yang et al., S3-FIFO, SOSP'23 (PDF)](https://jasony.me/publication/sosp23-s3fifo.pdf)
- SIEVE: "twice the throughput of an optimized 16-thread LRU", lock-free hits, up to 63.2% lower miss ratio than ARC, and best of 9 algorithms on over 45% of 1,559 traces. Adoption took fewer than 20 lines changed on average across 5 production libraries. — [Zhang et al., SIEVE, NSDI'24](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo)

**Cold-RL, NGINX, arXiv preprint Aug 2025, not peer-reviewed**
- This is the closest analogue to the hackathon design: a dueling DQN of about 10K parameters (128→64 ReLU, then value and advantage heads), served from an ONNX Runtime sidecar over a Unix domain socket. It samples K LRU-tail objects (K = 16 best) and uses 6 features: age, size, hit count, inter-arrival time, TTL remaining, last origin RTT. A hard 500 µs timeout falls back to LRU. Reported: p50 inference 127 µs, p95 eviction latency 498 µs, p99 710 µs, under 2% CPU at 50k req/s, 0.02% fallback rate. Hit ratio rises from 0.1436 to 0.3538 at 25 MB and from 0.7530 to 0.8675 at 100 MB, and ties at 400 MB. — [Gupta & Bhayani, Cold-RL, arXiv:2508.12485](https://arxiv.org/abs/2508.12485)
- Caveats: it is a preprint; the p99 exceeds the stated 500 µs hard timeout, which is internally inconsistent; and it has no comparison against LRB or other Belady-supervised methods. — [arXiv HTML](https://arxiv.org/html/2508.12485)

**Compiled GBDT inference**
- lleaves compiles LightGBM models to native code through LLVM. On NYC-taxi, single-row prediction takes 9.61 µs vs 52.31 µs in LightGBM, and a batch of 100 takes 31.88 µs vs 441.15 µs. The README says it outperforms Treelite and ONNX Runtime in its benchmarks. Usage: `lleaves.Model(model_file=...).compile()`. — [lleaves README](https://github.com/siboehm/lleaves)

### Inferences
- The current design (scoring 16 LRU-tail candidates on every miss) is already HALP/Cold-RL shaped. The candidate set is not the problem. The per-call cost of the Python/NumPy interpreter probably is: several small `np.dot` calls on 16×F arrays spend most of their time in call dispatch, not arithmetic. This is my reasoning, not a sourced measurement. Measure it with a profiler before rewriting.
- In order of effort, these should bring latency close to LRU:
  1. Cut candidates to 4–8. HALP found little gain beyond 4.
  2. Skip the model when a cheap rule decides. For example, if S3-FIFO's small queue has a one-hit wonder, evict it without scoring.
  3. Maintain features incrementally at request time, LRB-style: shift the deltas, and update EDCs from a precomputed decay table.
  4. Replace the MLP with LightGBM compiled through lleaves or Treelite, or with a one-hidden-layer MLP compiled by Numba `@njit`, so a whole eviction is one native call.
  5. Move the hot path (cache structure plus scoring) to Rust via PyO3/maturin, or to C through libCacheSim's native plugin path.
- Batching or asynchronous inference only helps in a real server (LRB's lock-free admission queue; Cold-RL's sidecar with fallback). In a single-threaded simulator it adds overhead. Do not build it.
- Caching scores per object and updating them lazily is implied by LRB updating features only on access or sampling. GL-Cache takes it further by scoring groups. For a hackathon, the simplest version is to compute an object's score when it is accessed and store it in the object record; eviction then becomes an argmax over stored scores (a staleness trade-off; features like age drift between updates).
- Report end-to-end simulated requests per second against libCacheSim's native LRU/S3-FIFO/SIEVE (which reach tens of millions of requests per second), not only miss ratio. S3-FIFO and SIEVE are the correct "is ML worth it" baselines in 2026.

### Gaps
- I found no source measuring int8 quantization for cache-eviction models. At about 10K parameters the model is likely dispatch-bound, not arithmetic-bound, so quantization probably will not help much. Unverified.
- No independent per-call latency numbers for ONNX Runtime or Treelite on tiny MLPs/GBDTs, other than lleaves' own comparison (vendor benchmark) and Cold-RL's sidecar numbers, which include IPC.
- No throughput figure (Mops) was extracted for LRB vs LRU in simulation beyond GL-Cache's relative "228×". I could not get 3L-Cache's absolute throughput or its artifact URL from the abstract page.
- No sourced Numba vs NumPy vs Rust micro-benchmarks for this exact workload. The team should measure its own.

---

## Training pipelines: labels, features, class balance, and online retraining

### Takeaway
The proven recipe is supervised imitation of (relaxed) Belady, not reward-based RL. Label each sampled object with its log time-to-next-request, computed offline from the trace or online with a delayed-labeling window. Train a GBDT on deltas, EDCs and static features. Retrain from a sliding window at a fixed sample count: LRB uses 128K samples; HALP does online mini-batches of 1,024 pairwise examples. The label and training-data design matters more than the model.

### Cited Findings

**Relaxed Belady and the Belady boundary (LRB)**
- Relaxed Belady evicts any object whose next request is beyond a threshold, the "Belady boundary" (the minimum time-to-next-request among objects Belady's MIN evicts, assumed roughly stationary). This lets random sampling work, because the sample only needs one object beyond the boundary. The cost: relaxed Belady has a 9–13% higher byte miss ratio than MIN on Wikipedia. — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)
- The "good decision ratio" is the fraction of evictions whose victim's next request is beyond the boundary. It needs one simulation to compute and "correlates strongly with byte miss ratio". LRB used it to tune every design choice (features, model, sample size, window) without full replays. — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)

**Labels and loss (LRB)**
- The target is log(time-to-next-request) via regression. Objects not re-requested within the sliding window get label = 2× window size. Rejected alternatives: time from last to next request, binary classification relative to the boundary, and raw time-to-next-request. L2 loss beat the other 7 LightGBM objectives. — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)
- Sampling for training: "periodically takes a random sample of objects in the sliding memory window". It samples over objects rather than requests, to avoid bias toward popular objects, and over all objects in the window, not only cached ones, so the data covers objects beyond the boundary. Labels are resolved on the next request, or once elapsed time exceeds the boundary. — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)
- Retraining: when the labeled set reaches 128K examples, train a new GBM, replace the old model, and empty the set. Accuracy kept rising up to 512K samples but with diminishing returns. Window length is a hyperparameter tuned on the first 20% of each trace (the "validation" section), and warmup sections are excluded from metrics. — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)

**Features that work**
- LRB uses 32 deltas (delta1 = time since last request, delta_n = gap between the (n-1)th and nth past requests, missing = ∞), 10 EDCs, and static features (size, type). EDC update: C_i = 1 + C_i × 2^(−Delta1 / 2^(9+i)). Ablations show each added feature group raises the good decision ratio, and 32 deltas plus 10 EDCs are past diminishing returns. — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf). The paper renders the EDC decay exponent as a superscript; I reproduced it from the paper's formula, so check it against the PDF.
- HALP feature table: time between accesses (32), EDCs (10), number of accesses (1), average time between accesses (1), time since last access (1), plus one domain feature ("end of chunk"). It notes these are "the same as the features used in" LRB. — [HALP PDF, Table 1](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf)
- Cold-RL's 6-feature set: age, size, hit count, inter-arrival time, TTL remaining, origin RTT. — [Cold-RL (preprint)](https://arxiv.org/abs/2508.12485)

**Pairwise and online labels (HALP)**
- Every pairwise comparison made at eviction time is saved as an unlabeled feature snapshot. A background process watches requests and assigns the label when one of the pair is re-accessed. Metadata for evicted keys lives in an LRU-bounded ghost cache sized as a multiple of the real cache. When the replay buffer reaches 1,024 examples, one mini-batch update runs. — [HALP PDF](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf)
- Retrain-interval ablation: updating every 1 example vs every 108 examples changed P95 byte miss by under 0.2%. They kept 1,024 for robustness at 1.8% CPU. — [HALP PDF §5](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf)
- Offline data collection through a trace simulator (Cold-RL: replay NGINX logs, reward +1 if a retained object is hit before TTL expiry). — [Cold-RL (preprint)](https://arxiv.org/abs/2508.12485)

**Class imbalance and one-hit wonders**
- One-hit wonders dominate short windows. The median one-hit-wonder ratio across S3-FIFO's traces is 26% over the full trace but 72% over sequences containing 10% of unique objects. — [S3-FIFO PDF](https://jasony.me/publication/sosp23-s3fifo.pdf)
- LRB's answers are to sample training examples over objects rather than requests, and to store one-hit objects compactly. — [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)

### Inferences
- Concrete replacement for the naive DQN in this project:
  1. Generate a longer trace, at least 1–10M requests. Use more than Zipf: mix in scans, one-hit wonders and popularity shift, or use real traces (next section).
  2. Compute each request's next-access index offline with a single reverse pass and a dict. libCacheSim's oracleGeneral format already stores `next_access_vtime`.
  3. At every miss in a simulated run, log the features of the (4–16) candidates and a label of log(next_access − now). Use 2× window for never-again or beyond-window objects.
  4. Train LightGBM with L2 loss, or with a lambdarank objective grouped by eviction event.
  5. Evict the argmax predicted reuse distance.
  This is behavioral cloning of Belady with LRB's labels. It needs no reward shaping, target network or replay-buffer tuning.
- Validation: split by time, like Parrot's 80/10/10 or LRB's first-20% validation section, never randomly. Track good decision ratio as the fast proxy, and full-replay miss ratio and byte miss ratio as the final metrics.
- Continual updates: keep LRB-style delayed labeling in the simulator (pending samples resolved on re-access or boundary timeout) and retrain every N labeled samples. Retraining LightGBM on 128K rows took 300 ms in LRB's C++ system, so periodic retraining from Python is plausible. HALP-style online mini-batches suit an MLP better.
- Class-balancing is less important with regression on log reuse distance than with binary labels. If using binary "beyond boundary" labels, reweight one-hit wonders or downsample them.

### Gaps
- LRB's exact decay-constant list and the static features for a KV workload were not fully re-verified (the extraction lost superscripts).
- I found no systematic study comparing lambdarank vs L2 regression for eviction. Parrot's ranking-loss result (next section) is the closest evidence.

---

## Offline RL and imitation alternatives to DQN (behavioral cloning, DAgger, CQL/IQL, Decision Transformer)

### Takeaway
The evidence favors imitation of Belady over RL. Parrot found that RL (a prior MDP-based approach) "results in lower performance". DAgger adds about 9.8% normalized hit rate over plain behavioral cloning on average but is program-dependent. A ranking loss beats plain log-likelihood by about 3.5%. I found no peer-reviewed evidence that CQL, IQL or Decision Transformer help eviction. The only 2025 "offline RL" eviction paper (Cold-RL) is a preprint that does not compare against Belady-supervised baselines.

### Cited Findings
- Parrot (Liu et al., ICML 2020): an LSTM over (address, PC) embeddings with attention over cache lines. It imitates Belady and beats Glider by 20% normalized hit rate on SPEC2006, and raises Web Search normalized hit rate by 61% over LRU (13.5% raw). — [Parrot, arXiv:2006.16239](https://arxiv.org/abs/2006.16239) ([PDF](https://arxiv.org/pdf/2006.16239))
- DAgger in Parrot: the first data collection follows Belady. After that, rollouts follow the current learned policy, and every visited state is relabeled by Belady, which is computable at any state because the future trace is known offline. The data buffer is recollected every 5,000 parameter updates. Ablation: "training on-policy leads to an average 9.8% normalized cache hit rate improvement over off-policy training", but on mcf and Web Search off-policy was as good or better. — [Parrot PDF](https://arxiv.org/pdf/2006.16239)
- Parrot's losses: a differentiable NDCG ranking loss with reuse distance as relevance. It equals a distillation target from a softmax-smoothed Belady and is +3.5% vs log-likelihood. An auxiliary loss regresses log reuse distance. Ablating reuse-distance prediction costs 16.8% normalized hit rate. Accuracy saturates at about 80 past accesses of history. — [Parrot PDF](https://arxiv.org/pdf/2006.16239)
- Parrot on practicality: "deploying such learned policies requires solving practical challenges, e.g., model latency may overshadow gains". Per-address embeddings "can require tens of megabytes"; a smaller byte-embedder variant still beats Glider by 8%. — [Parrot PDF](https://arxiv.org/pdf/2006.16239)
- Parrot on RL: Wang et al. (2019) "apply reinforcement learning instead of imitation learning, which results in lower performance." Hawkeye and Glider learn a binary cache-friendly/averse classifier from Belady and use a heuristic to break ties. — [Parrot PDF](https://arxiv.org/pdf/2006.16239)
- Glider distills an offline deep model into a tiny online model, which is the template for "teacher offline, cheap student online". — [Glider PDF](https://www.cs.utexas.edu/~akanksha/micro19glider.pdf)
- Cold-RL trains a dueling DQN offline from log replay (preprint), and has no Belady or LRB comparison. — [arXiv:2508.12485](https://arxiv.org/abs/2508.12485)

### Inferences
- For this project, drop DQN. A Belady oracle is free in simulation, so the problem is supervised. The standard RL reasons to prefer RL (no expert, delayed reward) do not apply. CQL, IQL and Decision Transformer solve the "no expert, fixed dataset" problem, which this project does not have.
- A cheap DAgger loop for the simulator: train the model with behavioral cloning on Belady-driven states, then replay the trace with the learned policy while logging candidates with their true next-access labels. Aggregate the data, retrain, and repeat 2–3 rounds. It is a few dozen lines on top of the logging you already need.
- If a larger teacher is wanted (a transformer over access history, à la Parrot), distill it into the GBDT or one-layer MLP student, following Glider. PEFT/LoRA only matters if that teacher is a pretrained transformer. For a hackathon this is gold-plating.

### Gaps
- No peer-reviewed application of CQL, IQL or Decision Transformer to cache eviction was found in the searches run. I may have missed workshop papers.
- A Feb 2026 preprint, "CacheMind" (arXiv:2602.12422, LLM reasoning about cache replacement), appeared in search results but was not read. Its relevance is unverified. — [arXiv:2602.12422](https://arxiv.org/pdf/2602.12422)

---

## Simulation and evaluation tooling: libCacheSim, traces, ghost/shadow caches

### Takeaway
libCacheSim is the de facto 2026 simulator. It has Python bindings (`pip install libcachesim`), built-in LRB, GL-Cache, 3L-Cache, S3-FIFO, SIEVE and Belady, an oracle trace format carrying next-access times, a Python plugin hook API, and thousands of public traces. For online comparison, both HALP and S3-FIFO use ghost caches (metadata-only history of evicted keys). HALP additionally argues for per-machine impact-distribution analysis rather than mean-only A/B tests.

### Cited Findings
- libCacheSim (C) reaches "over 20M requests/sec for a realistic trace replay". It supports txt, csv, binary and zstd traces. The oracleGeneral format stores `next_access_vtime` for Belady. LRB, GLCache and 3LCache need build flags (`-DENABLE_LRB=ON`, etc.). — [libCacheSim README](https://github.com/1a1a11a/libCacheSim)
- Python package: `pip install libcachesim`, with `SyntheticReader(num_objects, num_of_req, alpha, dist="zipf")` and `cache.process_trace(reader)` returning (object miss ratio, byte miss ratio). `PluginCache` takes Python hooks `init_hook`, `hit_hook`, `miss_hook`, `eviction_hook(data, request) -> int`, `remove_hook` and `free_hook`. `TraceReader` supports `TraceType.ORACLE_GENERAL_TRACE`. — [libCacheSim-python](https://github.com/cacheMon/libCacheSim-python); [libCacheSim README](https://github.com/1a1a11a/libCacheSim)
- Trace datasets: "thousands of traces hosted on S3" through the cacheMon cache dataset repository. — [cacheMon/cache_dataset](https://github.com/cacheMon/cache_dataset). S3-FIFO evaluated 6,594 traces from 14 datasets (including Twitter and MSR). — [S3-FIFO PDF](https://jasony.me/publication/sosp23-s3fifo.pdf)
- Ghost caches: HALP keeps feature metadata for evicted keys in an LRU-bounded ghost cache, so labels can resolve after eviction. S3-FIFO's ghost queue G holds as many entries as the main queue. — [HALP PDF](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf); [S3-FIFO PDF](https://jasony.me/publication/sosp23-s3fifo.pdf)
- Evaluation practice: HALP used simulation (first day as warm-up) plus production A/B tests, and warns that mean-shift tests hide machines where a learned policy regresses. Its "impact distribution analysis" examines the per-machine distribution of effects. LRB tunes on the first 20% of each trace and excludes warmup. — [HALP PDF](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf); [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)

### Inferences
- Lowest-effort path: keep the team's own simulator for the ML policy, but validate it by also running LRU, S3-FIFO, SIEVE, LRB and Belady from libCacheSim on the same traces. Check that the team's own LRU matches libCacheSim's LRU miss ratio.
- A Python `PluginCache` whose `eviction_hook` calls a model will be slow, because Python runs per eviction. Use it for correctness, and use native builds for speed numbers.
- A shadow A/B inside one simulator run: maintain ghost/shadow LRU and S3-FIFO caches of the same size (metadata only), and report the learned cache's miss ratio relative to them over sliding windows. If the relative ratio regresses past a threshold, fall back to the heuristic, which mirrors Cold-RL's LRU fallback and HALP's heuristic floor.

### Gaps
- I did not measure `libcachesim` Python plugin throughput, and the docs give no plugin-vs-native comparison.
- I did not verify dataset sizes or licensing for the individual traces in cache_dataset.

---

## Tools and frameworks for continual learning and fine-tuning (2025–2026)

### Takeaway
For a small Python team, the pragmatic stack is: LightGBM for training (optionally with warm-start or periodic full retraining on a sliding window), lleaves or Treelite for compiled inference, and river for true per-sample online learners and drift detection (ADWIN). Use PyTorch → ONNX Runtime only if an MLP is kept and served from a separate process. Hugging Face PEFT/LoRA only matters with a transformer teacher.

### Cited Findings
- river (latest shown: 0.26.1) is a Python library for online ML with `learn_one`/`predict_one`. It includes Hoeffding trees, Adaptive Random Forest, linear models and drift detectors such as ADWIN. — [river docs](https://riverml.xyz/latest/)
- lleaves compiles LightGBM to LLVM for 5.4× (single row) to 13.8× (batch 100) speedups over LightGBM's own predictor. It also offers "direct binary linking for eliminating Python overhead entirely". — [lleaves README](https://github.com/siboehm/lleaves)
- Serving an ONNX model via a sidecar costs roughly 100s of µs per call in Cold-RL, including IPC, with a 500 µs timeout and LRU fallback (preprint). — [Cold-RL arXiv:2508.12485](https://arxiv.org/abs/2508.12485)
- HALP's continual learning is plain online SGD on mini-batches of 1,024 delayed-labeled pairwise examples. LRB's is full retraining every 128K labeled samples. Neither uses a dedicated continual-learning framework. — [HALP PDF](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf); [LRB PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)

### Inferences
- Recommended minimal pipeline:
  1. `libcachesim` or a custom trace generator produces oracle next-access times.
  2. The simulator logs candidate features and log-reuse labels at each miss.
  3. LightGBM regression is trained with a time-based split.
  4. The model is compiled with lleaves.
  5. The simulator calls one compiled predict per eviction on 4–16 candidates.
  6. Every N labeled samples from a sliding window (with delayed labels), retrain and swap the model.
  7. Run 2–3 DAgger rounds offline.
- If the team keeps an MLP instead: shrink it to one hidden layer, since HALP found more layers did not help. Write the forward pass in Numba `@njit` or Rust, and do online mini-batch updates HALP-style.
- Drift handling: river's ADWIN on the windowed miss-ratio gap vs the shadow LRU is a cheap retrain trigger.

### Gaps
- I did not confirm whether LightGBM's `init_model` warm-start helps eviction models vs full retraining. No cache-specific source was found.
- No 2025–2026 source was found using PEFT/LoRA for cache eviction.
- I did not fetch ONNX Runtime and Treelite docs for current version numbers or single-call latency, so no numbers are claimed for them.
