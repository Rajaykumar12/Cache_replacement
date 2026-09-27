# Learned Cache Replacement / Eviction Policies: State of the Art (to Sept 2026)

Verification legend: **[V]** = number or claim checked this session against the paper PDF, abstract, or official page. **[R]** = recalled from the paper's well-known abstract. The citation is right, but the exact figure was not re-fetched this session, so confirm it before quoting it in a final report. **[Preprint]** = not peer-reviewed as far as found.

---

## Q1. Imitation learning from Belady (Hawkeye, Glider, Parrot, Mockingjay, LRB, HALP, 3L-Cache, 2024–2026)

### Takeaway
The dominant, best-performing line of work is supervised imitation of Belady's MIN. It either predicts the time to next request (a regression or multiclass label) or learns pairwise or binary "cache-friendly vs averse" labels, and it scores only a small sampled candidate set. In software caches the practical winners are GBM (LightGBM) regressors on about 6–40 cheap per-object features: LRB, then 3L-Cache. HALP uses a tiny 2-layer MLP with pairwise preference labels. The most recent result (LAH/S4-FIFO, 2026) shows that learning a heuristic's *parameters* can beat per-object learning in both robustness and cost.

### Cited Findings

**Hawkeye (CPU LLC)**: Jain & Lin, "Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement," ISCA 2016. DOI 10.1109/ISCA.2016.17.
- Core idea: **OPTgen** reconstructs what Belady's MIN *would have done* on past accesses of a few sampled cache sets. It uses an occupancy vector over a history window of about 8× the cache size, so for each reuse interval it can decide whether MIN would have hit. Those binary labels (cache-friendly / cache-averse) train a PC-indexed saturating-counter predictor. New lines from "averse" PCs are inserted at high eviction priority. — [Mockingjay paper's description of Hawkeye](https://www.cs.utexas.edu/~lin/papers/hpca22.pdf) [V: the binary friendly/averse framing]; [ResearchGate record](https://www.researchgate.net/publication/309138081_Back_to_the_Future_Leveraging_Belady's_Algorithm_for_Improved_Cache_Replacement)
- Features: load PC only. Model: counter table. Training: online and continuous from OPTgen labels. [R]

**Glider (CPU LLC)**: Shi, Huang, Jain, Lin, "Applying Deep Learning to the Cache Replacement Problem," MICRO 2019. DOI 10.1145/3352460.3358319.
- Core idea: train an offline attention-LSTM on the PC history, using Hawkeye/OPTgen labels. The attention weights show that an *unordered* history of the last k unique PCs suffices. This is distilled into an online **Integer SVM** (ISVM) per PC, with hardware-feasible cost. — [NSF PAR record](https://par.nsf.gov/biblio/10113801-applying-deep-learning-cache-replacement-problem)
- Reported (abstract): miss-rate reduction over LRU of about 8.9% vs about 7.1% for Hawkeye (single-core). [R — verify]
- Lesson for us: the heavy offline model was used as a *feature-discovery tool*. The deployed model is linear. This supports a "big model offline, tiny model online" design.

**Parrot**: Liu, Hashemi, Swersky, Ranganathan, Ahn, "An Imitation Learning Approach for Cache Replacement," ICML 2020. [arXiv:2006.16239](https://arxiv.org/abs/2006.16239)
- Core idea: imitation learning of Belady using **DAgger**, so the training states come from the *learned policy's own* rollouts and Belady relabels them. Model: LSTM over the access history (address and PC embeddings) plus attention over cache lines. Loss: a **ranking loss** over the cache lines' Belady reuse distances, plus an auxiliary reuse-distance prediction loss. Gym environment released. — [arXiv abstract](https://arxiv.org/abs/2006.16239) [V]
- Results: 20% miss-rate improvement over the prior state of the art on 13 memory-intensive SPEC apps, and a **61% hit-rate improvement over LRU** on a Google web-search benchmark. — [arXiv abstract](https://arxiv.org/abs/2006.16239) [V]
- Overhead: an LSTM per access is far too slow for hardware or production caches. The paper positions itself as a proof of the headroom, not a deployable policy. [R]
- Directly relevant: DAgger is the principled fix for the current project's bug, where the agent learns from an LRU trajectory and never sees the consequences of its own evictions.

**Mockingjay (CPU LLC)**: Shah, Jain, Lin, "Effective Mimicry of Belady's MIN Policy," HPCA 2022. DOI 10.1109/HPCA53966.2022.00048. [PDF](https://www.cs.utexas.edu/~lin/papers/hpca22.pdf)
- Core idea: replace Hawkeye's *binary* label with **multiclass reuse-distance prediction** from a PC-based predictor. Each inserted line gets an ETA (estimated time of arrival), and the line with the farthest ETA is evicted. That is a direct emulation of MIN. The paper argues multiclass beats binary because it gives a finer ordering and is more robust to prediction error. — [PDF abstract](https://www.cs.utexas.edu/~lin/papers/hpca22.pdf) [V]
- Results: on 100 multi-core mixes (SPEC06/17 and GAP, no prefetcher), **+15.2% over LRU vs +7.6% SHiP and +12.9% Hawkeye**. Single-core: +5.7% vs Belady MIN's +6.0%, so it nearly closes the gap. With a prefetcher on high-MPKI CVP traces: +20.1% vs +13.4% for Hawkeye. — [PDF abstract](https://www.cs.utexas.edu/~lin/papers/hpca22.pdf) [V]
- Relevance: "predict the reuse distance, then evict the max ETA" is the simplest principled replacement for a value-MLP, and it maps directly onto a key-value simulator (use key-hash or key features in place of the PC).

**LFO**: Berger, "Towards Lightweight and Robust Machine Learning for CDN Caching," HotNets 2018. DOI 10.1145/3286062.3286082.
- Core idea: compute the OPT decisions offline (via a min-cost-flow formulation) and train a GBDT to classify admit/keep. It is an early precursor to LRB. LRB's evaluation includes LFO as a baseline. — [LRB paper §6 baseline list](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V: that LFO is a supervised-learning baseline in LRB]

**LRB (Learning Relaxed Belady)**: Song, Berger, Li, Lloyd, NSDI 2020. [Paper](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) · [talk page](https://www.usenix.org/conference/nsdi20/presentation/song) · GitHub: github.com/sunnyszy/lrb
- **Relaxed Belady**: at each eviction, split the cache into set 1 (next request is within a threshold) and set 2 (next request is beyond the threshold). Evict a *random* object from set 2 if it is non-empty, and otherwise fall back to Belady on set 1. This means the predictor only has to find *some* object beyond the threshold, not the farthest one. Prediction can therefore run on a small sample (64 candidates), and training objects only need tracking until they are re-requested or cross the threshold. — [LRB §3.1](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]
- **Belady boundary**: the minimum time-to-next-request among all objects that Belady's MIN evicted. It is computed on a trace prefix and assumed to be roughly stationary. On Wikipedia, relaxed Belady costs 9–13% more misses than MIN, much less than the 25–40% gap between the best online heuristics and MIN. — [LRB §3.2, Table 2](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]
- **"Good decision ratio"** is used as an offline design metric: an eviction is "good" if the evicted object's next request is beyond the Belady boundary. This allows fast design-space exploration without full simulation. — [LRB §3.3](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]
- **Label construction**: sample *objects* (not requests, to avoid bias toward popular objects) from a sliding memory window. Label 1: when a sampled object is re-requested, the label is the time until that request. Label 2: when an object falls out of the sliding memory window, it is labelled as beyond the boundary. All such objects are "equivalently good" evictions. — [LRB §4.2](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]
- **Features**: up to 32 **deltas** (inter-request gaps; delta1 = age; these subsume LRU, LRU-K and S4LRU), 10 **exponentially decayed counters**, updated as C_i = 1 + C_i·2^(−Δ1/2^(9+i)) to give request rates over multiple horizons, and **static** features (size, content type). — [LRB §4.3.1](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]
- Model: **GBM (LightGBM)**, chosen over logistic regression and others by good decision ratio. The regression target is time to next request (log scale). [V: GBM chosen; LogReg compared in Fig. 6]
- Results: over 6 production CDN traces, **4–25% WAN-traffic reduction vs B-LRU** (LRU plus Bloom-filter admission, a typical production CDN). It consistently beats LRUK, LFUDA, S4LRU, Hyperbolic, GDSF, GDWheel, Adaptive-TinyLFU, LeCaR, UCB (RL), LFO, LHD and AdaptSize. — [LRB abstract, §6, Fig. 9](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]
- Overhead: an Apache Traffic Server prototype showed no measurable throughput loss at about 11.7 Gbps, with higher peak CPU and a memory overhead of a few GB. — [LRB Table 5](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]. However, in a large-scale simulator comparison by other authors, LRB's mean CPU cost was **172×** LRU at small cache sizes and **58×** at large ones, with peaks up to 300×. — [3L-Cache Table 1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]

**Raven**: Hu, Wang, Zhou, Jin, Zhang, Chen, "Raven: Belady-Guided, Predictive (Deep) Learning for In-Memory and Content Caching," CoNEXT 2022. DOI 10.1145/3555050.3569134.
- Core idea: predict the *distribution* of each object's next arrival time with a mixture density network, then evict by the probability of arriving farthest in the future. [R — not re-fetched]

**HALP (Google/YouTube)**: Song, Chen, Ma, Zhou, Liu, Liang, Du, Li, Lloyd, Berger (and others), "HALP: Heuristic Aided Learned Preference Eviction Policy for YouTube Content Delivery Network," NSDI 2023. [USENIX page](https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu) · [PDF](https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf) · [Google Research blog](https://research.google/blog/preference-learning-with-automated-feedback-for-cache-eviction/)
- Core idea: a **meta-algorithm**. A cheap heuristic (LRU by default, via random sampling that approximates a priority queue) proposes a few candidates, and a learned **reward model** (a lightweight **2-layer MLP**) picks among them. Randomization mixes the heuristic and the learned model. — [Google blog](https://research.google/blog/preference-learning-with-automated-feedback-for-cache-eviction/) [V]
- Training signal: **pairwise preference labels with automated feedback**. At each eviction, pairwise comparisons between candidates are appended to a pending buffer. They are resolved asynchronously when either item is re-accessed (the one re-accessed later is the better one to evict), mimicking the offline oracle. Training is fully online from random initialization, so each server specializes. — [Google blog](https://research.google/blog/preference-learning-with-automated-feedback-for-cache-eviction/) [V]
- Features: a metadata-only ghost cache holds internal features (time since last access, average inter-access time) plus external tags supplied with the request. — [Google blog](https://research.google/blog/preference-learning-with-automated-feedback-for-cache-eviction/) [V]
- Results: in YouTube CDN DRAM production since early 2022, **−9.1% byte miss at peak on average, for 1.8% CPU overhead**. — [search summary of USENIX abstract](https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu) [V]. The blog reports a +12% memory egress/ingress ratio and +6% memory hit rate. — [Google blog](https://research.google/blog/preference-learning-with-automated-feedback-for-cache-eviction/) [V]
- It samples only 4 candidates per eviction (LRB samples 64). — [3L-Cache §2.2](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]. In a libCacheSim-style comparison, HALP cost 23× LRU CPU at small cache sizes and 10× at large. — [3L-Cache Table 1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- Relevance: this architecture is closest to the current project (score bottom-k LRU candidates with a small MLP). The difference is that HALP's label is a principled pairwise "who comes back later", not a TD value on hit/miss reward.

**GL-Cache (group-level learning)**: Yang, Mao, Yue, Rashmi, "GL-Cache: Group-level Learning for Efficient and High-Performance Caching," FAST 2023. [USENIX page](https://www.usenix.org/conference/fast23/presentation/yang-juncheng)
- Core idea: cluster objects into groups (for example by write time, as in a log-structured cache), learn a utility per *group*, and evict whole groups. This amortizes learning cost. — [USENIX page](https://www.usenix.org/conference/fast23/presentation/yang-juncheng) [V]
- Results on 118 block I/O and CDN traces: vs LRB, **228× throughput**, +7% mean hit ratio and +25% at P90. Vs the best learned cache, +64% throughput, +3% mean hit ratio and +13% at P90. — [USENIX page](https://www.usenix.org/conference/fast23/presentation/yang-juncheng) [V]
- Caveat: an independent evaluation found GL-Cache has the best object miss ratio but a **worse byte miss ratio than LRU** (−8.8% small cache, −37.6% large cache), because it prefers evicting small objects. — [3L-Cache Table 1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]

**MAT**: Yang, Berger, Li, Lloyd, "A Learned Cache Eviction Framework with Minimal Overhead," [arXiv:2301.11886](https://arxiv.org/abs/2301.11886) (2023) **[Preprint]**.
- Core idea: use a heuristic as a filter so the ML model is invoked far less often than in LRB. [R — the exact speedup was not verified this session]

**3L-Cache**: Zhou, Niu, Xiong, Fang, Wang, "3L-Cache: Low Overhead and Precise Learning-based Eviction Policy for Caches," FAST 2025. [PDF](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) · [USENIX page](https://www.usenix.org/conference/fast25/presentation/zhou-wenbin) · code: github.com/optiq-lab/3L-Cache
- Model: **LightGBM GBM** regressing **log(next-arrival interval)** with MSE loss, using only **6 features**: age, size, frequency within the window, and the last 3 inter-arrival times (∞ if missing). The GBM was chosen because it handles missing features without imputation. Storage is about 67 B per object. — [§4.2](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- Labels: sample one object per request from a sliding window. When a sampled object is re-requested, its label is the interval until that arrival. If it leaves the window without a request, the label is its waiting time plus the longest waiting time seen among samples (a relaxed-Belady-style "far" label). The model retrains after every M labelled samples, with M best in [32K, 64K] across traces. The window size adapts to the trace and cache size. — [§4.2](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- Eviction: **bidirectional sampling**. It samples from the queue tail (old, likely unpopular objects) and from the head. This yields more unpopular candidates than LRB's uniform random 64, so it can evict *several* objects per prediction batch, which cuts inference cost. — [§4.3](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- Evaluation: 4,855 traces from 8 open datasets (CloudPhysics, Tencent CBS, Twitter, Alibaba, Wikipedia, Tencent Photo, Meta KV, Meta CDN), 14.7B objects and 267B requests. Cache sizes are 0.1% and 10% of the footprint, simulated in libCacheSim. Baselines are LHD, GDSF, ARC, SIEVE, S3-FIFO, TinyLFU, LeCaR, CACHEUS, GL-Cache, LRB and HALP. — [§5.1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- Results: **lowest byte miss ratio on 69.8% of traces (small cache) and 41.2% (large)**, vs 10.2% and 18.5% for the runner-up LRB. **Lowest object miss ratio on 66.3% and 29.3%**. On Tencent CBS at small size, the mean object-miss reduction vs LRU is 23.4% (>47.9% at P90). **CPU is 60.9% lower than HALP and 94.9% lower than LRB, at 6.4× LRU (small cache) and 3.4× (large)**. — [abstract, §5.2](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- Honest negatives from the same paper: **S3-FIFO came 3rd in byte miss ratio, beating HALP**, because most production workloads are Zipf-like. On large caches, GDSF beat 3L-Cache on all four small datasets for object miss ratio, because sampling-based eviction limits how precise the choice can be. LHD won on Alibaba (large). — [§5.2](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- The same paper's Table 1 reports mean miss-ratio reduction vs LRU (small/large cache), byte then object. LeCaR: 2.9/2.4%, 4.9/2.9%. CACHEUS: 5.1/2.4%, 8.1/3.1%. GL-Cache: −8.8/−37.6%, 6.3/9.6%. LRB: 6.8/7.1%, 6.0/−1.8%. HALP: 5.5/4.4%, 6.6/2.6%. In other words, **the mean gains of learned policies over LRU across thousands of real traces are single-digit percent**. — [3L-Cache Table 1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]

**Learning-Augmented Heuristics (LAH) / S4-FIFO (2026)**: Xia, Nixon, Marthen, Bhandari, Yang, "Learning-Augmented Heuristics: Simple, yet Smart, Robust and Interpretable Cache Eviction," [arXiv:2608.27975](https://arxiv.org/abs/2608.27975), submitted 28 Aug 2026. The arXiv page lists OSDI '26; peer-review status was not independently confirmed.
- Core idea: learn the **cache-level parameters** of a static heuristic (for example, S3-FIFO's small-queue ratio and promotion thresholds), not per-object scores. The data plane stays a fast heuristic, and a control plane learns asynchronously from cache-level features. One pre-trained model is trained on 4,140 production traces. — [arXiv abstract](https://arxiv.org/abs/2608.27975) [V]
- Results on 1,035 evaluation traces: S4-FIFO has **+26% mean efficiency vs S3-FIFO** and is +8% better than 3L-Cache. Its worst-case robustness vs FIFO is +0.8% miss ratio, against +8.8% for 3L-Cache. It is interpretable, and the authors show an LLM explaining its configuration choices. — [arXiv abstract](https://arxiv.org/abs/2608.27975) [V]. The exact definition of "efficiency" was not read.

**Other 2025–2026 items found (not deeply read):** SL-Cache (selective learning with retention of hot objects; [Springer](https://link.springer.com/chapter/10.1007/978-981-92-0363-5_36)). Vulcan, which uses LLM-driven search for instance-specialized, verifiable systems heuristics including cache eviction ([arXiv:2512.25065](https://arxiv.org/pdf/2512.25065), **[Preprint]**). Clock2Q+ for the VMware vSAN metadata cache ([arXiv:2511.21958](https://arxiv.org/pdf/2511.21958), **[Preprint]**). CacheMind, natural-language, trace-grounded reasoning about replacement ([arXiv:2602.12422](https://arxiv.org/pdf/2602.12422), **[Preprint]**).

### Inferences
- The consensus recipe is to **label from the future offline or with delay** (next-arrival time, or a Belady boundary / "beyond the window" label), **train a cheap supervised model** (GBM, or a small MLP with pairwise loss), and **run inference only on a small sampled candidate set** drawn from the cold end of a queue. That recipe is supervised regression or ranking, not TD learning.
- For a Python simulator with cache size 30, exact Belady (next-use distance) is trivially computable offline. That makes Parrot-style DAgger or Mockingjay-style reuse-distance regression cheap to implement: at each eviction, the label for every candidate is its true next-use distance.
- Replacing the current TD value target with the label "log(time to next access), capped at a boundary" and a ranking or regression loss fixes the core methodological flaw. Adding DAgger rollouts, where the learned policy drives the cache and Belady relabels the states, fixes the distribution-shift flaw.
- Feature upgrade path, in rough order of value per the LRB and 3L-Cache ablations: the last k inter-arrival deltas, multi-horizon exponentially decayed counters, age, frequency within the window, and size if objects vary in size.

### Gaps
- Exact Hawkeye and Glider headline numbers were not re-fetched (marked [R]).
- MAT's and Raven's quantitative results were not verified.
- LAH/S4-FIFO's metric definitions and final venue were not verified beyond the arXiv abstract.

---

## Q2. RL-based eviction (RLR, DeepCache / DRL caches, LeCaR, CACHEUS, 2023–2026 RL)

### Takeaway
Pure RL has rarely beaten Belady imitation on hit rate. Where RL "worked", it was either used offline as a *design tool* distilled into a simple rule (RLR), or it was really bandit / regret-minimization over a small set of heuristics (LeCaR, CACHEUS), which caps it at the best expert. Independent large-scale evaluations show LeCaR and CACHEUS losing to SIEVE and S3-FIFO on some datasets. The newest RL systems work (Cold-RL, 2025) is an unreviewed preprint with weak baselines.

### Cited Findings
- **RLR**: Sethumurugan, Yin, Sartori, "Designing a Cost-Effective Cache Replacement Policy using Machine Learning," HPCA 2021, pp. 291–303. DOI 10.1109/HPCA51647.2021.00033. They trained a DRL agent offline, analysed what it learned, and hand-derived a cheap hardware policy (RLR) that needs no PC. RLR gives **+3.25% single-core and +4.86% four-core** performance over LRU, at 16.75 KB of overhead for a 2 MB LLC. — [UMN Experts record](https://experts.umn.edu/en/publications/designing-a-cost-effective-cache-replacement-policy-using-machine/) [V via search summary]. The gains are smaller than Mockingjay's +5.7% single-core (above). The paper's selling point is low cost and not needing a PC, not the hit rate.
- **LeCaR**: Vietri, Rodriguez, Martinez, Lyons, Liu, Rangaswami, Zhao, Narasimhan, "Driving Cache Replacement with ML-based LeCaR," HotStorage 2018. It does online regret minimization over two experts (LRU and LFU) and updates weights when a miss hits the ghost history of an expert's past evictions. The claimed large wins over ARC at small cache sizes are [R — verify]. In an independent evaluation on 4,855 traces, LeCaR's mean miss reduction vs LRU was only 2.4–4.9%. — [3L-Cache Table 1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- **CACHEUS**: Rodriguez, Yusuf, Lyons, Paz, Rangaswami, Liu, Zhao, Narasimhan, "Learning Cache Replacement with CACHEUS," FAST 2021. It extends LeCaR with an adaptive learning rate and scan- and churn-resistant experts (SR-LRU, CR-LFU). [R]. Independent result: mean 2.4–8.1% miss reduction vs LRU. **"On some datasets LeCaR and CACHEUS do not perform as well as SIEVE and S3-FIFO... limiting their performance to those heuristics."** — [3L-Cache §5.2.1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- **UCB (bandit RL) baseline in LRB**: LRB beat it on CDN traces. — [LRB §6](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V: UCB listed as the RL baseline outperformed]
- **DeepCache**: Narayanan, Verma, Ramadan, Babarczi, Zhang, "DeepCache: A Deep Learning Based Framework for Content Caching," ACM SIGCOMM NetAI Workshop 2018. An LSTM encoder–decoder predicts future object popularity and injects "fake requests" into an existing policy such as LRU. It is popularity prediction, not RL. [R]
- **Cold-RL (2025)**: Gupta, Bhayani, "Cold-RL: Learning Cache Eviction with Offline Reinforcement Learning for NGINX," [arXiv:2508.12485](https://arxiv.org/abs/2508.12485) **[Preprint]**. It uses a **dueling DQN**, trained *offline* by replaying NGINX logs through a simulator and served through an ONNX sidecar with a **500 µs timeout that falls back to LRU**. On each eviction it samples the **K least-recently-used objects** and outputs a bitmask of victims. There are 6 features: age, size, hit count, inter-arrival time, remaining TTL, and last origin RTT. Reward: +1 if a retained object is hit before its TTL expires. Results: hit ratio 0.144→0.354 at 25 MB, 0.753→0.868 at 100 MB, and equal to the baselines (~0.918) at 400 MB. CPU overhead is below 2%. **Baselines were only LRU, LFU, size-based, adaptive LRU and a hybrid — no ARC, S3-FIFO, SIEVE, LRB or Belady.** — [arXiv abstract](https://arxiv.org/abs/2508.12485) [V]
  - Its architecture (bottom-K LRU candidates, a small Q-network, a latency-budget fallback) is almost identical to the current project. Unlike the current project, its reward is tied to the *retained* objects' future hits, which gives per-action credit assignment.
- **RL for LLM KV caches (2026)**: ForesightKV frames KV eviction as an MDP solved with RL and reports matching baselines at half the cache budget. — [arXiv:2602.03203](https://arxiv.org/html/2602.03203v1) **[Preprint]**

### Inferences
- The pattern across RLR, Parrot and the Google systems work is that when you *can* compute the oracle offline (you always can in a trace-driven simulator), supervised imitation of Belady beats RL on sample efficiency and stability. RL's cited advantages (delayed reward, no oracle needed) mostly do not apply to trace-driven eviction.
- If the team keeps an RL framing, the defensible versions are:
  - (a) Cold-RL-style per-candidate reward: "was the retained candidate hit before horizon H?", which is effectively a delayed binary Belady-like label.
  - (b) An expert-weighting bandit over strong heuristics, ARC/LeCaR-style, with S3-FIFO/SIEVE/LRU/LFU as experts.
  - (c) RL used only to tune heuristic parameters (the LAH idea).
- The current setup (a TD target on a global hit/miss reward along an LRU trajectory) combines the worst of both worlds. It has off-policy data with no importance correction, reward that is not attributable to the chosen victim, and no oracle signal.

### Gaps
- No peer-reviewed 2023–2026 paper was found showing a pure deep-RL eviction policy beating 3L-Cache, LRB or S3-FIFO on standard trace suites. That absence is itself a signal, but the search was not exhaustive.
- Exact LeCaR/CACHEUS headline claims and DRL-specific CDN papers (for example Kirilin et al., "RL-Cache," IEEE JSAC 2020) were not fetched.

---

## Q3. Strong modern heuristic baselines (ARC, LIRS, LHD, SIEVE, S3-FIFO, W-TinyLFU, GDSF)

### Takeaway
Since 2023, simple FIFO-family heuristics (S3-FIFO, SIEVE) are the baselines to beat. They match or beat many learned policies on average across thousands of traces, at LRU-or-better throughput. Any learned policy must be reported against S3-FIFO, SIEVE, ARC, W-TinyLFU and LHD/GDSF (when sizes vary), not just LRU.

### Cited Findings
- **S3-FIFO**: Yang, Zhang, Qiu, Yue, Rashmi, "FIFO Queues Are All You Need for Cache Eviction," SOSP 2023. DOI 10.1145/3600006.3613147. It uses three static FIFO queues: a small queue (10% of capacity) that quickly demotes one-hit wonders, a main queue (90%), and a ghost queue. It is lock-free-friendly and flash-friendly. Over 6,594 traces from 12 companies it achieves **up to 72% lower miss ratio than LRU** and needs **46% less cache for a 10% target miss ratio**. Throughput is **6× optimized LRU at 16 threads**. — [s3fifo.com](https://s3fifo.com/) [V]
- **SIEVE**: Zhang, Yang, Yue, Vigfusson, Rashmi, "SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for Web Caches," NSDI 2024 (Community Award). It is one FIFO queue with a visited bit and a "hand" that sweeps from tail to head, with no reordering on a hit. Over 1,559 traces from 7 sources it has **up to 63.2% lower miss ratio than ARC** and is **best of 10 algorithms on >45% of traces vs 15% for the runner-up**. Throughput is **2× an optimized 16-thread LRU**, and integrating it into production libraries took fewer than 20 lines of change on average. — [USENIX page](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo) [V]
- **S3-FIFO in the 3L-Cache comparison**: third-best in byte miss ratio, **ahead of HALP (learned)**. Its performance degrades as the Zipf exponent falls (more uniform popularity). — [3L-Cache §5.2.1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- **LHD**: Beckmann, Chen, Cidon, "LHD: Improving Cache Hit Rate by Maximizing Hit Density," NSDI 2018. It is a probabilistic, conditional hit-density model over age (and size), evaluated over sampled candidates. [R]. In the 3L-Cache comparison, LHD's mean object-miss reduction exceeded LRB's by 14.1 points on Tencent CBS. With uniform sizes (slab allocation) it degenerates to age-only and can do badly. On Tencent Photo (more than 50% one-hit wonders) it is best at small cache sizes but worse than LRU at large ones. — [3L-Cache §5.2.2](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- **GDSF**: Cherkasova, "Improving WWW Proxies Performance with Greedy-Dual-Size-Frequency Caching Policy," HP Labs Tech Report HPL-98-69, 1998. Priority = L + freq·cost/size. [R]. It **beat all policies, including 3L-Cache, on object miss ratio for large caches** on the four small datasets, because it scores all objects rather than a sample. — [3L-Cache §5.2.2](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- **ARC**: Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache," FAST 2003. It adapts the split between recency (T1) and frequency (T2), using ghost lists B1/B2. [R]
- **LIRS**: Jiang & Zhang, "LIRS: An Efficient Low Inter-reference Recency Set Replacement Policy," SIGMETRICS 2002. It ranks by reuse distance (inter-reference recency) and is scan-resistant. [R]
- **TinyLFU / W-TinyLFU**: Einziger, Friedman, Manes, "TinyLFU: A Highly Efficient Cache Admission Policy," ACM Transactions on Storage 13(4), 2017, DOI 10.1145/3149371. A count-min-sketch frequency *admission* filter with periodic halving (aging), combined with a small LRU window and an SLRU main region. It is the default in Java's Caffeine. [R]. LRB included Adaptive-TinyLFU as a baseline and beat it. — [LRB §6](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]
- **Belady gap**: on the CDN traces, the best online heuristics are still 25–40% worse than Belady MIN. — [LRB §2](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]

### Inferences
- For the hackathon: implement SIEVE (about 20 lines), S3-FIFO (about 50 lines), ARC and W-TinyLFU as baselines, plus Belady as the upper bound. A learned policy that only beats LRU is not a result.
- Admission matters as much as eviction when there are many one-hit wonders (S3-FIFO's small queue, TinyLFU's filter, B-LRU's Bloom filter). A learned policy that scores only the bottom-k LRU candidates cannot fix pollution from one-hit wonders. Consider pairing it with an S3-FIFO-style probationary queue.
- The LAH/S4-FIFO result (Q1) suggests a practical hybrid: keep S3-FIFO or SIEVE as the data plane and learn its parameters or its candidate scoring.

### Gaps
- The original ARC, LIRS and TinyLFU headline numbers were not re-fetched.
- The S3-FIFO page did not list authors. The author list above is from the ACM DOI record, which returned 403 this session, and is recalled.

---

## Q4. Learned eviction for LLM KV caches (transferable ideas only)

### Takeaway
LLM KV-cache eviction (H2O, SnapKV and newer learned variants) mostly uses accumulated attention as a "frequency" signal, protects a recency window, and increasingly *learns future utility directly*. The transferable ideas are a protected recent window plus a heavy-hitter or frequency score, and predicting forward-looking utility instead of relying on backward-looking statistics.

### Cited Findings
- **H2O**: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," NeurIPS 2023, [arXiv:2306.14048](https://arxiv.org/abs/2306.14048). It keeps recent tokens plus "heavy hitters" ranked by accumulated attention (an LFU analogue) and frames this as a dynamic submodular problem. [R]
- **SnapKV**: Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation," NeurIPS 2024, [arXiv:2404.14469](https://arxiv.org/abs/2404.14469). An observation window at the end of the prompt votes on which KV positions to keep, with pooling to keep clusters together. [R]
- **Learning to Evict from Key-Value Cache (2026)**: it learns a forward-looking, query-independent policy that predicts each token's future utility directly, instead of inferring utility from past attention or the current query. — [arXiv:2602.10238](https://arxiv.org/html/2602.10238v1) **[Preprint]** [V: abstract framing]
- **ForesightKV (2026)**: it learns long-term token contribution via an MDP/RL formulation and reports beating baselines at half the cache budget. — [arXiv:2602.03203](https://arxiv.org/html/2602.03203v1) **[Preprint]**
- **RAC (2026)**: relation-aware cache replacement for LLMs. — [arXiv:2602.21547](https://arxiv.org/pdf/2602.21547) **[Preprint]**, not read.

### Inferences
- "Recency window plus learned or heavy-hitter score" is structurally the same as S3-FIFO's small queue or W-TinyLFU's window plus a scored main region. It supports keeping a protected recency segment in front of any learned scorer.
- "Predict future utility directly" is the KV-cache restatement of LRB/3L-Cache's time-to-next-access regression.

### Gaps
- The exact H2O and SnapKV numbers were not fetched (and are not needed for a key-value simulator).

---

## Q5. Standard evaluation practice (traces, simulators, metrics)

### Takeaway
Credible 2023–2026 papers evaluate on hundreds to thousands of real production traces (CloudPhysics, MSR, Twitter, Wikipedia CDN, Meta KV/CDN, Tencent, Alibaba). They use libCacheSim, report **miss ratio reduction relative to LRU**, (mr_LRU − mr_alg)/mr_LRU, at 2+ cache sizes expressed as a fraction of the footprint, give object *and* byte miss ratio, and report CPU overhead or throughput relative to LRU. Synthetic Zipf traces at a single cache size of 30 are not accepted evidence.

### Cited Findings
- 3L-Cache's protocol is 8 datasets and 4,855 traces (CloudPhysics, Tencent CBS, Twitter, Alibaba, Wikipedia, Tencent Photo, Meta KV, Meta CDN), 14.72B objects and 266.95B requests, collected 2015–2023. Cache sizes are **0.1% and 10% of the footprint**. Metrics are the object and byte miss-ratio reduction relative to LRU and **CPU usage relative to LRU**. The simulator is built on **libCacheSim**, with a Python HTTP prototype. Box plots are used for large datasets and per-trace scatter plots for small ones. — [3L-Cache §5.1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]
- LRB's protocol is 6 production CDN traces (including Wikipedia), a trace prefix for validation and hyperparameter tuning, WAN traffic (byte miss) reduction vs B-LRU, and a prototype in Apache Traffic Server measuring throughput, CPU and memory. It also reports the good decision ratio as an offline design metric. — [LRB](https://www.usenix.org/system/files/nsdi20-paper-song.pdf) [V]
- SIEVE: 1,559 traces from 7 sources. S3-FIFO: 6,594 traces from 12 companies. Both report the fraction of traces on which each algorithm is best, plus multi-thread throughput. — [SIEVE](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo), [S3-FIFO](https://s3fifo.com/) [V]
- LAH: trains on 4,140 traces and evaluates on a held-out 1,035. It reports worst-case robustness (miss ratio relative to FIFO) as well as the mean. — [arXiv:2608.27975](https://arxiv.org/abs/2608.27975) [V]
- CPU-cache work (Hawkeye, Glider, Mockingjay, RLR) uses ChampSim with SPEC 2006/2017, GAP and CVP traces. It reports IPC speedup over LRU on single-core and multi-core mixes, with and without a prefetcher, plus the hardware budget in KB. — [Mockingjay abstract](https://www.cs.utexas.edu/~lin/papers/hpca22.pdf) [V]
- Parrot released a Gym environment for cache replacement. — [arXiv:2006.16239](https://arxiv.org/abs/2006.16239) [V]
- Trace sources and tools [R — links are canonical, contents not re-fetched]: **libCacheSim** (github.com/1a1a11a/libCacheSim, with a Python binding `libcachesim` on PyPI and many policies built in, including LRB, 3L-Cache-style GBM, S3-FIFO, SIEVE, ARC, LHD, TinyLFU, GDSF and Belady). The trace collection is at ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/ and the SNIA IOTTA repository. Twitter cache traces: Yang, Yue, Rashmi, "A Large Scale Analysis of Hundreds of In-memory Cache Clusters at Twitter," OSDI 2020. MSR Cambridge block traces: Narayanan, Donnelly, Rowstron, FAST 2008. CloudPhysics: Waldspurger et al., "Efficient MRC Construction with SHARDS," FAST 2015.
- A common production failure: several heuristics (LHD, ARC, TinyLFU, GL-Cache) "occasionally fail to reduce LRU's byte miss ratio". Robustness (the worst case vs LRU or FIFO) is now reported alongside the mean. — [3L-Cache §5.2.1](https://www.usenix.org/system/files/fast25-zhou-wenbin.pdf) [V]

### Inferences
- A minimum credible protocol for the hackathon:
  - Add 5–20 real traces through libCacheSim's Python binding or its CSV/oracleGeneral readers, for example Twitter cluster samples, MSR, and a Wikipedia CDN sample.
  - Sweep cache sizes at 0.1%, 1% and 10% of the footprint.
  - Report the miss-ratio reduction vs LRU, the gap closed to Belady ((mr_LRU − mr_alg)/(mr_LRU − mr_Belady)), p50/p99 per-eviction latency and requests/sec.
  - Include S3-FIFO, SIEVE, ARC, W-TinyLFU, LHD/GDSF and Belady.
  - Train and test on *different* traces, or on a time split, to show generalization.
- Cache size 30 on a synthetic Zipf trace is so small that almost any frequency-aware policy wins. Results there will not transfer.

### Gaps
- The current contents of libCacheSim's Python binding (the exact policy list and whether it includes 3L-Cache or HALP) were not verified this session. Check the repository README.
- The exact download URLs and licensing of the Meta, Tencent and Alibaba traces were not verified.
