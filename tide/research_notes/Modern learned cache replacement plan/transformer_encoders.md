# Transformer Encoders (ModernBERT, mmBERT, et al.) and Sequence Models for Cache Eviction / Prefetching — as of Sept 2026

Scope: whether modern pretrained encoders (ModernBERT, mmBERT, NeoBERT, Ettin, EuroBERT) fit a Python key-value cache simulator whose current eviction policy is a tiny 3-feature MLP; prior sequence-model work on access traces; how that work was made fast; key representation; fine-tuning recipes.

Preprint flag convention: **[preprint]** = arXiv-only as far as I could verify; **[peer-reviewed]** = venue confirmed in a fetched source; **[venue from memory]** = venue I believe is correct but did not confirm in a fetched page this session.

---

## Q1. What are ModernBERT, mmBERT, and the other modern small encoders (size, context, architecture, speed, license)? Does their natural-language pretraining help with opaque integer keys?

### Takeaway
ModernBERT (149M/395M) and mmBERT (140M/307M total, 42M/110M non-embedding) are 22-layer, 8k-context text encoders built for fast *GPU* inference over long natural-language/code sequences (RoPE, alternating 128-token local / global attention every 3rd layer, unpadding, FlashAttention 2). Their pretrained knowledge is about text tokens; for opaque integer key IDs it gives essentially nothing beyond "a transformer architecture", and it is only plausibly useful when keys carry semantic text (URLs, file paths, SQL, API routes, prompts).

### Cited Findings

**ModernBERT** (Warner, Chaffin, Clavié, Weller, Hallström, Taghadouini, Gallagher, Biswas, Ladhak, Aarsen, Cooper, Adams, Howard, Poli; Answer.AI / LightOn et al.) — *"Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference"*, arXiv:2412.13663, submitted 18 Dec 2024 **[preprint at time of arXiv submission]**.
- Native 8,192-token sequence length; trained on 2 trillion tokens; paper (the PDF) is licensed CC BY 4.0; describes itself as "the most speed and memory efficient encoder" and "designed for inference on common GPUs" — [arXiv 2412.13663](https://arxiv.org/abs/2412.13663)
- Sizes: Base 149M params, Large 395M params — [HF blog: Finally, a Replacement for BERT](https://huggingface.co/blog/modernbert)
- Architecture: RoPE positional embeddings; GeGLU activations; alternating attention with global attention every 3 layers and 128-token sliding-window local attention in between; no token type IDs — [HF blog](https://huggingface.co/blog/modernbert)
- Efficiency: unpadding + sequence packing to avoid computing on pad tokens; Flash Attention 2; hardware-aware design targeting RTX 3090/4090, A10, T4, L4 GPUs — [HF blog](https://huggingface.co/blog/modernbert)
- Data: 2T tokens of web documents, code, and scientific articles; English-only; claimed first encoder with substantial code in training data — [HF blog](https://huggingface.co/blog/modernbert)
- Speed claims: "twice as fast as DeBERTa", "up to 4x faster" with variable-length inputs, 2–3x faster than next-fastest model on long context; uses "less than 1/5th of DeBERTa's memory" — [HF blog](https://huggingface.co/blog/modernbert)
- Independent comparison: under identical pretraining conditions DeBERTaV3 retained better sample efficiency/benchmark performance, while ModernBERT is faster to train and infer — [ModernBERT or DeBERTaV3? arXiv 2504.08716](https://arxiv.org/pdf/2504.08716) **[preprint]** (claim taken from a search summary, not a full read)

**mmBERT** (Marone, Weller, Fleshman, Yang, Lawrie, Van Durme; JHU CLSP) — *"mmBERT: A Modern Multilingual Encoder with Annealed Language Learning"*, arXiv:2509.06888, 8 Sep 2025 **[preprint]**.
- Encoder-only, pretrained on 3T tokens across 1,800+ languages; novel recipe pieces: inverse mask-ratio schedule (high→low), annealed language sampling (biased→uniform), languages added per phase 60→110→1833 — [arXiv 2509.06888](https://arxiv.org/abs/2509.06888v1)
- Sizes: mmBERT-base 307M total / 110M non-embedding; mmBERT-small 140M total / 42M non-embedding; both 22 layers; hidden 768 (base) / 384 (small); 12 / 6 heads; intermediate 1152; vocab 256,000 (Gemma 2 tokenizer); max length 8,192; license **MIT** — [HF model card jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)
- Training phases: 2.3T tokens pretraining (60 langs), 600B mid-training (110 langs), 100B decay (1,833 langs) — [HF model card](https://huggingface.co/jhu-clsp/mmBERT-base)
- Architecture inherited from ModernBERT: 128-token local window, global attention every 3 layers, unpadding, Flash Attention 2; RoPE base 10k in pretraining, theta 160k after context extension to 8,192 — [mmBERT paper HTML](https://arxiv.org/html/2509.06888v1)
- Speed: "mmBERT base is more than 2x faster on variable sequences and significantly faster on long context lengths (~4x)" vs prior multilingual encoders; mmBERT-small ~2x faster than mmBERT-base; XLM-R/MiniLM cap at 512 tokens while mmBERT runs 8,192 tokens "as fast as other models at 512" — [mmBERT paper HTML](https://arxiv.org/html/2509.06888v1)
- Note: most of mmBERT-base's parameters (≈197M of 307M) are the 256k-token embedding table, which exists to cover 1,800 languages — [HF model card](https://huggingface.co/jhu-clsp/mmBERT-base) (arithmetic from card's numbers)

**Other modern encoders**
- **NeoBERT** (arXiv:2502.19587, v2 6 Jun 2025) **[preprint]**: 250M params, 4,096-token context, trained on 2T+ tokens, "optimal depth-to-width ratio", SOTA MTEB for its size — [arXiv 2502.19587](https://arxiv.org/abs/2502.19587). Uses RMSNorm + SwiGLU; reported 46.7% speedup over ModernBERT-base at 4,096 tokens — [Leonie Monigatti blog summary](https://www.leoniemonigatti.com/blog/neobert.html) (secondary source)
- **Ettin** (Weller, Ricci, Marone, Chaffin, Lawrie, Van Durme; *"Seq vs Seq: An Open Suite of Paired Encoders and Decoders"*, arXiv:2507.11412) **[preprint]**: paired encoder-only and decoder-only models, 17M → 1B params, identical data/recipe, up to 2T tokens (DCLM, Dolma v1.7, papers, code); encoders reported to beat ModernBERT; 400M encoder beats 1B decoder on classification — [arXiv 2507.11412](https://arxiv.org/html/2507.11412v1); [HF blog: Ettin](https://huggingface.co/blog/ettin); smallest checkpoint: [jhu-clsp/ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m)
- **EuroBERT** (arXiv:2503.05500) **[preprint]**: multilingual (European + major world languages) family of 210M, 610M, 2.1B params; native 8,192 context; stronger on code and math than XLM-R/mGTE-MLM — [arXiv 2503.05500](https://arxiv.org/abs/2503.05500v2)

**Does NL pretraining help with integer keys?**
- Parrot (the strongest attention-based replacement policy) learns embeddings **from scratch** for each PC and address "akin to word vectors", or a byte-level embedder over address bytes; it uses no pretrained language model — [Liu et al., ICML 2020, §4.5](https://arxiv.org/abs/2006.16239)
- Prefetching work similarly builds its own vocabulary from address deltas / address segments rather than using a text tokenizer (Hashemi et al. relate prefetching to n-gram language modeling but train RNNs from scratch) — [Hashemi et al. arXiv 1803.02329](https://arxiv.org/abs/1803.02329); TransFetch uses "fine-grained address segmentation" to cut vocabulary size — [NSF PAR record](https://par.nsf.gov/biblio/10376312-fine-grained-address-segmentation-attention-based-variable-degree-prefetching)
- Where keys *do* carry semantics, recent work exploits content type: SAECache finds "up to 756x variation in reuse rates" across token types in LLM prefix caches and learns per-type weights online, but does so with a lightweight adaptive weighting, not a pretrained text encoder — [Fang et al., arXiv 2605.18825, May 2026](https://arxiv.org/abs/2605.18825) **[preprint]**

### Inferences
- A pretrained ModernBERT/mmBERT tokenizer would split an integer key like `"4812337"` into arbitrary digit sub-tokens; the model's knowledge of English/1,800 languages has no causal relationship to reuse behaviour of opaque IDs. For integer keys, you would be using the architecture with random-ish useful weights, i.e., no advantage over a tiny transformer trained from scratch — and a large cost (150–400M params).
- mmBERT's multilingual advantage is irrelevant for cache keys unless the keys are multilingual text (e.g., search queries in many languages). Its large vocabulary actually makes it heavier than ModernBERT for this use.
- The only credible use of a pretrained encoder here is **semantic keys**: e.g., embed a URL/path/SQL string once, at *insert time*, and store the pooled vector (or a cluster ID / predicted "reuse class") as a per-key feature. That is a key-level feature extractor, not a per-eviction sequence model.
- If the team wants "a modern encoder" narratively, Ettin-encoder-17M is the most defensible pretrained choice (smallest, open data, same ModernBERT-style recipe); for integer traces, a 1–2 layer, 32–64-dim transformer trained from scratch is the honest option.

### Gaps
- Model-weight license for ModernBERT (I believe Apache 2.0 on the HF model card, but did not fetch the card this session; the 2412.13663 CC BY 4.0 license applies to the paper). Ettin, NeoBERT, EuroBERT licenses not verified.
- ModernBERT exact layer count / non-embedding parameter count not verified from a fetched primary source (mmBERT's card says 22 layers and it "inherits" ModernBERT's architecture).
- No published study found that applies ModernBERT, mmBERT, NeoBERT, Ettin, or EuroBERT to cache replacement or prefetching. I searched for LLM/foundation-model caching work and found none using these encoders.

---

## Q2. Prior work applying sequence / attention models to cache access streams (2018–2026)

### Takeaway
The lineage is: LSTMs for prefetching (Hashemi 2018) → offline LSTM+attention teacher distilled into an online ISVM for replacement (Glider 2019) → end-to-end imitation of Belady with LSTM + attention over cache lines (Parrot 2020) → hierarchical/attention prefetchers (Voyager 2021, TransFetch 2022) → distillation into lookup tables (DART, 2023/24). Recent (2024–2026) work in systems is dominated by LLM KV/prefix-cache eviction, much of it heuristic or lightweight-learned; no one has shown a BERT-scale encoder deciding evictions online.

### Cited Findings

**Hashemi et al., "Learning Memory Access Patterns"** — Hashemi, Swersky, Smith, Ayers, Litz, Chang, Kozyrakis, Ranganathan; arXiv:1803.02329 (Mar 2018) **[published ICML 2018 — venue from memory]**
- Relates prefetching to n-gram models in NLP; shows RNNs can serve "as a drop-in replacement" for prefetchers and "consistently demonstrate superior performance in terms of precision and recall" — [arXiv 1803.02329](https://arxiv.org/abs/1803.02329)

**Glider — Shi, Huang, Jain, Lin, "Applying Deep Learning to the Cache Replacement Problem", MICRO 2019** **[peer-reviewed]**
- Offline model: embedding layer → 1-layer LSTM → attention layer; shown to beat hardware predictors in accuracy offline — [UT Austin PDF](https://www.cs.utexas.edu/~lin/papers/micro19c.pdf); [ACM DL](https://dl.acm.org/doi/10.1145/3352460.3358319)
- Interpreting the LSTM yielded the insight to use an *unordered list of unique PCs* as history; online model is an Integer SVM (ISVM) matching offline accuracy "with orders of magnitude lower cost" — [UT Austin PDF](https://www.cs.utexas.edu/~lin/papers/micro19c.pdf)
- Results (33 memory-intensive SPEC06/SPEC17/GAP programs): miss-rate reduction over LRU 8.9% vs 7.1% Hawkeye, 6.5% MPPPB, 7.5% SHiP++; 4-core IPC +14.7% over LRU — [UT Austin PDF](https://www.cs.utexas.edu/~lin/papers/micro19c.pdf)

**Parrot — Liu, Hashemi, Swersky, Ranganathan, Ahn, "An Imitation Learning Approach for Cache Replacement", ICML 2020 (PMLR 119)** **[peer-reviewed]**
- Architecture: embed (address, PC) of each access → LSTM; keep past H hidden states; for each cache line l_w form a query e(l_w) and attend over the H hidden states (keys); dense + softmax over lines → eviction distribution — [arXiv 2006.16239, §4.1/Fig 3](https://arxiv.org/abs/2006.16239)
- Training: imitation of Belady; DAgger (re-collect states under current policy every 5,000 updates) gives +9.8% normalized hit rate vs off-policy; NDCG-style ranking loss with reuse distance as relevance gives +3.5% vs log-likelihood; auxiliary head predicting log reuse distance (MSE) gives +16.8% — [arXiv 2006.16239 §4.2–5.3](https://arxiv.org/abs/2006.16239)
- Using the predicted reuse distance *directly* to evict (evict highest predicted) was worse than using it as an auxiliary loss, because small errors in log reuse distance flip decisions; "ranking the lines may be relatively easy" — [arXiv 2006.16239 §5.3](https://arxiv.org/abs/2006.16239)
- Results: +16% raw hit rate over LRU averaged on 13 SPEC2006 apps; 20% higher normalized hit rate than Glider; on Google Web Search +61% normalized / +13.5% raw hit rate over LRU — [arXiv 2006.16239](https://arxiv.org/abs/2006.16239)
- History length: gains saturate at ~80 past accesses (Glider saturates ~30) — [arXiv 2006.16239 §5.4](https://arxiv.org/abs/2006.16239)
- Belady needs long lookahead: reaching 80% of Belady's performance on omnetpp requires reuse distances ~2,600 accesses into the future — [arXiv 2006.16239 Fig 2](https://arxiv.org/abs/2006.16239)
- Setup detail useful for a simulator: 64 sampled LLC sets, ~5M accesses/program; 80/10/10 train/val/test split in time — [arXiv 2006.16239 §5.1](https://arxiv.org/abs/2006.16239)
- Code + Gym environment: [google-research/cache_replacement](https://github.com/google-research/google-research/tree/master/cache_replacement)

**Voyager — Shi, Jain, Swersky, Hashemi, Ranganathan, Lin, "A Hierarchical Neural Model of Data Prefetching", ASPLOS 2021** **[peer-reviewed]**
- Splits addresses into page + offset with a mechanism to learn page/offset relations; learns address correlations, not only deltas; +41.6% IPC over no prefetcher on irregular SPEC06/GAP vs 21.7% (idealized Domino) and 28.2% (idealized ISB) — [UT Austin PDF](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf); [Google Research](https://research.google/pubs/pub50160/); code [Quangmire/voyager](https://github.com/Quangmire/voyager)

**TransFetch — Zhang et al., "Fine-grained address segmentation for attention-based variable-degree prefetching"** **[ACM Computing Frontiers 2022 — venue from memory]**
- Segments addresses into fine-grained pieces to reduce vocabulary; attention network; +38.75% IPC over no prefetching, +10.44% over BOP, +6.64% over Voyager — [NSF PAR](https://par.nsf.gov/biblio/10376312-fine-grained-address-segmentation-attention-based-variable-degree-prefetching)

**DART — Zhang, Gupta, Kannan, Prasanna, "Attention, Distillation, and Tabularization: Towards Practical Neural Network-Based Prefetching"**, arXiv:2401.06362 **[preprint per arXiv listing]** — see Q3 for numbers — [arXiv 2401.06362](https://arxiv.org/abs/2401.06362)

**LRB — Song, Berger, Li, Lloyd, "Learning Relaxed Belady for Content Distribution Network Caching", NSDI 2020** **[peer-reviewed]** (not a sequence model, but the closest analogue to a software KV cache)
- Imitates a *relaxed* Belady via a "Belady boundary"; introduces "good decision ratio" metric; GBDT-based; 4–25% WAN traffic reduction on 6 production CDN traces; prototype in Apache Traffic Server with modest overhead — [USENIX](https://www.usenix.org/conference/nsdi20/presentation/song); [PDF](https://www.usenix.org/system/files/nsdi20-paper-song.pdf)

**2024–2026 work (lower confidence: mostly from search snippets, not full reads)**
- LSTM-CRP: algorithm–hardware co-design of an LSTM cache replacement policy (Big Data & Cognitive Computing, 2024) — [doi 10.3390/bdcc8100140](https://doi.org/10.3390/bdcc8100140)
- GRUMA: GRU + multi-head attention for cache replacement (ACM conf. proceedings, 2025) — [ACM DL](https://dl.acm.org/doi/full/10.1145/3729706.3729815)
- T-CacheNet: transformer-based deep RL for internet content caching (ICNCC 2024) — [ACM DL](https://dl.acm.org/doi/10.1145/3711650.3711652)
- A "TBCR" transformer-encoder cache replacement method framing eviction as matching-style QA was mentioned in a search summary; I could not locate a primary source — treat as unverified.
- Review of ML in cache management (Frontiers in AI, 2025) — [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1441250/full)
- LLM-era caching: SAECache semantic-aware prefix-cache eviction, 1.4–2.7x TTFT improvement, fully online adaptive weighting — [arXiv 2605.18825](https://arxiv.org/abs/2605.18825) **[preprint]**; "When Fancy Eviction Fails: Rethinking Cache Replacement for LLM Prefix Reuse" argues recency dominates and frequency signals are ineffective for prefix reuse — [arXiv 2609.28870](https://arxiv.org/html/2609.28870) **[preprint, snippet only]**; CacheCraft uses LLM-guided *program evolution* to discover KV eviction scoring programs (the LLM writes the policy offline; the policy itself is cheap) — [arXiv 2608.14555](https://arxiv.org/pdf/2608.14555) **[preprint, snippet only]**; ArchAgent, agentic LLM-driven architecture discovery — [arXiv 2602.22425](https://arxiv.org/pdf/2602.22425) **[preprint, not read]**
- Learning-augmented caching theory with robustness guarantees (NeurIPS 2025) — [arXiv 2507.16242](https://arxiv.org/pdf/2507.16242)

### Inferences
- The consistent pattern across Glider, Parrot, LRB, and DART: the big sequence model is valuable as an **offline analysis/teacher tool**; the deployed policy is something small (ISVM, GBDT, tables).
- The 2025–2026 trend in LLM caching is "LLM designs the policy offline" (program evolution) rather than "LLM makes each eviction", which fits a hackathon story better than running ModernBERT per request.
- Parrot's ablations give a ready-made recipe (ranking loss + reuse-distance auxiliary head + DAgger) that transfers directly to a small from-scratch model.

### Gaps
- Twilight and other specific "LSTM/transformer prefetcher" names in the brief: I did not find/verify a paper named "Twilight" in this session.
- No full reads of 2024–2026 cache-transformer papers (GRUMA, T-CacheNet, TBCR); their results are unverified.
- No evidence found of contrastive pretraining on cache traces or of a "foundation model for access traces" released as a checkpoint.

---

## Q3. How were these models made fast enough? Are BERT-scale encoders appropriate per-request?

### Takeaway
None of the successful systems ran the big model on the critical path. Glider distilled an LSTM into an ISVM; DART distilled attention into table lookups (99.99% fewer ops); LRB used GBDT with batched background training; Parrot explicitly left latency as unsolved and suggested distillation/quantization/pruning, noting software caches tolerate more latency. A ModernBERT-class model costs milliseconds to hundreds of milliseconds on CPU per call — 3–5 orders of magnitude above a microsecond eviction budget — so it only fits as an offline teacher or a once-per-key feature extractor.

### Cited Findings
- Parrot authors: "deploying such learned policies requires solving practical challenges, e.g., model latency may overshadow gains due to better cache replacement"; full per-PC/address embeddings "can require tens of megabytes"; the byte embedder (few KB) still beats Glider by 8% normalized hit rate but loses to full Parrot — [arXiv 2006.16239 §1, §4.5, §5.2](https://arxiv.org/abs/2006.16239)
- Parrot future work: model-size reduction via distillation, pruning, quantization; "software caches ... tolerate higher latency"; decisions "can be made at any time between misses to the same set", giving a latency window "on the order of seconds for software caches" — [arXiv 2006.16239 §7](https://arxiv.org/abs/2006.16239)
- Glider: offline LSTM insight → online ISVM with "orders of magnitude lower cost" — [UT Austin PDF](https://www.cs.utexas.edu/~lin/papers/micro19c.pdf)
- DART: attention model → distilled model → "hierarchy of fast table lookups"; 99.99% fewer arithmetic ops vs large model, 91.83% vs distilled model; 170x speedup vs large, 9.4x vs distilled; F1 drop only 0.09; +33.1% IPC over TransFetch, +37.2% over Voyager; latency/storage comparable to rule-based BO with +6.1% IPC — [arXiv 2401.06362](https://arxiv.org/abs/2401.06362)
- LRB: the paper explicitly addresses "lightweight training and prediction" and memory overhead to be deployable on CDN servers — [USENIX NSDI 2020](https://www.usenix.org/conference/nsdi20/presentation/song)
- CPU latency reference points (weak sources): BERT-tiny with int8 quantization and dynamic input sizing has median ~10–20 ms per inference on commodity CPUs, inputs <500 tokens, unbatched — [HF community blog](https://huggingface.co/blog/tlogandesigns/using-ml-to-flag-fair-housing-violations). A search summary reported a ModernBERT-based model at ~108 ms median on an Intel i5-1235U and ~203 ms on a tablet via ONNX Runtime — [tech-insider.org](https://tech-insider.org/supersonic-labs-julia-1-cpu-decision-model-2026/) (low-quality source; not verified)
- ModernBERT's own speed claims are GPU-centric (RTX 3090/4090, A10, T4, L4) and relative to other encoders, not absolute microsecond latency — [HF blog](https://huggingface.co/blog/modernbert)

### Inferences
- Order-of-magnitude arithmetic (my estimate, not sourced): a ~100M non-embedding-param encoder does ~2×10^8 FLOPs per token, so a 64-token access history is ~10^10 FLOPs per decision, i.e., milliseconds on a GPU with kernel-launch overhead included and tens of ms+ on CPU. A 3-feature MLP is ~10^2 FLOPs; in Python its cost is dominated by interpreter overhead (single-digit microseconds). The gap is ~10^3–10^5x.
- For the simulator, "latency" is simulated, but wall-clock matters for replaying millions of requests: at 10 ms/decision, 1M evictions ≈ 2.8 hours. That alone rules out a per-eviction ModernBERT call in a hackathon.
- Viable patterns, ranked by fit:
  1. **Offline teacher → distilled student** (Glider/DART pattern): train a transformer (from scratch, small, or fine-tuned encoder) on Belady labels, then distill into the existing MLP or a lookup table. Demo story: "teacher closes X% of LRU→Belady gap, student keeps Y% of that at µs cost."
  2. **Amortize off the critical path**: score keys at insert/access time or in a background batch every N requests; eviction reads a cached score (priority queue). Parrot notes decisions can be made anytime between misses.
  3. **Per-key semantic embedding once** (only for text keys): encode the key string once with a small encoder (Ettin-17M, ModernBERT-base), cache a compressed feature (e.g., cluster ID or predicted reuse bucket), feed it to the MLP.
  4. **Sample-based eviction** (evict among k random candidates, as in LRB-style systems) reduces scoring to k items per eviction, making even a moderately sized model's batch cheaper.
- Quantization/ONNX export helps by ~2–3x-scale factors, not the 1000x needed; distillation/tabularization is the tool that actually closes the gap.

### Gaps
- No rigorous primary benchmark found for ModernBERT/mmBERT absolute CPU latency at short (≤128-token) inputs with batch size 1. The team should measure it themselves (one `timeit` over `AutoModel.from_pretrained("answerdotai/ModernBERT-base")` on their laptop) if they want to put a number in the pitch.
- Parrot's own inference latency was not reported numerically.

---

## Q4. How to tokenize/represent key streams

### Takeaway
Successful work never feeds raw key text to a language tokenizer: it uses learned embeddings of (hashed) IDs, address/byte segmentation, deltas, and per-key/per-PC history features. For a KV cache simulator with integer keys, hashed-ID embeddings plus per-key features (recency, frequency, inter-arrival stats) are the practical representation.

### Cited Findings
- Parrot: separate learned embedding per PC and per address (word2vec-like) — accurate but tens of MB; alternative byte embedder: embed each address byte with a shared 256×d table, concatenate, linear layer — few KB; captures hierarchy (upper bytes = region, lower bytes = object) — [arXiv 2006.16239 §4.5, Fig 4](https://arxiv.org/abs/2006.16239)
- Parrot generalizes to unseen addresses (21.6% of test addresses unseen in mcf; 5.3% in Web Search) — [arXiv 2006.16239 §5.2](https://arxiv.org/abs/2006.16239)
- Hashemi et al.: treat access deltas as tokens analogous to an n-gram/LM vocabulary — [arXiv 1803.02329](https://arxiv.org/abs/1803.02329)
- Voyager: hierarchical page/offset split rather than flat addresses, which lets it learn address correlations as well as deltas — [UT Austin PDF](https://www.cs.utexas.edu/~lin/papers/asplos21.pdf)
- TransFetch: fine-grained address segmentation to shrink vocabulary — [NSF PAR](https://par.nsf.gov/biblio/10376312-fine-grained-address-segmentation-attention-based-variable-degree-prefetching)
- Glider: unordered set of unique recent PCs as the history feature (derived from LSTM attention analysis) — [UT Austin PDF](https://www.cs.utexas.edu/~lin/papers/micro19c.pdf)

### Inferences
- For a Python KV simulator: map key → `hash(key) % B` bucket embedding (e.g., B=4096, d=16) plus numeric per-key features (time since last access, access count, mean/last inter-arrival gap, size, age), and a short sequence of the last H accesses to that key or globally. Deltas are meaningful only if keys are ordered (e.g., sequential block IDs); for random integer keys, deltas are noise.
- Byte-level embedding of the integer key (Parrot-style) is a cheap way to get generalization to unseen keys if key IDs have structure (e.g., tenant prefix, table ID in high bits).
- Contrastive pretraining on traces: no source found; not worth it in a hackathon.

### Gaps
- No verified source on contrastive/self-supervised pretraining over access traces.

---

## Q5. Fine-tuning recipes and training objectives

### Takeaway
Best-supported objectives are (a) binary Belady keep/evict ("cache-friendly vs averse", Hawkeye/Glider), (b) rank candidate lines by Belady reuse distance with a ranking loss plus an auxiliary log-reuse-distance regression head, trained on-policy with DAgger (Parrot). If a pretrained encoder is used at all, freeze it as a feature extractor with a small head; LoRA fine-tuning is possible but unjustified for integer keys.

### Cited Findings
- Hawkeye/Glider train binary classifiers from Belady labels (cache-friendly vs cache-averse) and rely on a heuristic among lines of the same class — [Parrot §1 describing them](https://arxiv.org/abs/2006.16239)
- Parrot objective: L = L_rank (differentiable NDCG with reuse distance as relevance, α=10) + L_reuse (MSE on log reuse distance per line); ranking loss is equivalent to KL to an exponentially smoothed Belady (a distillation view) — [arXiv 2006.16239 §4.3–4.4](https://arxiv.org/abs/2006.16239)
- DAgger matters: +9.8% avg normalized hit rate; highly program-dependent — [arXiv 2006.16239 §5.3](https://arxiv.org/abs/2006.16239)
- LRB labels relative to a "Belady boundary": objects whose next request is beyond the boundary are good eviction candidates (relaxed, easier to learn than exact Belady) — [USENIX NSDI 2020](https://www.usenix.org/conference/nsdi20/presentation/song)
- mmBERT/ModernBERT are released as HF `transformers` checkpoints (8k context), standard for feature-extraction or fine-tuning — [mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base); [HF ModernBERT blog](https://huggingface.co/blog/modernbert)

### Inferences
- Concrete student recipe for the hackathon (upgrading the 3-feature MLP):
  - Labels: replay trace with Belady offline; for each key at each access, compute next-reuse distance; label = bucket(log2 reuse distance) or binary "beyond Belady boundary / reused before eviction under OPT".
  - Model: MLP on ~8–12 per-key features (+ optional hashed-key embedding), trained with cross-entropy on buckets or pairwise ranking among sampled eviction candidates.
  - Optional teacher: a 1–2 layer transformer (from scratch) over last 32–80 accesses, trained with Parrot's ranking + reuse-distance loss; distill its scores into the MLP. Report both on the simulator.
  - Evaluate as "normalized hit rate" = (r − r_LRU)/(r_Belady − r_LRU), Parrot's metric — [arXiv 2006.16239 §5.1](https://arxiv.org/abs/2006.16239)
- Encoder route (only if keys are strings): freeze Ettin-17M or ModernBERT-base, mean-pool key-string embeddings at insert time, PCA to ~8–16 dims, append to MLP features. LoRA fine-tuning an encoder to predict reuse bucket from a key string is feasible offline but the team should first show frozen embeddings add signal (ablation) before spending time on it.
- Honest cost/benefit to present: a pretrained 150–400M encoder adds 3–5 orders of magnitude inference cost and brings no prior knowledge for integer keys; the literature's gains came from Belady-supervised objectives and history features, not model size or NL pretraining.

### Gaps
- No published LoRA fine-tuning of a text encoder for cache eviction found.
- No quantitative evidence found on how much semantic key embeddings improve hit rate on URL/path traces.
