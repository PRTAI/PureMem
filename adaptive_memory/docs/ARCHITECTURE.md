# 三阶段 Memory Retriever × PuzzleWorld 实验框架 —— 架构说明

> 目标：验证「三阶段 memory retriever + 精排 gate」这套 harness 相比裸基模（`no_memory`）能在 PuzzleWorld 上取得更高的 solve rate。本文只描述**架构与数据流**，不涉及具体实验数据。

---

## 1. 顶层目录

```
pureworld/
├── config.py                       # 全局 API/路径配置（唯一真源）
├── build_puzzle_memory.py          # 离线：split → 抽样 → 抽象 → 建库
├── subset_selector.py              # 抽样策略（random/stratified/diversity/curriculum）
├── evolution_workspace.py          # memory 版本落盘/召回工具（v_best、materialize_version）
├── quick_validate_memory.py        # 快速冒烟：检验一份 memory 是否可用
├── analyze_retrieval_categories.py # 离线分析：按 category 命中率评估召回
│
├── adaptive_memory/                # ★ Retriever 组件（本文重点）
│   ├── config.py                   # 三阶段超参 + prompt 模板
│   ├── api_client.py               # OpenAI 兼容 GenClient（带重试）
│   ├── retriever.py                # 双层 / 三层 / 三阶段 三种召回接口
│   ├── reranker.py                 # Stage-3 approach-fit LLM 精排
│   ├── puzzle_memory/versions/     # 每个抽样策略产出一版：v_stratified_20_… 等
│   │   └── <version_id>/
│   │       ├── memory.jsonl               # 具体 chunks（LLM 抽象后的 state→action→next_state）
│   │       ├── embeddings.npy             # chunk 状态嵌入（Layer 2）
│   │       ├── question_embeddings.npy    # 每个 demo 题目嵌入（Layer 1）
│   │       ├── abstract_memory.jsonl      # 抽象模式（Layer 0）
│   │       ├── abstract_embeddings.npy    # 抽象嵌入（Layer 0）
│   │       └── rerank_cache.jsonl         # Stage-3 分数缓存（保证可复现）
│   └── splits/                     # 分层划分 train/dev/test.jsonl + hf_match_table.json
│
├── PuzzleWorld/                    # vendored 上游（只读）
│   ├── data/{ref_puzzles,hf_full}  # 题面 + 图片 + visual_content.json
│   └── src/
│       ├── run.py                  # ★ 评测入口：解析 CLI，实例化 model+reasoner，逐题跑
│       ├── reasoner.py             # StandardReasoner / MemoryAugmentedReasoner
│       ├── modeling.py             # 多模态模型封装（OpenAI-兼容图文调用）
│       ├── data_loading.py         # Sample / Data，负责 metadata 与图片解析
│       ├── judge_utils.py          # LLM 评分器辅助
│       └── scorers.py              # ExactScorer / GPTScorer
│
└── runs/                           # ★ 唯一产物根：<version>_<model>/
    ├── no_memory_claude_opus_4_7/                             # 基模基线
    ├── v_stratified_20_cabb04_claude_opus_4_7/
    └── v_stratified_20_cabb04_claude_haiku_4_5_20251001/
        └── graded/test/
            ├── stepwise_test_<model>.csv
            ├── memory_injections_test_<model>.jsonl           # 每题注入的 precedent 全量追溯
            └── token_usage_<model>.json
```

设计约束：**vendored 的 `PuzzleWorld/` 从不被写入**，所有跑出来的东西都落到 `runs/<version>_<model>/`，方便重跑与对比。

---

## 2. Memory 构建管道 (`build_puzzle_memory.py`)

一次性离线流程。**输入 = train 集题面 + 官方解题过程**；**输出 = 一份带版本号的 memory bank**。

```
raw puzzles (metadata.json)
        │
        ▼
 create_split(seed, ratio)         # 按 modality×difficulty 分层
        │
        ├─→ splits/train.jsonl
        └─→ splits/test.jsonl
                    │
                    ▼
 SubsetSelector.select(...)        # random | stratified | diversity | curriculum
                    │
                    ▼
 逐题拆 reasoning steps → triples (state, action, next_state)
                    │
                    ▼
 LLM 抽象 (TPL_ABSTRACT_PATTERN)   # 去掉数字/实体，抽出 pattern_type
                    │
                    ▼
 _deduplicate_patterns             # 按 (pattern_type, 内容) 近似去重
                    │
                    ▼
 sentence-transformers 编码
        ├─→ embeddings.npy              (每 chunk state 一行)
        ├─→ question_embeddings.npy     (每 demo puzzle 一行，含 visual_content 摘要)
        ├─→ abstract_memory.jsonl       (=records 抽象层拷贝，带 source_demo_idx)
        └─→ abstract_embeddings.npy
                    │
                    ▼
 落到 puzzle_memory/versions/<version_id>/
 并同步到 puzzle_memory/current/ 与 v_best 软链
```

关键不变式：
- `source_idx`（chunk 上）== `source_demo_idx`（abstract 上）== `question_embeddings.npy` 的行号 → 三张表通过它对齐。
- `memory.jsonl` **必须 UTF-8**。Windows 默认 GBK 会把 `×` / `→` 写坏；retriever 读入时做了 UTF-8 → GBK → drop 的兜底解码，但仍应从源头修好。

---

## 3. 三阶段检索 (`adaptive_memory/retriever.py` + `reranker.py`)

这是本 harness 相对 baseline 的核心增益点。`MemoryRetriever` 暴露三种召回接口，实验里默认走最强的 `retrieve_three_stage`。

### 3.1 三种接口对比

| 接口 | 层级 | 说明 |
|---|---|---|
| `retrieve()` | Layer 1 + 2 | 双层：题目相似 → 状态相似的 top-K chunk |
| `retrieve_with_abstracts()` | Layer 0 + 1 + 2 | 三层：抽象模式命中 + 双层，硬分预算 |
| **`retrieve_three_stage()`** | 三阶段 | ★ 生产路径：tag 软加权 + 内容召回 + LLM 精排 + tau 门控 |

### 3.2 三阶段管线（`retrieve_three_stage`）

```
                Query puzzle (title, flavor, images)
                          │
                          ▼
  _resolve_query_visual_text(sample)  ← 只用 content*.png，figure*.png 禁入（防答案泄漏）
                          │
                          ▼
  embed(title + flavor + visual_text)   → question_emb == state_emb
                          │
                          ▼
 ┌────────────────────────────────────────────────────────────┐
 │  Stage 2  内容召回  _gather_candidate_pool                  │
 │    • abstract 池: 每个命中 pattern 里取「与 state 最相似」的一条   │
 │    • concrete 池: top-N demos 的所有 chunk，按 state 相似度     │
 │    → pool (≤ 2M 条)，每条带 raw cosine similarity              │
 └────────────────────────────────────────────────────────────┘
                          │
                          ▼
 ┌────────────────────────────────────────────────────────────┐
 │  Stage 1  Tag 软加权（不硬切！）                              │
 │    bonus = STAGE1_MODALITY_WEIGHT · |Qm ∩ Cm|                │
 │          + STAGE1_SKILL_WEIGHT    · |Qs ∩ Cs|                │
 │    blended = similarity + bonus                              │
 │    → 按 blended 排序，取 shortlist = top-M (M=5)              │
 └────────────────────────────────────────────────────────────┘
                          │
                          ▼
 ┌────────────────────────────────────────────────────────────┐
 │  Stage 3  LLM Approach-Fit 精排 (ApproachFitReranker)         │
 │    • prompt = 题面 + 每条 candidate 的 (state→action→next)     │
 │    • judge (haiku, temp=1.0) 采样 N=5 次 → fit ∈ [0,1]        │
 │    • 缓存 samples 到 rerank_cache.jsonl（跨 run 可复现）        │
 │    • 多数投票门控：≥ vote_threshold(=3) 个样本 clear tau(=0.7) 才通过 │
 │    • **全员未过 → 返回 []（degrade to no-memory）**            │
 │  → survivors 按 votes 再按 mean fit 排，截 top_k_chunks (=2)   │
 └────────────────────────────────────────────────────────────┘
                          │
                          ▼
              List[chunk]（可能为空）
```

**关键设计决策**：

1. **软 vs 硬**：Stage-1 用加权而不是硬过滤，避免把「跨模态但方法契合」的先例误杀，由 Stage-3 说了算。
2. **不注入 misfit**：Stage-3 的 tau gate 允许「宁空勿滥」——已经验证过一条无关先例足以把某些题从 solve 拖到 0.0，所以门槛设得高。
3. **可复现的随机 judge**：judge 用非零温度多次采样以覆盖决策边界抖动，但样本会持久化到 `rerank_cache.jsonl`，key 只依赖 `(prompt_hash, query_sig, candidate_sig)`，与 tau/threshold 无关——所以 rerun 完全一致。要强制重采样：`ADAPTIVE_RERANK_FRESH=1`。
4. **防答案泄漏**：query 端只喂 `content*.png` 的描述；`figure*.png`（答案图）和 `visual_content.summary`（覆盖了图书页）都禁入。Memory 端可以带完整解，因为那才是它该提供的信息。

### 3.3 相关超参（`adaptive_memory/config.py`）

| 参数 | 值 | 含义 |
|---|---|---|
| `TOP_N_DEMOS` | 8 | Layer 1 题目相似召回宽度 |
| `TOP_K` | 2 | 最终注入的 precedent 上限 |
| `TOP_N_ABSTRACTS` | 2 | Layer 0 抽象模式匹配数 |
| `STAGE1_MODALITY_WEIGHT` | 0.10 | 每个 overlap modality tag 的加分 |
| `STAGE1_SKILL_WEIGHT` | 0.20 | 每个 overlap skill tag 的加分（更细粒度 → 权重更高）|
| `RERANK_CANDIDATES_M` | 5 | Stage-2 → Stage-3 的 shortlist 长度 |
| `RERANK_FIT_TAU` | 0.7 | Stage-3 单样本通过阈 |
| `RERANK_SAMPLES_N` | 5 | judge 采样次数 |
| `RERANK_VOTE_THRESHOLD` | 3 | 多数投票门槛（≥3/5 过 tau） |
| `RERANK_TEMPERATURE` | 1.0 | judge 温度（0 会让 N 次采样退化成 1 次） |

---

## 4. 评测 harness（`PuzzleWorld/src/run.py` → `reasoner.py`）

### 4.1 CLI 契约

```bash
# baseline
python PuzzleWorld/src/run.py --model claude-opus-4-7 \
    --folder test --folder_path runs/_test_puzzles \
    --reasoner standard --attempt_puzzles --score_puzzles \
    --output_dir runs/no_memory_claude_opus_4_7

# memory 版本
python PuzzleWorld/src/run.py --model claude-opus-4-7 \
    --folder test --folder_path runs/_test_puzzles \
    --reasoner memory_augmented --memory_version v_stratified_20_cabb04 \
    --attempt_puzzles --score_puzzles \
    --output_dir runs/v_stratified_20_cabb04_claude_opus_4_7
```

`--folder` 只用作**输出标签**；真实题面路径走 `--folder_path`，让生成的测试目录待在 vendored 仓库外。

### 4.2 单题控制流

```
run.py::main()
  ├─ select_model(args.model)                          # modeling.EvalModel
  ├─ _init_memory_retriever(memory_version)            # 加载 memory bank
  │     └─ MemoryRetriever(memory_file, embs, q_embs)
  │        · 顺带加载 abstract_memory / abstract_embeddings
  └─ select_reasoner("memory_augmented", model, retriever)
        └─ MemoryAugmentedReasoner
             · memory_log_path  = graded/memory_injections_...jsonl  ← 每题追溯
             · rerank_cache_path = versions/<v>/rerank_cache.jsonl   ← 可复现

for puzzle in folders:
    evaluate(...)
       └─ reasoner.run(sample, puzzle_content)
            ├─ _resolve_query_visual_text(sample)    # 只取 content*.png
            ├─ _embed_text(title + flavor + visual)  # sentence-transformers
            ├─ retriever.retrieve_three_stage(       # ← 三阶段
            │      ..., reranker=self._get_reranker(),
            │      title=sample.title, flavor=sample.flavor_text,
            │      query_modality=sample.modality, query_skills=sample.skills,
            │  ) → retrieved_chunks (可能为空)
            ├─ _build_memory_prompt(...)             # 拼 prompt + 写 log
            └─ model.run(prompt, puzzle_content)     # 多模态图文调用
       │
       ├─ ExactScorer / GPTScorer 打分
       ├─ 写 stepwise_test_<model>.csv (逐题一行)
       └─ 累计 token_usage_<model>.json
```

### 4.3 缓存与失败处理

- `--skip_existing` 只在**旧输出非空**时才复用，否则重跑（否则一次 proxy 超时留下的空串会永久锁死 0.0，参见 memory 里的 PuzzleWorld eval pitfalls）。
- 前 3 题若都异常，`run.py` 主动 abort：防止 setup 级 bug 隐没在几百个「skipped」里。
- 单题异常只 skip 该题；`memory_injections_...jsonl` 忠实记录每题实际注入了什么（包括 `gated: true` 的空注入案例）。

---

## 5. 实验对照结构

| 维度 | 变体 | 落点 |
|---|---|---|
| **模型** | `claude-opus-4-7`, `claude-haiku-4-5-20251001` | `runs/*_<model>/` |
| **Reasoner** | `standard` (baseline) vs `memory_augmented` | 目录名前缀 `no_memory_` vs `v_<version>_` |
| **Memory 抽样策略** | random / stratified / diversity × {10, 20, 30} | `puzzle_memory/versions/v_<method>_<size>_<hash>/` |
| **测试集** | 由 `create_split()` 分层出 `splits/test.jsonl` | `runs/_test_puzzles/` 薄壳 metadata |

对比范式：**相同测试集 + 相同模型 + 不同 reasoner**（`standard` vs `memory_augmented`）— 单变量只有「有没有 memory harness」。

每次跑完可对比：
1. **正确率** — `stepwise_...csv` 的 `correct` / `score_ratio` 列
2. **注入质量** — `memory_injections_...jsonl` 里的 `fit_votes`、`gated` 分布
3. **成本** — `token_usage_...json`

---

## 6. 复现要点 Checklist

- [ ] `config.py::API_CONFIG` 的 `base_url` / `api_key` 有效
- [ ] `python build_puzzle_memory.py --split`（一次即可，seed=42）
- [ ] `python build_puzzle_memory.py --build --selector stratified --subset-size 20` 产出 `versions/v_stratified_20_…/`
- [ ] baseline：`--reasoner standard`，输出到 `runs/no_memory_<model>/`
- [ ] memory：`--reasoner memory_augmented --memory_version v_stratified_20_…`，输出到 `runs/v_…_<model>/`
- [ ] 对比 `stepwise_...csv` 的 `score_ratio` 均值；查 `memory_injections_...jsonl` 判断增益来源

关键复现前提：
- Memory bank 文件必须 UTF-8（否则触发 GBK 兜底并 log warning）
- `rerank_cache.jsonl` 已存在时 rerun 是位一致的；要 A/B 精排逻辑变化时 `ADAPTIVE_RERANK_FRESH=1`
- solve-path 图片渲染分辨率应 ≥ 1536（低分辨率会让部分题永久 0.0）
