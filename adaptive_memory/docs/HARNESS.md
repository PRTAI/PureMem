# Adaptive Memory Harness — 系统描述 (v8.6.2)

> 本文档描述当前生效的三阶段 memory retriever + LLM 精排 gate 系统。目标：在 PuzzleWorld 上跑得比裸基模 (`no_memory`) 更好。与之前版本的差异见 §7。相关：`ARCHITECTURE.md`（更侧重目录/数据流的旧总览）、`pureworld_v8.6.md`（历史里程碑）。

---

## 1. 系统定位

- **输入**：一个 PuzzleWorld 题（title + flavor_text + 多张图 + modality/skills tag）
- **输出**：多模态模型的解答字符串，交 GPTScorer / ExactScorer 打分
- **中间产物**：向模型 prompt 里注入 0 ~ `TOP_K` 条 memory precedent（**允许注入 0 条**）
- **优化目标**：test 集 solve rate 高于 baseline（相同模型、相同测试集、`--reasoner standard`）
- **约束**：
  - 严禁答案泄漏（`figure*.png` 与 `visual_content.summary` 只在 memory 侧使用，不进 query 侧）
  - 结果需可复现（rerun 位一致，rerank 分数持久化）
  - vendored `PuzzleWorld/` 只读，所有产物落到 `runs/<version>_<model>/`

---

## 2. 组件全景

```
┌─────────────────────────── 离线构建 ──────────────────────────┐
│  build_puzzle_memory.py                                     │
│    ├── create_split (stratified by modality×difficulty)     │
│    ├── SubsetSelector (random | stratified | diversity | …) │
│    ├── LLM 抽象 (TPL_ABSTRACT_PATTERN)  ← mechanism-preserving │
│    ├── _deduplicate_patterns (cosine ≥ 0.90 视为重复)         │
│    └── 落到 puzzle_memory/versions/<version_id>/             │
│         ├── memory.jsonl              (concrete chunks)      │
│         ├── embeddings.npy            (Layer 2)              │
│         ├── question_embeddings.npy   (Layer 1)              │
│         ├── abstract_memory.jsonl     (Layer 0 metadata)     │
│         ├── abstract_embeddings.npy   (Layer 0)              │
│         └── meta.json                 (build 参数快照)         │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────── 在线检索 ───────────────────────────┐
│  adaptive_memory/retriever.py    MemoryRetriever             │
│    └── retrieve_three_stage(...)  ← 生产路径                  │
│         ├── Stage 2 内容召回 (_gather_candidate_pool)          │
│         ├── Stage 1 tag 软加权                                 │
│         └── Stage 3 LLM approach-fit 精排 + tau + 多数投票 gate │
│  adaptive_memory/reranker.py     ApproachFitReranker          │
│    ├── prompt 带 anti-pattern rubric  ← 本轮新增                │
│    ├── 采样 N=5, temp=1.0                                     │
│    └── rerank_cache.jsonl (prompt_hash 变化自动失效)           │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────── 评测 harness ──────────────────────┐
│  PuzzleWorld/src/run.py                                     │
│    ├── select_model(args.model)         → modeling.EvalModel │
│    ├── _init_memory_retriever(version)  → MemoryRetriever    │
│    ├── select_reasoner("memory_augmented", model, retriever) │
│    └── for puzzle: evaluate → GPTScorer                     │
│                                                             │
│  PuzzleWorld/src/reasoner.py    MemoryAugmentedReasoner      │
│    ├── _resolve_query_visual_text (只取 content*.png)         │
│    ├── _embed_text(title + flavor + visual)                 │
│    ├── retriever.retrieve_three_stage(...)                  │
│    └── _build_memory_prompt → model.run(prompt, images)     │
│                                                             │
│  产物：runs/<version>_<model>/graded/test/                    │
│    ├── stepwise_test_<model>.csv           (每题得分)         │
│    ├── memory_injections_test_<model>.jsonl (每题注入追溯)     │
│    └── token_usage_<model>.json            (成本)             │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 三阶段检索 pipeline（当前生产路径）

`retriever.py::retrieve_three_stage` 是 `MemoryAugmentedReasoner` 走的唯一路径。

```
Query puzzle (title, flavor, images, modality, skills)
        │
        ▼
_resolve_query_visual_text(sample)
  · 只取 content*.png 的 description + key_elements
  · 严禁 figure*.png（答案图）与 visual_content.summary（因其覆盖了 figure）
        │
        ▼
embed_text = title + flavor + visual_text
question_emb = state_emb = SentenceTransformer(all-MiniLM-L6-v2).encode(embed_text)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 2  内容召回  (_gather_candidate_pool)                    │
│   • Abstract 池：Layer 0 命中的每个 abstract pattern           │
│     → 从其 source_demo 的所有 chunk 中挑与 state 最相似的一条    │
│   • Concrete 池：Layer 1 topN demos 的所有 chunk               │
│     → 按 state 相似度排序                                       │
│   → pool (≤ 2M 条)，每条带 raw cosine similarity                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 1  Tag 软加权（不是硬切！）                                │
│   bonus = 0.10 · |Qm ∩ Cm|   (modality overlap)               │
│         + 0.20 · |Qs ∩ Cs|   (skill overlap，更细粒度权重更高)   │
│   blended_score = similarity + bonus                          │
│   → 按 blended 排序，取 shortlist = top-M (M=5)                │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 3  LLM approach-fit 精排 + gate  (ApproachFitReranker)   │
│   • prompt = 题面 + M 条 candidate (abstract + concrete example)│
│   • judge = haiku, temp=1.0, N=5 samples per candidate         │
│   • rubric 含 4 条 anti-pattern 硬规则 (fit≤0.3 mechanism mismatch)│
│   • samples 写入 rerank_cache.jsonl（跨 run 位一致）             │
│   • 多数投票 gate：≥ 3/5 samples 通过 tau=0.7 → 该 candidate 通过│
│   • **全员未过 → 返回 []（degrade to no-memory）**              │
│   → survivors 按 (votes, mean_fit) 排序，截 top_k = 2           │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
List[chunk]  (长度 0 ~ 2)
```

**为什么每一阶段这样设计**：

| 阶段 | 关键决策 | 原因 |
|---|---|---|
| Stage 2 | 双池（abstract + concrete）合并 | 抽象命中弥补题面 wording 差异大的情况；具体命中兜底 |
| Stage 1 | 软加权而不是硬过滤 | 跨 modality/skill 但方法契合的先例不能被 tag 误杀，交给 Stage 3 说话 |
| Stage 3 | tau + 多数投票 + 允许空注入 | 判官在 temp=1.0 下会抖动；投票稳边界；宁空勿滥（misfit 会拖垮 solver）|

---

## 4. 数据表对齐不变式

三张表通过一个整数索引 `source_idx == source_demo_idx == question_embeddings.npy 行号` 对齐。破坏这个不变式会让整个 pipeline 静默出错。

```
concrete records (memory.jsonl 第 i 行)
   ├── source_idx = k                 ← puzzle-level 索引
   └── 对应 embeddings.npy 第 i 行

abstract records (abstract_memory.jsonl 第 j 行)
   ├── source_demo_idx = k            ← 必须等于某些 concrete records 的 source_idx
   └── 对应 abstract_embeddings.npy 第 j 行

question_embeddings.npy 第 k 行
   └── = subset[k] 这个 demo puzzle 的 title+flavor+visual 嵌入
```

已在跨 10 个 memory 版本上验证：**6455 条 abstract 全部对齐**、行数完全匹配。

---

## 5. 配置面 (`adaptive_memory/config.py`)

### 5.1 三阶段核心参数

| 参数 | 默认 | 何时改 |
|---|---|---|
| `TOP_N_DEMOS` | 8 | Layer 1 召回宽度：memory 库很大时可以调低省时间 |
| `TOP_K` | 2 | 最终注入的 precedent 上限：模型 context 紧张时降到 1 |
| `TOP_N_ABSTRACTS` | 2 | Layer 0 抽象命中：抽象层质量差时降到 0 关掉 |
| `STAGE1_MODALITY_WEIGHT` | 0.10 | modality 特别重要的题域可以调到 0.2~0.3 |
| `STAGE1_SKILL_WEIGHT` | 0.20 | 与上同理 |
| `RERANK_CANDIDATES_M` | 5 | Stage 3 shortlist 长度：M 越大 judge 成本越高 |
| `RERANK_FIT_TAU` | 0.7 | 通过阈值：想更严就调到 0.75~0.8 |
| `RERANK_SAMPLES_N` | 5 | 判官采样次数：判官已经很确定时可降到 3 |
| `RERANK_VOTE_THRESHOLD` | 3 | 通过所需票数：调到 4/5 更严，2/5 更宽 |
| `RERANK_TEMPERATURE` | 1.0 | 判官温度：0 会让 N 次采样退化为 1 次 |

### 5.2 Prompt 模板

| 模板 | 作用 | 何时改 |
|---|---|---|
| `TPL_ABSTRACT_PATTERN` | 离线：concrete step → abstract pattern（本轮改）| 想让抽象保留更多/更少 mechanism 细节 |
| `TPL_EXTRACT_TRIPLES` | 离线：CoT → triples（gsm8k 时代产物，PuzzleWorld 不用）| — |
| `TPL_INITIAL_STATE` / `TPL_STEP_REASONING` / `TPL_SINGLE_CALL` | gsm8k 时代 online step 循环（PuzzleWorld 不用）| — |
| `TPL_EXTRACT_ABSTRACTS` / `TPL_ABSTRACT_RETRIEVAL_CONTEXT` / `TPL_EVOLVE_MEMORY` | v8.6 早期抽象化 + evolve 模板 | evolve 闭环接线时启用 |
| `_RERANK_PROMPT` (`reranker.py`) | Stage 3 judge rubric（本轮改）| 发现新 misfit 类别时补 anti-pattern 规则 |

---

## 6. 使用方式

### 6.1 一次完整 A/B 流程

```bash
cd D:\zhibin.pu\pureworld

# ─── 1. 划分 train/test（seed=42，仅需一次）──────────────
python build_puzzle_memory.py --split

# ─── 2. 构建一版 memory bank ─────────────────────────────
# 输出到 adaptive_memory/puzzle_memory/versions/v_stratified_20_<hash>/
python build_puzzle_memory.py --build --selector stratified --subset-size 20

# ─── 3. baseline 跑 test 集 ──────────────────────────────
python PuzzleWorld/src/run.py --model claude-opus-4-7 \
    --folder test --folder_path runs/_test_puzzles \
    --reasoner standard \
    --attempt_puzzles --score_puzzles \
    --output_dir runs/no_memory_claude_opus_4_7

# ─── 4. memory 版本跑 test 集（同模型、同测试集）─────────
python PuzzleWorld/src/run.py --model claude-opus-4-7 \
    --folder test --folder_path runs/_test_puzzles \
    --reasoner memory_augmented --memory_version v_stratified_20_<hash> \
    --attempt_puzzles --score_puzzles \
    --output_dir runs/v_stratified_20_<hash>_claude_opus_4_7

# ─── 5. 比较 ────────────────────────────────────────────
# 打开两个 runs/*/graded/test/stepwise_test_<model>.csv
# 对比 score_ratio 均值 + per-puzzle 差异
```

### 6.2 关键 env / flag

| 变量 | 效果 |
|---|---|
| `ADAPTIVE_RERANK_FRESH=1` | 忽略 rerank_cache，重新采样（改判官逻辑但不想改 prompt 时用）|
| `--skip_existing` | 复用已有输出，但**空输出会重跑**（避免 proxy 超时锁死 0.0） |
| `--limit N` | 只跑前 N 题（冒烟测试）|

---

## 7. 本轮 (v8.6.2) 变更清单

只有两处代码改动 + 一个自动效应，其余组件不动。

### ① `TPL_ABSTRACT_PATTERN`（`build_puzzle_memory.py`）
- **变化**：从"remove all specific content"改为"保留 mechanism 细节，只去掉 NAMES"
- **必须逐字保留**：spatial reading direction、cardinality、concrete verb、object type
- **附加**：Good/Good/Bad 三个 example（Bad = "50% Off" misfit 的抽象塌陷方式）
- **前置条件**：**只有重建 memory bank 才生效**（`build_puzzle_memory.py --build …`）
- **失败模式覆盖**：
  - `50% Off`（column-read acrostic）与 `Saw That Coming`（half-word recombining）之前都被抽象成 "collect letters → concatenate → answer" 这个空壳
  - 新 prompt 下会分别抽象成 "read the middle column top-to-bottom" 和 "recombine top half of one word with bottom half of its pair"，判官能分辨

### ② `_RERANK_PROMPT`（`adaptive_memory/reranker.py`）
- 明说 candidate 会同时给 abstract + concrete example，**冲突时以具体为准**
- 4 条 `fit ≤ 0.3` 硬规则：abstract-vs-concrete mismatch / 错方向 / 错基数 / 错 op-family
- **宁空勿滥**：`loosely related (0.4)` vs `would mislead (0.0)` 二选一时选 0.0
- **前置条件**：无需重建 memory，rerun 即生效

### ③ Prompt hash 自动失效缓存
- `_PROMPT_HASH` 从 `353edee5de8e7e1f` → `67397bcae88ea673`
- 因为 `_cache_key` 把 hash 拼进 key，**旧 rerank_cache.jsonl 里的样本下次 rerun 会自动重采样**
- 无需手动删除 cache 或设 `ADAPTIVE_RERANK_FRESH=1`

---

## 8. 与前代对比

| 维度 | v8.5 baseline | v8.6 早期 | **v8.6.2（当前）** |
|---|---|---|---|
| Memory 内容 | Concrete triples 单一 | + abstract patterns | 同左，abstract 层信息量大幅增加 |
| 召回架构 | 2 层：question → state | 3 层：abstract → question → state | **3 stage**：内容召回 → tag 加权 → LLM 精排 |
| Tag 使用 | 无 | 无 | modality/skill 软加权（0.10/0.20）|
| 注入决策 | Top-K 硬塞 | Top-K 硬塞 | **tau + 多数投票**；全员未过则空注入 |
| Judge 精排 | 无 | 无 | LLM approach-fit + 采样投票 + 缓存 |
| 抽象 prompt | N/A | 过度泛化，塌成同一 shell | **保留 mechanism 细节** |
| Rerank rubric | N/A | N/A | 4 条 anti-pattern 硬规则 |
| 防答案泄漏 | 无区分 | content*.png only | 同左 |
| 注入追溯 | 无 | 无 | `memory_injections_..jsonl` 全量记录 |
| 可复现性 | 强 | 强 | rerank_cache.jsonl + prompt hash 位一致 |
| 版本管理骨架 | 单库 | `evolution_workspace.py` + `v_best` | 同左（**闭环未接**）|
| 失败自进化 | 无 | prompt 存在但未接线 | 同左（第一轮不做）|
| 产物落点 | `PuzzleWorld/outputs/` 混杂 | 同 | 统一 `runs/<version>_<model>/` |

---

## 9. 观测与调试

### 9.1 每题追溯
`runs/<version>_<model>/graded/test/memory_injections_test_<model>.jsonl`（一题一行）：

```json
{
  "title": "50% Off",
  "num_precedents": 0,
  "precedents": [],
  "injected_block": "",
  "gated": true,
  "gate_config": {"tau": 0.7, "n_samples": 5, "vote_threshold": 3, "temperature": 1.0}
}
```
或

```json
{
  "title": "…",
  "num_precedents": 2,
  "precedents": [
    {"idx": 1, "layer": "abstract", "pattern_type": "…", "source_demo_idx": 0,
     "source_puzzle": "…", "similarity": 0.42, "fit_mean": 0.75, "fit_std": 0.08,
     "fit_votes": 5, "fit_n": 5, "fit_tau": 0.7, "fit_samples": [0.7, 0.8, 0.7, 0.8, 0.7]},
    …
  ],
  "injected_block": "Reasoning Pattern 1 (…): …",
  "gated": false
}
```

### 9.2 该问的问题

- 每题的 **`fit_votes` 分布** — 边界（0<votes<5）比例高说明 judge 抖动大
- **`gated: true` 的题数** — 太高说明 rubric 过严或 memory 覆盖不足；太低（< 5%）说明 gate 形同虚设
- **对比 baseline 与 memory 版本的 per-puzzle 得分差** — 找出被 memory 拉高/拖低的题，逐个看 `injected_block` 判断增益/损失来源
- **抽象层的实际长相** — 打开 `versions/<v>/abstract_memory.jsonl` 抽样若干条，验证 mechanism 细节确实被保留了

### 9.3 常见坑

| 症状 | 原因 | 排查 |
|---|---|---|
| 某题永远 0.0 | 空输出被 `--skip_existing` 锁死 | 该逻辑已修（空输出会重跑），若仍复发看 `evaluate()::has_valid_cache` |
| 抽象层出现 `× → ` 乱码 | Windows GBK 编码写入 | retriever 读入有兜底，但源头应从 UTF-8 写出 |
| Rerank 结果跨 run 不一致 | 缓存被删或 `ADAPTIVE_RERANK_FRESH=1` | 保留缓存文件即可 |
| Memory 版本"错位"（abstract 与 concrete 不对应）| memory bank 被中途重建过 | log 里的 memory 是历史版本，用 `--memory_version` 显式绑定 |

---

## 10. 已知限制 & 下一步

1. **自进化闭环未接**：`evolution_workspace.py` 有骨架、`TPL_EVOLVE_MEMORY` 有 prompt，但没有 caller 把 `runs/*/graded/*` 里的失败题喂给 evolve。第一轮验证结束、确认 (1)+(2) 起效后再决定是否接。
2. **投票 vs 单采样的价值未量化**：目前只在 50 条候选上有数据（40% 边界候选、12% 与 mean-gate 分歧）。第一轮跑完后应做 A/B 消融：相同 memory + 相同题，单采样 vs 均值 vs 投票，看 solve rate 差异是否值那 5× 判官调用。
3. **抽象 dedup 阈值 0.90 未调过**：过高会留下近似重复；过低会砍掉细粒度差异。改抽象 prompt 后（细节更多），阈值可能需要重新校准。
4. **subset selector 的 diversity 未验证**：目前只在 stratified 上跑过实验。
5. **evolve 未接的隐含风险**：如果 abstract 抽取本身有 bug，evolve 只会把坏 pattern 更多地灌进去——这也是 § 3 的 "abstracter 是否可靠" 必须先于 evolve 验证的原因。

---

## 附录 A：核心不变式检查（新版本 build 后建议跑一次）

```python
# 验证 source_idx / source_demo_idx / embedding 行号对齐
import json, os, numpy as np
from collections import defaultdict

vdir = 'adaptive_memory/puzzle_memory/versions/v_stratified_20_<hash>'
records = [json.loads(l) for l in open(f'{vdir}/memory.jsonl', encoding='utf-8')]
abstracts = [json.loads(l) for l in open(f'{vdir}/abstract_memory.jsonl', encoding='utf-8')]
demo_to_chunks = defaultdict(list)
for i, r in enumerate(records):
    demo_to_chunks[r['source_idx']].append(i)

bad = 0
for ap in abstracts:
    sdi = ap['source_demo_idx']
    for ci in demo_to_chunks.get(sdi, []):
        if records[ci]['source_idx'] != sdi:
            bad += 1

assert bad == 0, f'{bad} abstract-concrete alignment errors'
assert np.load(f'{vdir}/embeddings.npy').shape[0] == len(records)
assert np.load(f'{vdir}/abstract_embeddings.npy').shape[0] == len(abstracts)
print('OK')
```
