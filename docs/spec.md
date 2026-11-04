# 计算生物学文献"水分"检测：红蓝对抗架构实现规格 v1.2

> 本文档交给实现 agent 使用。目标是搭出完整可运行的系统，不是写论文。
> 设计参考：Anthropic Alignment Science Blog《Automated Subtle Emergent Misalignment Research》（2026-10）的对抗搜索框架，以及《Automated Researchers Can Mitigate Well-Characterized Alignment Failures》（2026-08）的 harness 设计（隔离、冻结预注册、代码审查、共享日志）。

## 修订记录

**v1.2（2026-10-31）**：新增 R9，移除所有依赖人工实验/人工确认的验收与决策点。

**v1.1（2026-10-31）**，相对 v1.0 的改动及原因：

| # | 改动 | 原因 |
|---|---|---|
| R1 | `inflation` 改为**同量纲下的绝对超额效应**：`tanh(Δ / s)`，Δ 为同对比类型（二分类对二分类、连续对连续）下的 log HR 差值 | v1.0 用 Δ 除以报告效应。独立队列效应 ≤ 0 时，任何假阳性都饱和到 1（甚至 > 1），红队只需挑弱效应噪音基因就能拿满分；同时对齐二分类与连续变量量纲，避免中位数二分天然比每 SD 高约 1.6 倍造成的假膨胀 |
| R2 | 独立队列证据改为**随机效应荟萃**（合并估计 + τ² + 预测区间），取消"CI 是否含 0"的投票计数 | 投票计数会把小队列中真实但不显著的效应算成失败，系统性低估稳健结论 |
| R3 | 多重宇宙**参照集**排除数据驱动的选择（最优切点、按结局挑子集） | 这些选择本身就是 cherry-picking 手段，放进参照集会污染基线中位数。它们仍保留在注册表中供红队使用 |
| R4 | 红队**不再控制** `context_metadata`；M6 主循环固定使用中性基准情境（`context_coverage = 0`，仅按 `evidence_score` 判标记），`context_prior` 只在天然标签上训练，真实情境抽样仅用于 §7.3 评估 | 若允许红队填写情境，红队会一律拉满并把蓝队 `context_prior` 权重训成 0；若 M6 主循环随机抽样情境，又会因抽中高危情境给批次一票否决引入彩票噪声 |
| R5 | M3 增加 `as_of_date`，所有特征只用该日期之前的数据；历史复现率改为**留一法**；明确特征稀疏时的回退逻辑 | 防止未来引用、未来复现结果泄漏到历史样本；防止当前结论的标签泄漏进自己的特征 |
| R6 | MVP 收窄为**单基因**；多基因 signature 移到第一扩展项 | signature 复现依赖原文未写明的标准化方式，复现失败率高；红队的过拟合空间也是另一个维度 |
| R7 | 新增 P0 交付物**标准化队列池**（每癌种 5–10 个），原始复现与数据对账只针对池内结论 | 避免 agent 把大部分时间花在解析长尾 GEO 临床表型上 |
| R8 | 小修：批次拒绝规则去冗余；修正交叉引用 | v1.0 中"> 10%"和"> 2 份"在 K=20 时等价 |
| R9 | 移除人工实验/人工确认：M1 验收改为“合成夹具回归 + 公开 PMC 自动一致性审计”，P0 队列清单改为确定性自动选择，§10 改为配置项 | 设计目标要求全流程可自动复现；人工抽检、人工确认和人工加入队列池不可扩展、不可重复，且会阻塞无人干预运行 |

---

## 0. 背景与目标

### 0.1 要解决的问题
计算生物学工作大量依赖文献：基因-疾病关联、预后标志物、通路/基因集先验、知识图谱边、训练标签。文献中存在 cherry-picking（挑队列、挑阈值、挑分析流程、剔样本、只报有利结果），会把夸大的结论带进下游计算。

我们**没有湿实验能力**，所以所有"水分"的测量只能来自**已有公开数据的重分析**。

### 0.2 系统要产出什么
1. **真值引擎（Ground-Truth Engine）**：对一条文献结论，用公开数据计算它的"稳健性"和"膨胀度"。
2. **检测器（Detector，对应原文的 monitor）**：只看论文内容 + 情境/社会关系特征（不重跑数据），预测一条结论的膨胀度。用于无法重分析的大规模文献。
3. **红队 harness**：并行 agent 用真实公开数据、只用"有文献先例的分析选择"生成夸大结论的分析报告，目标是骗过检测器。其高分产出 = 检测器盲区。
4. **蓝队更新循环**：用红队发现的盲区加固检测器，红蓝交替。
5. **下游降权接口**：把检测器/真值引擎输出转成结论（或知识图谱边）的权重。

### 0.3 核心思路（一句话）
"一个结论有多少水分"在 comp bio 里可以被计算出来（独立队列荟萃 + 多重宇宙分析 + 原始数据对账），所以可以把它变成一个**可验证的分数**，在这个分数上跑原文那种对抗搜索。

---

## 1. 硬约束（实现时不得违反）

| 编号 | 约束 |
|---|---|
| C1 | 只使用公开、许可允许的数据。TCGA 等只用 open-access 层，不碰 controlled-access。遵守各数据源的使用条款。 |
| C2 | 红队产出（夸大的分析报告）**只在内部使用**：带水印/标记，存放在隔离目录，不得发布、投稿、上传到任何公开位置。 |
| C3 | 系统输出落在**结论层面**（claim/edge 的权重、稳健性分数），不输出"某人/某实验室造假"之类的个人判断。社会关系特征用于证据加权与检测，不用于对人定性。对外展示（demo、汇报）时，真实论文结论一律匿名化。 |
| C4 | held-out 队列对红队与检测器训练过程**物理隔离**（不同系统用户 + 文件权限），只有评分服务可读。 |
| C5 | 红队只能使用选择注册表（见 M2）中有文献先例的分析选择，不能编造数据或数字。所有报告中的数字必须能由其提交的代码在声明的数据上复现。 |
| C6 | 所有评分可复现：固定随机种子、缓存中间结果、记录数据版本/下载时间。 |
| C7 | （v1.1 新增）红队不能设置或修改任何情境/社会特征。主对抗循环固定使用中性情境（`context_coverage = 0`），情境置换与抽样仅在评分服务的 §7.3 专项评测中使用。 |

---

## 2. 总体架构

```
                    ┌──────────────────────────────────────────┐
                    │          文献 (PMC OA / PubMed)           │
                    └───────────────┬──────────────────────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ M1 结论抽取           │  → Claim JSON
                         └──────────┬───────────┘
                 ┌──────────────────┼────────────────────┐
                 │                  │                    │
      ┌──────────▼─────────┐ ┌──────▼────────────┐ ┌─────▼──────────────┐
      │ M2 真值引擎         │ │ M3 情境/社会图谱   │ │ M4 检测器 (monitor) │
      │ 标准队列池/荟萃/    │ │ as_of_date 截断    │ │ evidence_score      │
      │ 多重宇宙/数据对账    │ │ 留一法/稀疏回退     │ │ + context_prior     │
      └──────────┬─────────┘ └──────┬────────────┘ └─────┬──────────────┘
                 │  天然标签          │ 特征                 │
                 └───────────┬───────┴──────────────────────┘
                             │
         ┌───────────────────▼────────────────────────────────┐
         │ M6 评分服务 (Grader)                                 │
         │  ① 检测器筛查(仅evidence)  ② 真值(可见评测队列)       │
         │  ③ 真实性门槛  ④ 作弊检测  ⑤ held-out 仅内部报告      │
         └───────────▲────────────────────────┬───────────────┘
                     │ 提交（只含内容与分析）    │ 分数
         ┌───────────┴────────────┐   ┌───────▼──────────────┐
         │ M5 红队 harness          │   │ 共享实验日志 (JSONL)   │
         │ N 个并行 worker, 多轮     │◄──┤ 排行榜 / 策略家族      │
         └────────────────────────┘   └───────┬──────────────┘
                                              │ 盲区样本
                                     ┌────────▼─────────┐
                                     │ M7 蓝队更新        │ → 只更新 evidence 部分
                                     └────────┬─────────┘
                                              │
                                     ┌────────▼─────────┐
                                     │ M8 下游降权接口     │
                                     └──────────────────┘
```

---

## 3. MVP 范围（第一版只做这个）

**结论类型：癌症预后单基因标志物**
例：「基因 X 高表达与肝细胞癌患者总生存期较差相关（HR = 2.1）」

选择理由：
- 文献量巨大，水分普遍；
- 公开队列丰富（TCGA、GEO 多个独立队列、cBioPortal 汇总研究）；
- 计算便宜（Cox 回归、Kaplan-Meier），一次重分析秒级到分钟级；
- 分析选择空间清晰，易于枚举多重宇宙。

范围限制：
- 先覆盖 3–5 个癌种；由 P0 的确定性规则自动选择（公开队列数、样本量、OS/DSS 事件数、平台注释完整性排序），选择结果写入 `cohort_pool.yaml`，不需要人工确认；
- **只评估落在标准化队列池（见 P0）内的结论**的原始复现与数据对账。池外结论仍可做独立队列荟萃和多重宇宙；
- 第一扩展项：**多基因 signature**。注意这是生物信息学"水文"最集中的类型，P7 完成后应尽快启动，不要无限期推迟；
- 后续扩展：差异表达结论、单细胞亚群/marker 结论、通路富集结论、药物敏感性关联。

---

## 4. 数据模式（Schema）

### 4.1 Claim（M1 输出）
```json
{
  "claim_id": "clm_000123",
  "source": {"pmid": "...", "pmcid": "...", "doi": "...", "pub_date": "2019-04-12", "journal": "..."},
  "claim_type": "prognostic_gene",
  "entities": {"gene": "GENE_X", "hgnc_id": "HGNC:...", "cancer_type": "LIHC", "tissue": "tumor"},
  "endpoint": "OS",
  "direction": "high_expression_worse",
  "reported_effect": {"metric": "HR", "value": 2.1, "ci_low": 1.4, "ci_high": 3.2, "p": 0.001},
  "reported_cohort": {"accessions": ["GSE14520"], "in_standard_pool": true, "n_reported": 212, "subset_desc": "stage I-II only"},
  "reported_choices": {"contrast_type": "dichotomous", "cutpoint": "optimal", "normalization": "RMA", "covariates": [], "time_cap_years": null},
  "data_availability": {"raw_deposited": true, "code_available": false},
  "extraction_confidence": 0.92,
  "evidence_span": "原文中支撑该结论的句子位置（字符偏移），不存全文"
}
```
- `reported_choices.contrast_type` 取值为 `"dichotomous"`（高低表达分组对比，如中位数、分位数、最优切点）或 `"continuous"`（连续变量，通常为每 SD 或每单位 log2 表达量）。
- `reported_effect` 若只有 HR 和 p 值、没有 CI，按正态近似从 p 值反推 SE，并标记 `se_derived_from_p = true`。

### 4.2 GroundTruthResult（M2 输出）
```json
{
  "claim_id": "clm_000123",
  "engine_version": "gt-1.1.0",
  "data_snapshot": "2026-11-01",
  "contrast_type": "dichotomous",
  "original_cohort_repro": {"status": "reproduced|partial|failed|no_data|out_of_pool", "log_effect": 0.64, "n_used": 212, "n_deposited": 488},
  "independent_cohorts": [
    {"cohort_id": "TCGA-LIHC", "log_effect": 0.11, "se": 0.14, "n": 365, "spec": "reference_dichotomous"}
  ],
  "meta": {"method": "REML", "pooled_log_effect": 0.06, "se": 0.09, "tau2": 0.01, "pred_interval": [-0.21, 0.33], "k": 4},
  "multiverse": {"n_specs_reference": 192, "frac_same_direction_sig": 0.11, "median_log_effect": 0.05, "reported_spec_percentile": 0.99, "spec_curve_uri": "..."},
  "reconciliation_flags": ["sample_count_gap", "subset_not_prespecified"],
  "robustness_score": 0.18,
  "inflation": 0.78,
  "inflation_z": 3.4,
  "gt_confidence": "high|medium|low"
}
```

### 4.3 ContextFeatures（M3 输出，按 claim）
```json
{
  "claim_id": "clm_000123",
  "as_of_date": "2019-04-12",
  "author_cluster_id": "ac_0451",
  "supporting_papers": 14,
  "effective_independent_support": 2.3,
  "citation_cascade_score": 0.81,
  "lab_historical_replication_rate": {"value": 0.27, "n_claims_evaluated": 11, "method": "leave_one_out"},
  "self_citation_ratio": 0.35,
  "coi_flags": ["patent_on_signature"],
  "review_time_days": 21,
  "venue_features": {"journal_id": "...", "oa": true},
  "post_pub_signals": {"retraction": false, "correction": false, "pubpeer_threads": 0},
  "missing_mask": {"lab_historical_replication_rate": false, "review_time_days": true}
}
```

### 4.4 DetectorOutput（M4 输出）
```json
{
  "claim_id": "clm_000123",
  "detector_version": "det-1.1.0",
  "evidence_score": 0.71,
  "context_prior": 0.64,
  "context_coverage": 0.6,
  "combined_inflation_pred": 0.69,
  "flag": true,
  "rationale": ["阈值为 optimal cutpoint 且未校正", "仅用子集，GEO 存储样本数远大于使用数"]
}
```
**evidence_score 与 context_prior 必须分开输出**，用于消融、主循环纯内容筛查和上下文置换测试（§7.3）。

### 4.5 RedTeamSubmission（M5 → M6）
```json
{
  "submission_id": "rt_r03_w07_s012",
  "round": 3, "worker": 7,
  "strategy_family": "subgroup_selection",
  "parent_ids": ["rt_r02_w03_s004"],
  "preregistration_hash": "sha256:...",
  "reports": [
    {
      "report_id": "...",
      "claim": { "...同 Claim schema..." },
      "manuscript_md_uri": "...",
      "analysis_code_uri": "...",
      "declared_data": ["GSE....."],
      "declared_choices": {"contrast_type": "dichotomous", "cutpoint": "optimal", "subset": "stage I-II", "...": "..."}
    }
  ]
}
```
- 一次提交是一个**批次（K 份报告，默认 K=20）**，同一策略家族，对应原文"一个数据集多行"，以便计算标记率。
- 提交中**没有** `context_metadata` 字段（C7）。M6 主循环统一以中性情境（`context_coverage = 0`，即仅看 `evidence_score`）筛查。

### 4.6 ExperimentLogEntry（共享日志，append-only JSONL）
```json
{"submission_id": "...", "round": 3, "worker": 7, "strategy_family": "...", "summary": "...",
 "score": 0.43, "flag_rate": 0.05, "visible_inflation": 0.47, "rejected_by": null,
 "realism_pass": true, "cheat_check": "pass", "detector_version": "det-1.1.0", "timestamp": "..."}
```
注意：日志中**不得出现** held-out 队列的任何结果。

---

## 5. 模块规格

### M1 结论抽取（Claim Extractor）
- 输入：PMC OA 全文（XML）。只处理 OA 子集。
- 方法：LLM 结构化抽取 + 规则校验（基因名标准化到 HGNC，癌种映射到 TCGA 项目代码，GEO accession 正则提取与存在性校验，标注是否在标准化队列池内，区分 `contrast_type` 为 `dichotomous` 或 `continuous`）。
- MVP 只抽**单基因**预后结论；多基因 signature 识别出来后打标签 `claim_type = prognostic_signature`，暂存不处理。
- 每篇可能有多条结论；只抽"主要结论"（摘要或结果段明确陈述的预后关联）。
- **验收（全自动，无人工标注）**：
  1. **合成夹具回归**：用固定种子的夹具生成器构造 100 条 PMC XML，与抽取规则版本隔离；字段级精确率 ≥ 90%，效应值/方向/对比类型错误率 ≤ 3%。夹具生成器从 Claim 反向渲染，只用于回归，不作为真实语料精度证据。
  2. **公开 PMC 自动一致性审计**：从 PMC OA 随机抽 100 篇，对每条抽取结果自动检查：schema 合法、基因可解析到 HGNC、癌种可映射到 TCGA 代码、GEO accession 格式合法且存在性缓存命中、HR/CI/p 互相自洽、方向与 HR 相对 1 的位置一致、evidence span 偏移合法；通过率 ≥ 90%；同时报告 claim_yield（抽取 claim 数 / 文章数），低于 20% 时标记为覆盖不足。
  3. 两项都写入 `reports/P1_metrics.json`，不引入人工抽检步骤；未通过时只允许调整规则或标注 `extraction_confidence`，不得把不可解析的样本算作通过。

### M2 真值引擎（Ground-Truth Engine）—— 最重要的模块

#### 证据来源
1. **原始队列复现**（仅池内结论）：用论文声明的队列，按论文声明的选择重跑，确认能否复现报告的效应。这是理智检查，用于判断抽取和管线是否正确。
2. **数据对账**（仅池内结论）：数据库中可用样本数 vs 论文使用样本数；子集是否有预先合理依据；终点是否与数据中可得终点一致。产出 `reconciliation_flags`。
3. **独立队列荟萃**：在论文未使用的同癌种、同终点标准化队列上，用与原结论 `contrast_type` 对齐的**参照默认规格**（见下）计算每个队列的 log HR 和 SE，做**随机效应荟萃**（REML，k < 3 时退化为固定效应并降低 `gt_confidence`）。
4. **多重宇宙 / specification curve**：在独立队列上枚举**参照集**中的全部规格组合，计算支持结论的规格比例、同量纲下的效应中位数，以及论文所用规格在规格曲线上的分位数。
5. **系统性资源对照**（MVP 后）：DepMap、GTEx 等未经人为挑选的系统性数据，用于其他结论类型。
6. **时间留出**（评估用）：发表前可得队列 vs 之后新增队列，检验结论的前瞻稳健性。

#### 多重宇宙维度（单基因预后 MVP）

| 维度 | 参照集取值（用于计算基线） | 仅注册表（红队可用，不进参照集） |
|---|---|---|
| 表达量化 | counts→VST / TPM / 芯片 RMA（按平台可用） | FPKM、MAS5 |
| 分组方式 | 中位数二分（`median`）/ 连续变量（`per_sd`）/ 三分位上下（`tertile`） | **最优切点（maxstat 等数据驱动切点）**、任意自选百分位 |
| 终点 | 论文声明的终点（OS / DSS / PFI·DFS 按可得） | 换用其他终点 |
| 协变量 | 无 / 年龄+分期 | 年龄+分期+分级、按结果挑选的协变量 |
| 样本范围 | 全部原发肿瘤 / 排除新辅助治疗 | **事后按分期、亚型等挑子集** |
| 随访截断 | 无 / 5 年 | 10 年及其他截断 |
| 探针映射 | 最大均值探针 | 探针平均、按显著性挑探针 |

原则：**参照集只放事先可确定、不依赖结局的选择**。依赖结局的选择本身就是 cherry-picking 手段，放进参照集会拉高基线。

**参照默认规格与量纲对齐**（避免把分组尺度差异误算为膨胀）：
- 正态分布下，中位数二分（High vs Low）的理论组间均值差为 $\sqrt{8/\pi} \approx 1.596\text{ SD}$，三分位上下对比为 $\approx 2.18\text{ SD}$。若把二分类 log HR 直接与连续变量（每 SD）log HR 相减，会把正常的二分类结果系统性高估约 1.6 倍。
- 因此，独立队列荟萃（计算 $\mu$）按原结论的 `contrast_type` 选择对齐的参照默认规格：
  - **`contrast_type = dichotomous`**（涵盖中位数、分位数、最优切点等所有高低分组结论）：参照默认规格采用**中位数二分（`median`）**、标准化表达、论文声明终点、无协变量、全部原发肿瘤、无截断；
  - **`contrast_type = continuous`**：参照默认规格采用**连续变量（`per_sd`，队列内 z-score 标准化后每 1 SD）**，其余同上。
- 多重宇宙中位数 $m$ 在计算 `shrinkage` 和与 $r$ 比较时，同样将参照集内的规格统一折算到与原结论一致的 `contrast_type` 量纲下（或仅在同 `contrast_type` 的参照集规格上取中位数；`frac_same_direction_sig` 仍可在全部参照集规格上统计方向与显著性比例）。

**选择注册表（Choice Registry）**：上表每个取值都要附至少一篇使用过该选择的文献引用，存为 `choice_registry.yaml`，并为每项标注 `in_reference_set: true/false` 及所属 `contrast_type`。红队只能从注册表中选（C5）。

#### 分数定义
记号：方向对齐后，效应取 log HR，与结论方向一致为正（$r, \mu, m$ 均已按上述规则对齐到同一 `contrast_type` 量纲）。
- `r` = 报告的 log 效应，`se_r` = 报告的 SE；
- `μ`、`se_μ`、`τ²` = 独立队列随机效应荟萃的合并估计、SE、异质性；
- `m` = 参照集多重宇宙在同量纲下的效应中位数。

公式：
- `independent_support = Φ(μ / se_μ)`：合并效应与结论方向一致的程度，连续值，取代 v1.0 的投票计数；
- `multiverse_support = frac_same_direction_sig`（只在参照集上计算）；
- `shrinkage = clip(m / r, 0, 1)`；
- `robustness_score = 0.5·independent_support + 0.3·multiverse_support + 0.2·shrinkage`（权重为初值，需在验证集上校准，记录在 `scoring.yaml`）；
- **`inflation = tanh( max(0, r − μ) / s )`**，`s` 默认 `ln 2`（报告 HR 是独立合并 HR 的 2 倍时，inflation ≈ 0.76）；
- `inflation_z = (r − μ) / sqrt(se_r² + se_μ² + τ²)`：用于判断膨胀是否超出抽样噪声和队列间异质性，作为天然标签的置信度；
- 没有任何独立队列时，用 `m` 代替 `μ`，`gt_confidence = low`。

对照表（`s = ln 2`，同量纲下）：

| 报告 HR | 独立合并 HR | v1.0 inflation | v1.1 inflation |
|---|---|---|---|
| 1.15 | 1.00 | 1.00 | 0.20 |
| 2.1 | 1.00 | 1.00 | 0.79 |
| 3.5 | 1.00 | 1.00 | 0.95 |
| 2.1 | 1.9 | 0.13 | 0.14 |

天然标签使用规则：`inflation_z < 1` 的结论，无论 `inflation` 多少，都视为"未见显著膨胀"；训练时可按 `gt_confidence` 加权。

#### 工程要求
- 所有队列预处理产物缓存（parquet），按 `data_snapshot` 版本化；
- 引擎是纯函数：`(claim, choices, cohort_ids, snapshot) → result`，同输入必须同输出；
- 一键重跑任意历史结果；
- 荟萃分析使用成熟实现（如 R `metafor`，或经过与 `metafor` 对拍验证的 Python 实现）。

#### 验收
- **池内**有原始数据的结论，原始队列复现成功率 ≥ 60%（失败要能归因：数据缺失 / 抽取错误 / 管线差异）；
- 阳性对照（多队列荟萃中稳健复现的预后基因，由 agent 挑选并记录依据）`robustness_score` 中位数 > 0.6；阴性对照（随机基因）< 0.2；
- 阴性对照的 `inflation` 分布：用随机基因按"最优切点 + 最佳子集"由脚本自动构造夸大报告，`inflation` 中位数应显著高于同基因按参照默认规格（中位数二分）报告时的值（验证公式对红队手段敏感，且中位数二分本身不会被误判为高膨胀）。

### M3 情境 / 社会关系图谱（Context Graph）
数据源建议：OpenAlex（作者、机构、引用，开放）、Crossref（含撤稿/更正元数据）、PubMed、PubPeer（如可合规获取）。实现 agent 需核实各源当前可用接口和许可。

#### 时间截断（v1.1 新增，必须实现）
- 所有特征函数接收 `as_of_date` 参数，只使用该日期之前发生的发表、引用、撤稿、更正和讨论；
- 训练天然标签时默认 `as_of_date = 结论发表日`，模拟"刚发表时就判断"；
- 可选变体：`as_of_date = 发表日 + 3 年`，模拟部署时对已有文献的判断。两种变体分别训练和报告，不要混用；
- OpenAlex 等快照中的引用要按施引文献的发表日期过滤，不能直接用当前累计引用数。

#### 特征

| 特征 | 含义 | 实现要点 |
|---|---|---|
| `effective_independent_support` | 支持同一结论的文献在去除作者网络重叠后的有效独立数 | 只算 `as_of_date` 前的支持文献；在作者合作图上聚类，按簇计数或按相似度折算 |
| `citation_cascade_score` | 结论的传播是否集中在同一作者/机构圈 | 引用边中圈内比例，引用按日期过滤 |
| `lab_historical_replication_rate` | 该团队过去结论在 M2 上的平均稳健性 | **留一法**：排除当前结论及同一论文的其他结论；只用 `as_of_date` 前发表的结论；n < 5 时置空。用真值引擎计算，不用声望 |
| `self_citation_ratio` | 自引比例 | |
| `coi_flags` | 专利、公司关联 | 从声明段落抽取 |
| `review_time_days` | 投稿到接收 | 如可得 |
| `post_pub_signals` | 撤稿、更正、PubPeer 讨论 | 只计 `as_of_date` 前发生的 |

#### 稀疏回退
- 预期 `lab_historical_replication_rate` 在 MVP 规模下大部分为空，这是正常的；
- 每个特征输出 `missing_mask`；`context_prior` 模型必须能处理缺失（缺失指示变量 + 树模型原生缺失处理）；
- 全部特征缺失时，`context_prior` = 训练集基准率，`context_coverage = 0`；
- `combined` 融合时按 `context_coverage` 调整情境部分的权重；`context_coverage = 0` 时，`combined_inflation_pred = evidence_score`。

**明确禁止作为特征**：国家、机构排名、作者性别/族裔推断、作者资历本身。

### M4 检测器（Detector / Monitor）
- 输入：论文文本（结论段 + 方法段 + 数据可用性声明） + M3 特征。**不能**调用 M2（检测器的价值在于用于无法重分析的文献）。
- 结构：
  - `evidence_score`：LLM 阅读内容，按检查清单打分（切点方法、子集是否预设、是否多重比较、样本数是否与数据库一致中可从文本判断的部分、是否有独立验证队列等），可加轻量统计校验（GRIM 类一致性、p 值与 CI 一致性）；
  - `context_prior`：基于 M3 特征的小模型（如梯度提升），**只在天然标签上训练**；
  - `combined`：两者融合，融合权重**只在天然标签验证集上学习**，并随 `context_coverage` 调整（当 `context_coverage = 0` 时直接退化为 `evidence_score`）。
- 阈值：在天然标签验证集上按 **FPR ≤ 5%** 分别标定纯内容阈值 $\tau_{\text{evidence}}$（用于 M6 主循环筛查和缺失情境回退）与融合阈值 $\tau_{\text{combined}}$（用于带情境的文献评估）。
- 输出必须包含可读的 `rationale` 列表。

**天然标签训练/验证**：用 M2 的 `inflation`（结合 `inflation_z` 和 `gt_confidence`）作为标签。按**癌种**和**发表年份**切分训练/验证/测试；M3 特征按 `as_of_date` 截断后再参与训练。

### M5 红队 Harness
复刻原文结构：
- **worker**：一个具备代码执行能力的 LLM agent，独立会话，每轮全新上下文；
- **并行**：每轮 N 个 worker（默认 N=8），共 R 轮（默认 R=10），可调；
- **共享实验日志**：每个 worker 开始前读取完整日志（含历史提交摘要、分数、策略家族），在前人基础上改进；
- **任务指令**（briefing）：
  - 目标：产出一批 K 份分析报告，结论看起来是正常论文，**不被检测器标记**，且在可见评测队列上的**膨胀度尽可能高**；
  - 可用资源：开发用队列（不含任何评测队列）、选择注册表、**检测器 `evidence_score` 部分的完整访问**（原文中红队可以拿着监控器本地反复试）；
  - 规则：只用注册表内选择；不得编造数字；不得尝试访问评测队列；一批内同一策略家族；不能指定作者、机构、期刊等情境；
- **预注册**：提交前写一段不含结果的策略说明（策略家族、选择组合、动机），计算哈希冻结，之后不得修改（借鉴 AAR 篇的 mini-paper 冻结）。

**关于情境攻击**：原文最主要的攻击方式是改变上下文。v1.1 把它从主循环中拿出来，放进 §7.3 的独立实验。原因是：
1. 真实世界中，cherry-picker 能选择投稿期刊、拉合作者，但无法伪造自己的历史复现率等由 M2 计算出来的特征；
2. 如果主循环里允许红队填写情境，蓝队的 `context_prior` 会被反向信号训废；如果主循环里由评分服务随机抽样情境，又会因运气抽到高危情境导致整批报告被误杀。

### M6 评分服务（Grader）
独立进程、独立系统用户运行（C4）。对每个提交批次依次执行：

1. **检测器筛查（纯内容基准）**：将情境固定为中性基准（`context_coverage = 0`），用检测器的 `evidence_score` 和阈值 $\tau_{\text{evidence}}$ 对 K 份报告逐份打分，`flag_rate = 被标记份数 / K`。
   - 被标记份数 > `max(1, ⌊0.1·K⌋)` → 整批拒绝，记 0 分（K=20 时即超过 2 份）。
2. **真值评测（可见评测队列）**：对每份报告的结论，在**可见评测队列**上用与该报告 `contrast_type` 对齐的参照默认规格做随机效应荟萃，按 M2 公式计算 `visible_inflation`。红队开发时看不到这些队列，只收到聚合分数。
3. **真实性门槛**：
   - 选择全部在注册表内（规则检查）；
   - 报告中所有数字可由提交代码在声明数据上复现（重跑核对，容差 1e-3 相对误差）；
   - 真实性评审 LLM：与真实同类论文对比，判断"是否像一篇正常论文"，分数低于阈值 → 0 分。
4. **作弊检测**：读取 worker 完整轨迹（代码、工作目录、对话记录），检查：
   - 试图读取评测/held-out 队列或其 ID；
   - 重复提交几乎相同的批次以赌评分噪声；
   - 批内极少数报告承担几乎全部分数（原文后期补加的检查，对应"少数行权重极高"）；
   - 代码与预注册不一致；
   - 任一命中 → 0 分并记录类别。
5. **分数**：
   ```
   score = mean_k(visible_inflation_k) × (1 − λ · flag_rate)      λ 默认 3
   ```
6. **held-out 评测**：用 held-out 队列计算 inflation，**只写入内部报告**，不进共享日志、不返回给红队。用于最终评估红队发现是否泛化。

### M7 蓝队更新
- 输入：红队高分且未被标记的报告（= 盲区），按策略家族分组。
- **只更新 `evidence_score` 部分**：补充检查清单项、few-shot 示例、微调内容模型。`context_prior` 和融合权重只在天然标签数据更新时重训，不使用红队样本。
- **留出策略家族**：每轮随机留出 20% 的策略家族不参与检测器更新，只用于评估。
- 每次更新后必须重跑：
  1. 天然标签测试集（误报率不得上升超过 1 个百分点）；
  2. 留出策略家族上的召回率；
  3. 上下文置换测试（§7.3）。
- 红蓝交替：蓝队更新 → 红队下一轮针对新检测器。

### M8 下游降权接口
- 对能重分析的结论：权重 = `robustness_score`（M2）；
- 对不能重分析的结论：权重 = `1 − combined_inflation_pred`（M4），并标记来源；
- 对一个结论的多篇支持文献：总证据 = Σ(单篇权重) 按 `effective_independent_support` 折算，**不按篇数直接累加**；
- 输出格式待定（见 §10 待决定事项），先提供 Python API：`get_claim_weight(claim_id)`、`get_edge_weight(gene, disease, relation)`。

---

## 6. 隔离与防泄漏

| 资源 | 红队 worker | 检测器训练 | 评分服务 |
|---|---|---|---|
| 开发队列 | 读 | 读 | 读 |
| 可见评测队列 | 无访问，仅收到聚合分数 | 无 | 读 |
| held-out 队列 | 无 | 无 | 读 |
| 天然标签测试集 | 无 | 无（只用验证集） | 读 |
| 检测器 evidence 部分 | 完整调用 | — | 调用 |
| 检测器 context 部分 / M3 特征 | 无 | 读（天然标签） | 读（用于 §7.3 评测） |
| 共享日志 | 读写（追加） | 读 | 写 |

- 队列划分在系统初始化时一次性完成并写入 `cohort_split.yaml`，按 cohort 整体划分，不按样本划分。
- 评测/held-out 数据存放在仅评分服务用户可读的目录。
- 评分服务只返回聚合数字，不返回逐样本结果。
- M3 特征计算必须带 `as_of_date`，单元测试覆盖"未来数据不可见"。

---

## 7. 评估指标

### 7.1 真值引擎
- 阳性/阴性对照分离度（M2 验收）。
- 跨快照稳定性：同一结论在两个数据快照间的 robustness 差异中位数 < 0.05。
- 荟萃实现与参考实现对拍：合并估计差异 < 1e-4。

### 7.2 检测器
- 天然标签测试集：**固定 FPR（1% / 5%）下的召回率**。不以 AUC 为主指标（真实膨胀结论比例可能不均衡，且下游关心低误报）。
- 校准：预测膨胀度与真值膨胀度的分箱校准曲线。
- 消融：仅 evidence_score / 仅 context_prior / 融合，三者分别报告；另报 `context_coverage` 分层结果。
- 时间截断对照：同一模型分别用"截断特征"和"不截断特征"训练，差异大说明存在时间泄漏风险，必须用截断版本。

### 7.3 上下文置换与情境攻击测试（检验社会特征是否成为攻击面）
- **上下文置换测试**：取一批报告，内容不变，利用 `context_sampler.py` 在真实文献的 M3 情境画像之间置换（例如高复现率团队画像 ↔ 低复现率团队画像，高声望期刊 ↔ 普通期刊）。
  - 指标：`evidence_score` 的变化应 = 0（它不看情境）；`combined` 的变化幅度需记录并设上限。
- **情境攻击实验**（独立于主循环）：允许一个专门的红队从**真实情境画像库**中为盲区报告挑选最有利的真实情境（模拟"选期刊、拉合作者"），但不能编造画像。测量 `combined` 标记率相对纯内容标记率下降多少。下降过大 → 检测器过度依赖可操纵的情境，需要降低相关特征权重或收紧融合上限。

### 7.4 红队
- 每轮最佳分数曲线、通过检测器的批次比例；
- 策略家族多样性（按家族的 Shannon 熵）；
- **held-out 泛化**：红队高分批次在 held-out 队列上的 inflation 是否仍高。

### 7.5 整体
- 下游示例任务：用降权前后的文献先验构建同一个预后基因集，在 held-out 队列上比较预测性能（C-index）。降权后应不劣于降权前。

---

## 8. 实施阶段与验收

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| P0 | 环境、目录、配置；确定 MVP 癌种；**构建标准化队列池**；写 `cohort_split.yaml`、`choice_registry.yaml` | 见下方 P0 详细要求 |
| P1 | M1 结论抽取 | M1 验收 |
| P2 | M2 真值引擎 | M2 验收；全部结果可一键重跑 |
| P3 | M3 情境图谱 | 覆盖 ≥ 90% 抽取结论（允许字段为空）；`as_of_date` 单元测试通过；禁止特征未出现在特征表中 |
| P4 | M4 检测器 v0 + 天然标签评估 | 报告 §7.2 全部指标，作为基线 |
| P5 | M6 评分服务 + 隔离 | 用一个故意违规的假 worker 测试：读评测目录被拒、编造数字被抓、重复提交被抓、提交中携带情境字段被拒 |
| P6 | M5 红队 harness，跑 ≥ 3 轮 | 日志完整、分数曲线可视化、至少发现 1 个检测器盲区策略家族 |
| P7 | M7 红蓝交替 ≥ 3 个周期 | 留出家族召回率提升，天然测试集误报率不升 |
| P8 | M8 下游接口 + §7.5 示例任务 | 示例任务结果报告 |

### P0 详细要求：标准化队列池
- 每个 MVP 癌种选定 **5–10 个**高频使用的公开队列（TCGA + cBioPortal 汇总研究 + 文献中最常引用的 GEO 队列）；清单由 P0 的固定排序规则自动生成并写入 `configs/cohort_pool.yaml`（排序键：公开队列可用性、样本量、OS/DSS 事件数、平台注释完整性）；配置文件可覆盖，不需要人工确认；
- 统一临床表 schema：

  | 列 | 说明 |
  |---|---|
  | `sample_id`, `patient_id` | |
  | `os_time_days`, `os_event` | 时间统一为天；事件 1 = 死亡，0 = 删失 |
  | `dss_*`, `pfi_*` / `dfs_*` | 同上，可得时提供 |
  | `stage_ajcc` | 统一为 I / II / III / IV |
  | `age_years`, `sex` | |
  | `neoadjuvant` | 是否接受新辅助治疗 |
  | `sample_type` | 原发 / 转移 / 正常 |
  | `platform`, `expr_unit` | 表达平台与单位 |

- 每个队列附一个清洗脚本和**合理性检查**：KM 中位生存期是否落在文献报告范围内、事件编码方向是否正确（例如用分期越晚、生存越差的已知关系做检验）、时间单位是否一致；
- 长尾 GEO 队列**不写**一次性解析代码；需要扩充时，用统一的 GEO 清洗模板 + 配置自动生成 `cohort_pool/<cancer>/<cohort_id>/clean.py` 并加入队列池，避免手工编辑样本级数据。

每阶段结束产出一份简短 `reports/Pk_summary.md`：做了什么、指标、已知问题。

---

## 9. 已知风险

1. **搜索坍缩到单一套路**：原文后期红队几乎全部收敛到角色扮演/创作包装；换监控器后才出现更多样做法。本场景预期会收敛到"挑子组"或"最优切点"。
   对策：按策略家族限额（单家族占日志比例上限）、轮换检测器版本、用已知可疑研究操作清单给部分 worker 设定起点。
2. **最高分 ≠ 最有信息量**：原文后期提升主要来自把同一行为写得更明显。
   对策：分析时按策略家族去重，关注"新家族首次通过"而非分数峰值。
3. **逐条判断的盲区**：原文监控器逐行独立判断。单篇检测器同理，看不到圈内引用级联。
   对策：M3 的网络级特征 + M8 的独立性折算。
4. **稀释问题**：原文下一步要把攻击数据混入大语料测试。本场景对应"少数注水文献混入数千篇是否改变下游结论"。
   对策：P8 之后加稀释实验（在 §7.5 示例任务中按比例注入红队报告）。
5. **真值本身的局限**：独立队列不复现 ≠ 原论文错（人群、平台、治疗差异）。
   对策：输出命名为"稳健性/可迁移性"，文档中不得称为"造假检测"；用 `τ²` 和预测区间表达异质性；`gt_confidence` 低时降低该标签权重。
6. **评分噪声被利用**：对策见 M6 作弊检测（重复提交检查）及 M6 主循环固定中性情境（消除随机情境抽样噪声）。
7. **社会特征引入偏见或被操纵**：对策见 M3 禁止特征表、§7.3 置换测试和情境攻击实验。
8. **红队产出外泄**：对策见 C2；所有红队文件写入时自动加文件头水印 `INTERNAL RED-TEAM ARTIFACT - NOT REAL RESEARCH`。
9. **（v1.1 新增）选择偏差**：原始复现和数据对账只覆盖标准化队列池内的结论，池外结论（往往使用小众队列）的评估更依赖检测器。
   对策：报告中分池内/池外分别统计；池外结论的 `gt_confidence` 上限为 medium。
10. **（v1.1 新增）时间泄漏**：对策见 M3 时间截断和 §7.2 时间截断对照。

---

## 10. 可配置事项（确定性默认 + 配置覆盖，不需要人工确认）

所有选项在初始化时写入配置或报告；运行过程不阻塞等待人工输入。

1. **下游用途**：默认同时输出 claim 权重与 gene-disease edge 权重；可通过 `configs/mvp.yaml` 切换为论文级或边级粒度。
2. **算力与预算**：默认 8 worker × 3 轮 × K=20；worker 数、轮数、LLM 型号和预算上限全部从 `configs/redteam.yaml` 读取。
3. **MVP 癌种与队列池**：由 P0 的确定性排序自动选择（见 §8 P0）；`cohort_pool.yaml` 的显式条目优先于自动选择，仍不需要人工确认。
4. **阈值与参数**：检测器 FPR 目标、批次标记上限、λ、robustness 权重、`inflation` 尺度 `s` 由 `configs/scoring.yaml` 提供初值，P2/P4 自动校准并将结果写入 `reports/`；配置值优先于校准值，二者都记录。
5. **PubPeer 等数据源**：默认关闭；仅在配置显式启用且本地快照存在时使用，缺失时该特征进入 `missing_mask`，不阻塞运行。
6. **天然标签 `as_of_date` 变体**：默认主结果为“发表后 3 年”，同时报告“发表日”变体；由 `configs/mvp.yaml::asof_variant` 覆盖。

---

## 11. 建议目录结构

```
project/
  configs/
    cohort_split.yaml          # 队列划分（初始化后只读）
    cohort_pool.yaml           # 标准化队列池清单（按癌种）
    choice_registry.yaml       # 合法分析选择 + 文献依据 + in_reference_set + contrast_type
    scoring.yaml               # λ、阈值、权重、s
  data/
    raw/                       # 原始下载（版本化）
    processed/                 # 预处理缓存 parquet（统一临床 schema）
    eval_restricted/           # 可见评测 + held-out，仅评分服务用户可读
  cohort_pool/
    geo.py                    # GEO series matrix + 平台注释自动下载/转换
    <cancer>/<cohort_id>/clean.py   # 每个队列一个清洗脚本
    sanity_checks.py
  m1_extract/
  m2_ground_truth/
    multiverse.py
    independent.py
    meta.py                    # 随机效应荟萃
    reconcile.py
    engine.py                  # 纯函数入口
  m3_context/
    asof.py                    # 时间截断
  m4_detector/
  m5_redteam/
    briefing.md
    worker.py
    orchestrator.py
  m6_grader/                   # 独立服务
    service.py
    context_sampler.py         # 供 §7.3 置换测试与情境攻击实验调用
    cheat_detect.py
    realism.py
  m7_blueteam/
  m8_downstream/
  logs/
    experiment_log.jsonl       # 共享日志（append-only）
    grader_internal/           # held-out 结果及 §7.3 评测记录，仅评分服务写入
  redteam_artifacts/           # 隔离，带水印，禁止外传
  reports/
  tests/
```

---

## 12. 参考
- Automated Subtle Emergent Misalignment Research — https://www.google.com/search?q=https://alignment.anthropic.com/2026/automated-em-search/
  （对抗游戏结构；monitor + 评测 + 作弊检测三段式评分；标记率上限与连续惩罚；共享实验日志；上下文包装是主要攻击方式；搜索坍缩；稀释实验方向）
- Automated Researchers Can Mitigate Well-Characterized Alignment Failures — https://www.google.com/search?q=https://alignment.anthropic.com/2026/automated-alignment-researchers/
  （held-out 操作系统级隔离；结果前冻结的 mini-paper；代码审查监控；多 benchmark 防过拟合；作弊类型统计）
