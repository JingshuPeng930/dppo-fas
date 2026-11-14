# DPPO-FAS 后续工作计划（2026-11-13）

依据：proposal 原文（RQ1–RQ3、baseline、LEGO 任务）、助教的反馈、截至 11-13 的实验结果。
用法：每做完一项就在这里更新状态，结果数字写进 FAS_README.md。

---

## 0. 助教反馈要点，以及我们的数据已经说明了什么

| 助教的意见 | 我们怎么回应 |
|---|---|
| 去看 PLR、Active Domain Randomization 等 adaptive curriculum 工作 | Related Work 里加一节"自动课程 / 环境优先级"。PLR、ADR、TSCL、ALP-GMM、Goal-GAN 中的打分方式，能放进同一个采样框架的，就作为 baseline 实现（见第 2 节） |
| 只看失败率未必是最好的标准，几乎总失败的条件学习信号可能很弱 | **我们的数据已经支持这一点**：hopper 最难的 bin 在预训练时成功率只有 7.5%，FAS 没有任何提升；walker2d 最难的 bin 成功率约 60%，能学会，FAS 提升显著（t = 3.8）。报告要把这一点当作核心发现 |
| 应该和"中等难度 / 能力边界"以及"学习进度"类课程比较，而不只是和 uniform 比 | 新增 frontier（边界）采样和 learning-progress（学习进度）采样两个 baseline，再加一个 PLR 风格的 value-error 采样。三者都用和 FAS 相同的 p(c) 公式，只换打分函数，这样比较才公平 |
| 引用了 "SGS" | 已找到：Success Guided Sampling，Zhang et al.，"A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control"，CoRL 2026，arXiv 2610.12465。它用以目标成功率 t 为峰值的 Beta 核打分，再取 softmax(log(K+ε)/T)。我们按原文公式实现，即 `fas.score=frontier`，β=1，t=0.66，κ=5，T=2，ε=1e-8（原文给 locomotion 的默认值）。和原文的区别：成功率用 EMA 估计，而不是最近 100 个回合的滑动窗口 |

调整后的叙事：**不是"FAS 比 uniform 好"，而是"在 DPPO 微调里，采样分数应该衡量什么？什么条件下优先采失败的环境才有用？"** 这样 proposal 原本的 RQ 保持不变，只是把 RQ1 的比较对象扩大了。

---

## 1. 当前状态

| 项目 | 状态 |
|---|---|
| DPPO 复现（hopper，3 个 seed，另有 1 次 1000 轮） | 完成 |
| RQ1 hopper（grid v2）：DPPO / uniform / FAS α=1 / FAS α=3 | 完成，FAS 与 uniform 打平 |
| RQ1 walker2d：同上，5 个 seed；另有 400 轮的 uniform 和 FAS 各 1 次 | 完成，FAS α=3 显著更好 |
| RQ1 halfcheetah：DPPO / uniform / FAS α=3 各 3 个 seed | **11-13 16:32 开始跑**，预计今晚到明早结束 |
| RQ2：α/β 强度扫描 | 部分完成（α=1/β=0.5 对 α=3/β=0.9） |
| RQ2：奖励稀疏度、critic error | 未开始 |
| RQ3：RFT | 未开始 |
| LEGO | 由其他组员负责，不在本计划范围内 |
| 第一阶段：SGS 和 learning progress 采样，walker2d 和 hopper v2 各 3 个 seed | **11-13 23:30 开始跑**（tmux `fas_{walker2d,hopper}_{sgs,lp}_s{0,1,2}`），代码见 commit 3367bef |

截止时间：大约两周后（11-27 前后），所以 PLR value_error 打分和稀疏奖励包装器都放在可选项里。

---

## 2. 第一阶段：比较不同的采样分数（回应助教，优先级最高）

### 2.1 实现（`util/fas.py`，新增 `fas.score` 选项）
所有方法都用 p(c) = β·w_c/Σw + (1−β)/|C|，只有打分 w_c 不同：

| `fas.score` | 打分 | 来源 |
|---|---|---|
| `failure`（现有） | (s_c + δ)^α，其中 s_c 是失败率的 EMA | 我们的 FAS |
| `frontier` | (4·s_c·(1−s_c) + δ)^α，在成功率 50% 时最大 | Goal-GAN（中等难度的目标）、助教提到的 SGS |
| `progress` | (\|EMA_fast(succ) − EMA_slow(succ)\| + δ)^α，按最大值归一化 | TSCL、ALP-GMM |
| `value_error` | 该 bin 内平均 \|GAE advantage\|，按排名做优先级，并加 staleness 项 | PLR |

- 前两种只需要改 `FASSampler.probs()`，`progress` 要多维护一个 EMA，加起来不到半天。
- `value_error` 需要把每一步的 bin 编号写进 PPO buffer，在算完 GAE 之后按 bin 汇总 \|A\|，约 1 天。如果时间紧可以先跳过，只在 Related Work 里讨论。
- 新增单元测试，并保证 `score=failure` 时结果和现有实现逐位一致。
- 不实现 ADR：它需要训练判别器，而且是连续参数空间，偏离我们的框架太远，只在 Related Work 里讨论。

### 2.2 实验
- **任务：**walker2d（FAS 有效）、hopper grid v2（FAS 无效，是检验 frontier 采样的关键场景）、halfcheetah。
- **方法：**`sgs`（frontier，β=1，按 SGS 原文设置）和 `lp`（progress，α=1，β=0.8，即 80% 按学习进度、20% 均匀，与 ALP-GMM 一致），以及可选的 value_error，每个 3 个 seed，200 轮。uniform 和 FAS α=3 复用已有结果。
- **按预训练策略的成功率模拟采样分布：**在 hopper v2 上，FAS α=3 把 58% 的概率放在成功率低于 0.2 的 bin，SGS 只放 5%；在 walker2d 和 halfcheetah 上，没有成功率低于 0.2 的 bin，两者都把约 80–95% 的概率放在成功率 0.2–0.8 之间的 bin。因此 hopper 是区分两种采样的关键任务。
- **规模：**先做 walker2d 和 hopper，2 种打分 × 2 个任务 × 3 个 seed = 12 个任务，gpu3 同时跑 9–12 个，约 6–8 小时。再加 halfcheetah 6 个任务。
- **小调参：**frontier 在 walker2d 上补跑 α=1，避免因为 α 选得不合适而对它不公平。

### 2.3 预先写下的假设（结果出来后对照）
- H1：在 hopper 上 frontier > failure ≈ uniform，因为 frontier 不会把采样浪费在学不会的 bin 上。
- H2：在 walker2d 上 frontier ≈ failure > uniform，因为难的 bin 本来就在能力边界附近。
- H3：progress 的打分噪声大（每个 bin 每次评估只有 1–3 个回合），可能不如前两种稳定。

### 2.4 分析（报告的核心图）
- 每种方法的学习曲线（平衡成功率、最难 12 个 bin）。
- **"什么时候优先采失败条件有用"：**横轴是预训练策略在各 bin 的成功率，纵轴是微调后该 bin 的提升；每种采样方法一条线，三个任务各一张图。
- 采样分布随时间的变化（bin × 迭代的热力图），以及最难的 bin 实际拿到的训练步数占比（`diag_fas.py`）。

---

## 3. 第二阶段：RQ2，奖励稀疏度和训练稳定性

proposal 原文："vary reward sparsity and failure-sampling strength, tracking success rate and critic error"。

1. **成功信号的稀疏度**（成本低）：把成功判据从 `survive_steps` 换成严格的 `survive`（必须活满 1000 步）。hopper 的预训练策略在这个标准下全部失败，正好对应助教说的"几乎总失败"的情况。在 walker2d 和 hopper 上比较 failure、frontier、uniform。
2. **奖励本身的稀疏度**（风险较高）：增加一个可选的稀疏奖励包装器，例如只在每存活 100 步时给 +1，或者只在回合结束时给成功奖励。PPO 在这种设定下可能完全学不动，所以先在 walker2d 上用单个 seed 试 50 轮，能学再扩展到更多 seed。
3. **critic error**：value loss 和 explained variance 已经有日志；新增按 bin 统计的 value loss，看 FAS 是否让 critic 在难 bin 上更不准。
4. **α 扫描**：在 walker2d 上补 α ∈ {1, 2, 3, 5}、β=0.9 各 1–2 个 seed，画"采样强度 vs 最终成功率"的曲线。

---

## 4. 第三阶段：RQ3，RFT（"提升来自发现成功轨迹，还是来自策略梯度？"）

proposal 原文：从原始 checkpoint 出发，只用 DPPO 发现的成功轨迹训练 RFT 策略，在相同交互预算下和对应的 RL 策略比较。

1. **保存轨迹**：在训练 agent 里加 hook，只保存成功回合的观测、执行的动作块和 bin 编号（不用上游会把所有轨迹留在内存里的 `save_trajs`）。walker2d 上重跑 uniform、FAS 以及第一阶段效果最好的采样方法，各 3 个 seed；同时可以检验之前结果能否复现。
2. **RFT 训练**：从预训练 checkpoint 出发，在这些成功轨迹上做扩散 BC，复用预训练阶段的 loss 和 agent，不再和环境交互，所以预算自动对齐。可选：按 bin 加权，保证难 bin 有足够的样本。
3. **对照组**：预训练策略在 uniform 条件下跑同样多的步数，做纯拒绝采样，然后同样训练 RFT。这组用来分离"RL 帮助发现了成功轨迹"的那部分贡献。
4. **结果表**：行是轨迹来源（uniform / FAS / frontier / 预训练策略），列是 RL 策略和 RFT 策略的成绩，评估用训练网格和 OOD 网格。
   - 如果 RFT-FAS ≈ DPPO-FAS，说明提升主要来自 FAS 找到了更多成功轨迹。
   - 如果 RFT 明显更差，说明策略梯度更新是关键。

---

## 5. 第四阶段：LEGO 装配任务（取决于时间，需要团队决定）

- proposal 承诺了这个任务，但目前完全没开始。需要新环境，基于 SimFoundry [13]，以及一个实现 `set_condition_probs` / `fas_ep_*` 接口的包装器。
- 这个任务本身是二元成功奖励，正好用来检验第二、三阶段的稀疏奖励结论。
- **建议**：由 1–2 名组员并行负责环境搭建；我负责 gym 部分的实验。如果最后时间不够，就在报告里做小规模的结果，或者写成未来工作，并向助教说明原因。

---

## 6. 统一的实验规范
- 所有方法：200 轮（1600 万环境步），3 个 seed，关键比较用 5 个；采用相同的评估网格（train 和 OOD）和相同的 checkpoint（state_199）。
- 指标：平衡成功率、最差 bin 成功率、回报、学习曲线下面积（样本效率）；显著性用 Welch t 检验，报告 seed 数。
- 工具：`scripts/fas/finetune.sh`、`evaluate.sh`、`summarize_evals.py`、`learning_curves.py`、`diag_fas.py`。
- gpu3 同时最多跑约 12 个任务；tmux 会话名不能带 "."。
- 每个阶段结束后把结果追加到 FAS_README.md，commit 并 push。

## 7. 顺序与时间估计

| 顺序 | 内容 | 实现时间 | 计算时间 |
|---|---|---|---|
| 1 | halfcheetah RQ1 收尾和评估 | – | 正在进行 |
| 2 | 实现 frontier 和 progress 打分，walker2d 和 hopper 各 3 个 seed | 0.5 天 | 约 8 小时 |
| 3 | 实现 RFT 的轨迹保存 hook 和训练脚本（与第 2 项的训练同时做） | 1 天 | 约 6 小时 |
| 4 | halfcheetah 上的 frontier 和 progress，PLR 的 value_error 打分（可选） | 0.5–1 天 | 约 8 小时 |
| 5 | RQ2：严格判据、α 扫描，稀疏奖励先做小规模试验 | 0.5 天 | 约 8 小时 |
| 6 | 分析图表和报告 | 1–2 天 | – |
| – | LEGO | 团队决定 | – |

## 8. 已确认（11-13）
1. SGS 的论文已找到，见第 0 节。
2. 截止时间大约两周后。
3. LEGO 由其他组员负责。
