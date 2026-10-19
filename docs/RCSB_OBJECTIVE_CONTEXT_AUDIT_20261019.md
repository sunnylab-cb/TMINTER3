# RCSB pairwise PPIS: loss、接触角色与输入语义复核

日期：2026-10-19。本文件补充[原诊断交接](RCSB_PAIRWISE_DIAGNOSTIC_HANDOFF_20261019.md)，回应独立审阅意见。

仓库：https://github.com/sunnylab-cb/TMINTER3 。实查为 **PRIVATE**；未授权访问的 404 不代表文件不存在。没有修改仓库可见性。仓库内原始 Markdown 未发现审阅附件中的 `\|`、转义代码围栏或 `**#`；这可能是附件转存问题。

## 1. 本轮范围和结论边界

- 没有修改正在运行的三个训练作业、标签、特征、split、checkpoint selection 或阈值。
- 完整 train/val 目标函数与 sampler 审计；完整 train mask 计数；完整 val mask 差异程度和装配分层。
- 只使用 train/validation 标签。覆盖漏斗使用预先冻结的全 split **资产元数据**，不是 test 标签或预测。
- 新诊断代码经 gpt-6-luna 定向 review；运行源码另与冻结 trainer 的 AST/hash 对齐。
- 当前仍不能把原生验证 AP 差距归因于重复数据、缺失残基、loss 或模型容量中的某一项。

共同验证子集 +0.058067 AP 的证据仍有效，但比较的是当前两套完整流程，不是数据量单变量因果实验。同 epoch 的训练步数不等，特征来源也不等。

## 2. 完整训练 mask 审计

| 项目 | Train | Validation |
|---|---:|---:|
| Pair instances | 1,150,993 | 123,863 |
| Directional records | 2,301,986 | 247,726 |
| Ordered exact-sequence-pair groups | 499,240 | 37,049 |
| 重复组 | 204,538 | 16,839 |
| 不同 mask 的组 | 150,682 | 12,484 |
| 不同-mask 组中的记录 | 1,836,388 (79.77%) | 216,006 (87.20%) |
| 不同 ordered-input + mask 组合 | 995,617 | 82,376 |

标签的索引范围、contact 唯一性、左右 contact 投影及 observed-only 合同均检查通过。这里的“相同输入”首先指序列 UID 分组；实际函数与张量的一致性需结合第 7 节。

运行纠正：首次作业 736665 的 objective 阶段成功，但新 mask 审计错误地要求冗余 `observed_indices` 列表。冻结 materializer 实际使用 `observed_sequence_only=true` 和完整 observed count。修正诊断后，736704 在 6 分 49 秒内完整完成；没有补造训练标签。若记录提供显式 observed indices，仍严格验证每个索引。

## 3. 多 mask 不是多构象的同义词

### 3.1 同一个同序列链对的左右界面

Validation 有 74,543 个同序列 pair instances。其中 55,191 个 (74.04%) 的左右标签不同：

- 左右 mask 平均 Jaccard：0.40455。
- 左右 mask 对应残基位点的平均 disagreement：10.3338%。
- 无序 exact pairs 为 22,274，有序组为 37,049；两方向完整保留，故同序列 unique pairs = `2*22274-37049=7499`。

同序列二聚体或大型装配中的两个亚基可以用不同表面互相接触，不需要先假设跨实验构象变化。序列相同也不等于原始实验坐标相同；本模型使用 UID 级缓存的预测单体，而不是该实验实例的坐标或相对位姿。

### 3.2 固定装配、固定输出侧后仍然存在差异

| 分组 key | 总组数 | 不同-mask 组数 | 不同-mask 组中的方向记录 |
|---|---:|---:|---:|
| query UID + partner UID + PDB + assembly + left/right side | 65,950 | 17,487 | 193,555 (78.13%) |
| query UID + partner UID + PDB + assembly | 48,401 | 15,740 | 206,746 |
| query UID + partner UID + PDB | 43,856 | 14,759 | 209,902 |
| query UID + partner UID | 37,049 | 12,484 | 216,006 |

第一行固定 side 后的重复必须来自不同 pair instances，而不是仅把同一个 pair 的左右两行混在一起。说明**同一装配内不同实例/邻居角色已足以产生大量异质标签**。

这些是嵌套且重叠的统计，不能相加为独立原因份额；也没有进一步将不同邻居角色和同装配中不同单体构象完全分开。跨 PDB 数量不等于“构象差异数量”。RCSB 官方也将 [6YF7](https://www.rcsb.org/structure/6YF7) 的两个 biological assemblies 列为二十面体 homo 180-mer，支持大型装配背景，但该页面本身不能验证我们计算的 mask 数量。

### 3.3 差异不只是“至少翻转一个残基”

对同一 ordered UID group 的经验标签频率 `q_i`，计算：

$$
H_g=\frac1L\sum_i[-q_i\log q_i-(1-q_i)\log(1-q_i)],
\qquad
d_g=\frac{n}{n-1}\frac1L\sum_i2q_i(1-q_i).
$$

`d_g` 是从该组不放回抽两条不同记录时的期望残基 disagreement；不是抽两种不同 mask。另计算各记录对众数 mask 的频率加权 Jaccard。单例组的经验熵、disagreement 记零，但这不代表其总体没有不确定性。

| Val 聚合 | group-macro disagreement | record-weighted disagreement | 对众数 mask 的 group-macro Jaccard |
|---|---:|---:|---:|
| 全部 ordered groups | 3.86% | 11.32% | 0.85894 |
| 重复组 | 8.50% | 12.33% | 0.68965 |
| 不同-mask 组 | 11.46% | 12.99% | 0.58139 |

不同-mask 组的 group-macro 经验熵为 0.12099 nats/residue；record-weighted 为 0.18704。它们不是总体不可约误差的无偏估计，也不是 AP 上限。尚未完成 5.5 Å 邻域边界翻转统计和 BCE 的 `H(q)+KL(q||p)` 实测分解。

## 4. 当前实际 loss，而非简化 BCE

冻结运行使用 `PairResidueLoss`，对链 c 的 L 个有效残基：

$$
D_c=L_c+(w-1)\sum_i y_{ci},\quad
t_{ci}=(1-\epsilon)y_{ci}+\epsilon/2,
$$
$$
\ell_c=\frac{\sum_i[-w t_{ci}\log p_{ci}-(1-t_{ci})\log(1-p_{ci})]}{D_c}
+\lambda(\bar p_c-\bar y_c)^2,
\quad \epsilon=0.05,\ \lambda=0.25.
$$

每个 batch 为 `0.5*(mean(left chains)+mean(right chains))`，每 batch 一次 optimizer step，无梯度累积。当前 pair-contact 和 consistency loss 权重为零。正类权重来自训练残基计数：PDBTM 6.93963986，RCSB 10.38066838。

PyTorch 整数 target + weighted mean 的分母是 hard target 对应的权重和；label smoothing 不把该分母改成 soft target 权重和。概率 target 的 mean 归一化不同。已用冻结类的 loss 和梯度测试，而非只凭文档推测。[PyTorch 2.7.1 LossNLL.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/aten/src/ATen/native/LossNLL.cpp)

### 4.1 多 hard masks 仍可能正确学习边际监督

固定同一输入预测 p、上下文外层系数 `alpha_c` 时，CE 的有效 context 系数为 `omega_c=alpha_c/D_c`，有效标签为：

$$
q_i^{\rm eff}=\frac{\sum_c\omega_c y_{ci}}{\sum_c\omega_c}.
$$

固定 class weight 和线性 smoothing 本身不破坏 target 线性；改变的是 mask-dependent 分母和外层权重。压缩为 soft target 时还必须保留总尺度。RCSB 权重下，同为 100 aa、阳性数为 1 和 50 的两个上下文，`D_50/D_1=5.2023`；相同外层系数时稀疏 mask 有较大的 CE 前置系数。

平方 ratio 项满足：

$$
\mathbb E_\alpha[(\bar p-\rho_c)^2]
=(\bar p-\mathbb E_\alpha\rho_c)^2+\mathrm{Var}_\alpha(\rho_c).
$$

最后一项与参数无关，因此梯度等价。**但 ratio 依赖 alpha 加权均值，CE 依赖 alpha/D 加权均值，不能随便用同一个平均 mask 替换两者。** 期望目标一致也不保证有限步 SGD/Adam 轨迹一致。

### 4.2 Smoothing 与较大正类权重的概率尺度

仅看 CE、每个残基概率可以自由调整时，真实负残基的最优概率：

$$
p_0^*=\frac{w\epsilon/2}{1-\epsilon/2+w\epsilon/2}.
$$

| class weight | smoothing | 负残基 CE-only 最优概率 |
|---|---:|---:|
| 1 | 0.05 | 0.02500 |
| PDBTM 6.93964 | 0.05 | 0.15106 |
| RCSB 10.38067 | 0.05 | 0.21022 |

使用实际冻结 loss 的梯度验证了驻点。加入 ratio=0.25 的 label-aware 两参数 toy（100 aa、10 个正例），RCSB 的负/正例概率约 0.18965/0.99753、平均概率 0.27044。

这是已知标签且自由分离两类的 toy，不是模型预测，也不是网络输出的硬下界。它支持“原始概率不是自然接触频率”，不证明 AP 损失由 smoothing 导致，也不授权直接套逆变换校准。

## 5. 动态 batch 的量化结果

固定参数处，对一个 epoch 所有 batch loss 等权平均，pair j 的外层系数为 `1/(T*B_j)`，每个方向再除以 2。以下不包含 CE 内部的 `1/D_c`，更不是实际 gradient norm 或 Adam 更新份额。

| Training 数据/配置 | >=1000 aa 方向记录占比 | 这些长链的 loss 外层系数占比 | >20 重复组记录占比 | >20 重复组系数占比 |
|---|---:|---:|---:|---:|
| PDBTM S5, 1M | 2.20% | 17.34% | 25.30% | 15.69% |
| RCSB, 1M | 2.05% | 12.76% | 49.10% | 38.79% |
| RCSB, 4M | 2.05% | 20.80% | 49.10% | 38.29% |
| RCSB, 8M | 2.05% | 20.79% | 49.10% | 37.51% |

这否定“记录重复次数就是训练贡献比例”的简单解释。长链相对上权并非 RCSB 独有；PDBTM 也存在，因此不能据此单独解释跨域差距。4M/8M 除了吞吐和步数，也改变了有效外层权重。

PDBTM 系数复核使用其原始运行 config/manifest 和本轮冻结的 sampler 实现，训练记录数、正负类计数、class weight、每 epoch 2,004 batches 全部对上。该复核在本机 CPU 上完成；JSON 中 PyTorch 版本指审计环境，不应当作历史 HPC2 训练环境版本。RCSB 审计及实际 checkpoint probe 在 HPC3 的 PyTorch 2.7.1+cu128 上执行。

RCSB train 的方向平均阳性率为 0.10163，按 batch 系数加权后为 0.08763/0.08579/0.08558。PDBTM 对应 0.17688 -> 0.11957。应把这些分布差异与目标概率尺度一起看。

### 5.1 当前日志 Val loss 不是 packing-invariant 指标

`_evaluate` 最后调用 `np.mean(losses)`，每个 batch 等权。固定所有预测为 p=0.5，在同一 RCSB validation 上，仅变 packing，实际 loss 就是：

| 配置 | Val batches | 固定 p=0.5 的日志口径 loss |
|---|---:|---:|
| 1M | 14,835 | 0.827612 |
| 4M | 3,794 | 0.830689 |
| 8M | 2,003 | 0.834247 |

这是目标函数解析计算的控制，不是训练模型结果，也不是把真实 val loss 差异作因果分解。共同固定 chain/pair 权重的 loss 才适合跨 packing 比较。正式 checkpoint 选择用 `chain_ap_macro`，不是此日志 loss；AP 仍需结合预测的 batch 数值稳定性检查。

## 6. BCE 边际概率与 instance-macro AP 不等价

采用审阅者的三残基反例：60% mask 为 `(1,1,0)`，40% 为 `(0,0,1)`。普通 BCE 边际为 `(0.6,0.6,0.4)`。

| 排序 | 第一种 mask AP | 第二种 mask AP | context 平均 AP |
|---|---:|---:|---:|
| r1,r2,r3 | 1 | 1/3 | 0.73333 |
| r3,r1,r2 | 7/12 | 1 | 0.75000 |

已用 `average_precision_score` 单测核验。正确概率目标不保证最佳实例 AP，尤其各上下文阳性数不同；这个反例不是本数据 AP 差距的定量解释，也不意味着普通 BCE 一定优于当前 loss。

## 7. 输入函数与 observed sequence 合同

源码中 outer product 的 latent axes 显式对称化、两侧 residue projection/readout 共享，row/column pooling 复用同一函数。RCSB loader 由 UID 读取固定 ESM-C/PS，chain index 置零，position 重新编号为 `0..L-1`。装配名称、实验坐标、邻居角色不进入 forward。需要 eval 模式才能排除 dropout 引入的随机差异。

`prepare_sources` 按 `label_seq_id` 排序后只将实际保留下来的有重原子残基拼入序列。完整 `residue_map` 保留 author numbering、insertion code 和 `label_seq_gaps`，但这些 gap 不作为当前 ESM-C、预测单体和 TCN 的输入。

这符合既定 observed-sequence 实验合同，不是本轮发现的 label/index 错位。它可能把内部未观察片段两侧残基视为连续 token，并让预测折叠使用该拼接序列。要判断性能影响，必须量化 gap 分层或进行独立对照，不能凭索引一致宣布语义无风险，也不能擅自改回 UniProt 序列。

mmCIF 的 [label_seq_id 定义](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_atom_site.label_seq_id.html)提供 polymer 位置基准。内部跳号不能直接给出 terminal missingness 或完整实验构建体长度。

### 7.1 冻结 checkpoint 的实测，而非仅源码推断

作业 736734 已完成，用时 2 分 53 秒；epoch-3 checkpoint SHA256 为 `e445f2a710ce676d10b2bca73659c8593963c61c78c5d40d83b9d80e098bb1f1`。

从 validation 用固定 pair-ID hash 选 4 组同序列、4 组异序列的不同序列对，长度均 <=256 aa。严格加载冻结源码和非零 partner readout，FP32、eval 模式，实际执行整个模型：

| 检查 | 8 组内最大误差/结果 |
|---|---:|
| A/B 交换后的对应 logits | 0 |
| 同一次输入重复推理的 probability | 0 |
| 同序列 pair 两侧 probability | 0 |
| 同序列 pair 两侧实际 model-input tensor hashes | 全部相同 |
| 与另一个更长真实 pair 组成 batch 后的 probability | 0.0000129044 |

两侧 padding 宽度均实际增加，donor 与原序列对不同。没有观察到粗大分支不对称或 batch 污染；这不是全量、长链或 AMP 的严格数值等价证明，尤其不能保证任何临界 tie 的 AP 永不变化。源模块哈希与实际输入张量哈希在 `input_contract.json` 中。

### 7.2 完整 validation 的内部 gap 统计

逐一读取 14,514 个 validation PDB 的冻结 tag source，先确认这些 PDB 与 train/test PDB 没有交集。统计仅针对 ready pair 中实际使用的 source/asym。JSON 文件可能同时包含该 validation PDB 内未选的 asym；它们不计入结果。

| 聚合单位 | 总数 | 存在内部 label_seq gap | 比例 |
|---|---:|---:|---:|
| PDB + source asym | 49,706 | 16,636 | 33.47% |
| Exact UID，任一对应 source 有 gap | 21,662 | 10,824 | 49.97% |
| Biological-assembly chain instance | 74,539 | 23,836 | 31.98% |
| Pair-direction record | 247,726 | 63,338 | 25.57% |

49,706 个 sources 中累计有 561,360 个内部跳过的 polymer positions；76 个 sources 保留非空 insertion code 残基。该统计没有完整构建体长度和首尾缺失信息。当前 UID 的 gap presence 在对应 sources 间没有混合 true/false，不代表 gap 的精确位置或宽度完全相同。

这证明 gap 风险不是极少数边缘样本，却**没有证明 gap 导致 AP 下降**。还需比较 PDBTM 同口径 gap 频率，以及 RCSB 内控制长度、阳性率和重复度后的 gap/no-gap 表现。不能因某组分数低就将它从评估中删去。

## 8. Feature-ready 覆盖漏斗

| 阶段 | Unique sequences |
|---|---:|
| ge50 split universe | 251,267 |
| 至少参与一个双方均 >=50 aa 的 pair | 243,822 |
| 完整特征 ready pair endpoints | 214,097 |

`251267-214097=37170` 的差额可拆成：

- 7,445：没有进入符合双方长度合同的 materialized pair endpoints。
- 25,964：冻结 readiness 表中 `missing_protenix`。
- 3,761：自身不在 missing-unique 表，但因所有可用 partner 被排除而不再出现于 ready pairs。

25,964 个直接缺失中，25,879 个是 50–99 aa；此前折叠队列仅覆盖 >=100 aa。另 85 个 >=100 aa 位于 train，其中 3 个 <1000 aa、82 个 >=1000 aa。这里确认的是缺失 raw-PS 资产，不把所有情况自动写成折叠 OOM。

| Split 元数据 | 缺 Protenix 50–99 aa | 缺 Protenix >=100 aa | 因 partner 排除的额外 endpoints |
|---|---:|---:|---:|
| Train | 21,276 | 85 | 3,374 |
| Val | 2,426 | 0 | 203 |
| Test，资产元数据而非标签 | 2,177 | 0 | 184 |

因此当前模型不能按“所有 ge50 chains 都已用于训练”呈现。正式训练的是 feature-ready 子集；这一点应在论文方法主文中明确。

## 9. 下一步，不提前宣布原因

1. **冻结预测的 component-grouped 交叉校准。** 先复用共同 validation 的概率 NPZ；全 native val 的逐链 CSV 只有摘要，不能凭它伪造 residue logits。拟合正斜率 Platt，仅报告 out-of-fold CE/Brier/bias；每折严格单调保证折内和链内排序不变，但不同折校准器合并后的 pooled AP 不保证不变。
2. **统一统计单位的 val loss。** 保留旧日志，在离线统计中并列固定 chain/pair 平均的 unweighted CE 和实际 weighted CE。不可将旧 loss 静默替换后混画同一曲线。
3. **采样 x 目标 2x2 小型受控训练，尚未启动。** 当前 instance vs 等权 sequence pair，当前复合 loss vs 明确定义的普通 BCE。组内保留原 context 频率，不删多 mask。固定每 optimizer update 的抽样 pair 数，用 microbatch 梯度累积处理长度，不让 packing 再引入未知外层权重；固定 steps、LR schedule、初始化和特征，同时记录 residue/pair-cells 与时间。
4. **条件信息检查。** 在同一 query 的不同真实 partner 上比较预测差异与 mask 差异，不用随机 partner 替换量单独证明 specificity。若训练诊断子集 AP 也低，才更支持进一步增加交互容量。
5. **Gap/接触边界敏感性。** 在 frozen val 分组后量化差异，不用 test 挑解释，不重新定义已冻结标签。

目前更有依据的表述：RCSB 原生验证混合了接触角色异质性、不同统计权重与概率目标；这些已得到具体量化，但改善方向仍需受控实验，不应直接宣布“重复数据有害”或“扩充数据导致退步”。

## 10. 证据与复核

- [本轮 JSON 证据](../data/rcsb_ppis_review_20261019/objective_audit/)
- [目标函数审计代码](../scripts/analysis/audit_pair_objective.py)
- [完整 mask/装配上下文审计](../scripts/analysis/audit_repeated_pair_labels.py)
- [输入数值 probe](../scripts/analysis/audit_pair_input_contract.py)
- [Observed gap 审计](../scripts/analysis/audit_observed_sequence_gaps.py)
- 每份 objective JSON 包含冻结 trainer 文件 SHA256、两个核心 class AST SHA256、config SHA256、所选 manifest rows SHA256。
- Mask 审计的 row hash 是排序后的整列表 JSON；objective/gap 审计是原顺序逐行 canonical JSON。两种 digest 定义不同，不能直接要求数值相同；gap 与 objective 的 val 行摘要一致。
- 全 train mask CSV 留在 HPC3 `.../diagnostics/objective_audit_v2_20261019/train_label_diversity/`，轻量 summary 同步仓库；不上传完整特征、结构或 checkpoint。
- Validation source-gap 与 ordered-group 逐组 CSV 已压缩归档到证据目录，可重算摘要；合计定向单测 21 项通过。

```bash
/home/husrcf/anaconda3/envs/AIAA/bin/python -m pytest -q \
  tests/test_pair_objective_audit.py tests/test_repeated_pair_labels.py \
  tests/test_pair_input_contract_audit.py tests/test_observed_sequence_gap_audit.py
```
