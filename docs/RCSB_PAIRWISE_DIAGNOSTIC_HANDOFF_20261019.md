# RCSB Pairwise PPIS：进展与独立诊断交接

更新：2026-10-19。运行快照：09:44:34，UTC+08:00。

**代码仓库：[sunnylab-cb/TMINTER3](https://github.com/sunnylab-cb/TMINTER3)**

克隆地址：https://github.com/sunnylab-cb/TMINTER3.git

本地目录：/home/husrcf/Code/ProtBind/pypropel_unibio/TMINTER3

报告整理前本地 HEAD：6001d5c8df6979e8af961aff4de988020d3a8826。

本文是诊断交接，不是已确定的因果解释。新增分析脚本和报告应与运行时冻结代码分别追溯；不能把上述 HEAD 当作所有历史训练的代码哈希。大型训练张量、结构和完整标签在集群路径中，不应假定它们已上传 GitHub。

[轻量训练与诊断证据包](../data/rcsb_ppis_review_20261019/README.md)包含训练配置、epoch 指标、训练 UID 清单和小样本 manifest；它不是完整可运行的训练数据集。大型诊断 CSV 提供 gzip 版本。

## 1. 要解决的问题

我们将原先以 PDBTM 膜蛋白为主的 partner-specific residue-level PPIS 训练扩展到全 RCSB。新数据更多，但原生验证 AP 较低，验证损失也较高。

用户提出的重要反问：

> 同一序列对在不同构象或装配环境下出现不同界面，为什么不能像构象扰动、NMA 等做法一样，成为有益的数据增强或正则化？为什么被怀疑是性能下降原因？

**当前不能回答“它已经被证明有害”。** 已有证据支持：

1. 新旧验证集的难度、阳性率和重复权重明显不同。
2. RCSB 验证集中的少量重复序列对，对 instance-macro 指标贡献很大。
3. 相同 query/partner 序列输入确实对应不同界面标签，但这可以是合法的多环境监督。
4. 缺少装配姿态的输入不能唯一指定每个实验实例的界面，但仍可学习跨环境的边际概率。
5. 尚未通过受控重训分离采样权重、多标签监督、结构特征差异和概率尺度的作用。

近期诊断只使用 train/validation 信息，没有用 test 标签选择解释或方案。没有因本报告修改正在运行的训练、标签或 split。

## 2. 任务与模型合同

### 2.1 预测对象

- 输入：一对蛋白链 A、B 的序列特征和各自预测单体结构描述符。
- 输出：A 中与 B 接触、B 中与 A 接触的逐残基概率。
- 标签：biological assembly 中，5.5 Å 重原子接触投影到两条链。
- 序列基准：实际结构中观察到的残基；保留 insertion code 对应的残基映射。
- 当前数据含至少一个接触的链对，不是任意两条蛋白是否结合的二分类。
- 这是 pair-specific 任务，**不是之前 M18 的 single-chain / all-partner-union 任务**。旧 M18 的高 AP 不能直接当作同任务基线。

### 2.2 当前架构

~~~text
Chain A: ESM-C layers 12/24/36 concat (3456D) + monomer PS (90D)
Chain B: ESM-C layers 12/24/36 concat (3456D) + monomer PS (90D)
                  |
        shared residue encoder
        projection + 2-layer gated-depthwise TCN
                  |
        low-rank outer-product interaction
        pair representation + row/column pooling
                  |
        partner-conditioned residue classifiers
                  |
              p(A|B), p(B|A)
~~~

核心设置：model dimension 256，hidden 512；TCN kernel 7、dilations 1/2；outer-product rank 8、hidden 32、bottleneck 64、pooling softmax temperature 0.5；pair convolution layers 0。**本报告的 RCSB 运行是 outer-product 模型，不是完整 AlphaFold Pairformer。**

| 项目 | PDBTM S5 | RCSB |
|---|---|---|
| 训练 pair instances | 16,103 | 1,150,993 |
| 验证 pair instances | 4,030 | 123,863 |
| ESM-C | 相同层号、3456D、final-layernorm | 相同规格 |
| 预测单体结构 | ESMFold | Protenix Mini ESM v0.5.0 |
| PS 输入 | 90D | 89 个标准化 scalar + availability |
| ESM2 contact graph | 有；另有关闭 graph 的受控重训 | 无 |
| 参数量 | graph: 2,756,623 | no-graph: 2,426,895 |
| Positive class weight | 6.93964 | 10.38067 |
| Label smoothing / ratio penalty | 0.05 / 0.25 | 0.05 / 0.25 |
| Scratch seed | 49 | 49 |
| 初始学习率 / weight decay | 1e-4 / 0.1 | 1e-4 / 0.1 |
| 原始 epoch 预算 | 20 | 15 |

这不是仅改变训练样本量的单变量实验。预测结构来源、图特征、类别权重、数据域和学习率进程均不同。

同 epoch 也不等于同计算量：S5 每 epoch 约 2,004 个训练 batch，RCSB 1M 约 133,609 个，约 66.7 倍。比较曲线应同时看 optimizer steps、处理的 pair/residue 数及 wall time。

## 3. 最新训练进度

快照时 HPC3 三个作业均 RUNNING；只列已完整写出的 epoch。

| 作业 | 配置 | 已完成 epoch | Train loss | Val loss | Chain-macro AP | Pooled AP | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| 728963 | 1M pair-cell budget | 5 | 0.657602 | 0.756929 | 0.295268 | 0.259492 | 0.762007 |
| 729720 | 4M pair-cell budget | 6 | 0.637479 | 0.754517 | 0.303907 | 0.267186 | 0.765073 |
| 730850 | 8M pair-cell budget | 5 | 0.649989 | 0.752389 | 0.307944 | 0.270712 | 0.764436 |

1M/4M/8M 是动态 batch 的 pair-cell 预算，不是参数量。大 batch 有小幅排序改善，但尚无证据表明仅继续训练就能消除原生验证差距。AP 指 average precision，不应与 PR 曲线梯形积分混称。

## 4. 两个看起来矛盾、实际可同时成立的观察

### 4.1 各自完整原生验证集：RCSB 较低

统一导出 epoch 3。PDBTM 此处是匹配 S5 的 graph-control 重训，不是历史原始 checkpoint。

| 指标 | RCSB | PDBTM S5 graph control |
|---|---:|---:|
| Pair instances | 123,863 | 4,030 |
| Directional chain records | 247,726 | 8,060 |
| Unique unordered exact-sequence pairs | 22,274 | 1,098 |
| Instance / directional-chain macro AP | 0.291519 | 0.432344 |
| Equal unique-sequence-pair macro AP | 0.398481 | 0.441163 |
| Macro positive fraction | 0.109488 | 0.155845 |
| Unweighted macro CE | 0.61898 | 0.54628 |
| Macro mean probability minus prevalence | +0.32358 | +0.23155 |

Unique-pair AP：先在同一无序序列对的所有实例和方向内平均 AP，再跨序列对等权平均；没有合并标签或丢弃不同构象。

差距从 0.140825 缩为 0.042682，说明聚合权重很重要，但不是“重复解释了多少性能下降”的因果分解。两种聚合回答不同问题，不能因为后一种更好看就替换正式主指标。

### 4.2 相同标签、相同链对的共享验证子集：RCSB 排序更好

冻结交集：100 PDB、351 pairs、702 个方向、371,113 个残基；pooled 阳性率 0.100835。已排除与任一训练集存在 exact sequence / PDB 重叠的项目，六次评估标签和长度哈希一致。

| 模型 | Epoch 1 macro AP | Epoch 3 macro AP | Epoch 3 pooled AP | Epoch 3 AUROC | Epoch 3 unweighted macro CE |
|---|---:|---:|---:|---:|---:|
| PDBTM graph control | 0.363596 | 0.365030 | 0.303292 | 0.798687 | 0.517478 |
| PDBTM no-graph control | 0.377832 | 0.358582 | 0.295862 | 0.796821 | 0.520831 |
| RCSB no-graph | 0.409249 | 0.423097 | 0.336430 | 0.842723 | 0.573215 |

RCSB 相对 graph control 的 epoch-3 macro AP 差为 +0.05807；探索性 PDB-cluster bootstrap 95% CI 为 [0.04217, 0.07436]。

限制：这是共同交集、单 seed、不是新 sealed holdout；PDB bootstrap 未覆盖全部跨 PDB 同源依赖。它不能代替全 RCSB 评估，但反对“扩充数据在所有地方都使模型变差”的表述。

## 5. SI30 为什么没有消除重复？

~~~text
观察到的结构链序列
  -> exact sequence UID：特征只提取一次
  -> MMseqs SI30 + 双侧 coverage >= 0.8 + 同 PDB 分组
  -> 连通分量整体分入 train / val / test
  -> 每个 assembly 的具体链实例重新生成 pair-specific 标签
  -> 同一序列对可以有多个结构上下文和标签模式
~~~

当前用 cluster mode 1；SI30 是检测到的相似性边隔离，不是每个簇只保留一个训练样本，也不是穷举 Biopython pairwise SI 无泄漏证明。

需要分清 split isolation、feature deduplication、example balancing。前两者存在，第三者没有组均衡：动态 batch sampler 每个 epoch 遍历全部实例。

## 6. 重复、多标签与界面稀疏性

### 6.1 验证集重复权重

| 序列对 pair-instance multiplicity | RCSB directional records | RCSB AP | PDBTM directional records | PDBTM AP |
|---|---:|---:|---:|---:|
| 1 | 27,472 | 0.43170 | 1,020 | 0.45663 |
| 2-5 | 30,250 | 0.36554 | 2,194 | 0.42803 |
| 6-20 | 43,240 | 0.30714 | 3,500 | 0.43674 |
| >20 | 146,764 | 0.24542 | 1,346 | 0.40954 |

RCSB 中仅 884 个 unique sequence pairs 贡献了 59.24% 的 directional records。这里证明的是验证指标权重集中，不是已证明训练梯度被这些相同组支配。

阳性率 <=5% 的稀疏界面占 RCSB 记录的 33.94%，PDBTM 为 14.39%。AP 随 prevalence 改变。RCSB 的 10%-20% 阳性率区间内，singleton AP 0.49741、>20-repeat AP 0.29035；但 <=5% 区间两者几乎相同。宽分层仍未控制 family、几何和界面类型。

### 6.2 完整 mask 审计，不只是 positive-count 比较

| RCSB validation 项目 | 数量 |
|---|---:|
| Ordered exact (query UID, partner UID) groups | 37,049 |
| 包含多个 directional records 的组 | 16,839 |
| 存在多个不同 positive masks 的组 | 12,484 |
| 属于不同-mask 组的 records | 216,006 / 247,726 = 87.20% |
| 不同 ordered-input + mask 组合 | 82,376 |
| 超出不同组合的额外出现次数 | 165,350 |

审计检查 observed-coordinate 对齐、contact 索引唯一性和范围，以及 contact 投影与左右标签完全一致。**这些完整 mask 数字全部来自 validation，不是 training。** 87.20% 不是错误率、矛盾残基比例或准确率上限；一个 mask 只差一个位置也计为不同。

例子：144 aa 的相同序列对有 3,600 个 directional records，来自 13 PDB，共 64 种 mask；最常见 mask 出现 762 次。另一个 114 aa 序列对仅在一个 PDB（6yf7）内就有 1,800 个方向记录和 45 种 mask。大型装配本身即可产生这种现象。

Query 和 partner 都按 exact UID 分组，因此不只是 A-B、A-C 中 B/C 序列不同的正常 partner specificity。替代结合位姿、对称副本、实验状态或装配定义都可能是真实原因，不应直接称为污染。

早期 positive-count screen 显示 PDBTM validation 的相同序列 partner 比例更高（80.15%，RCSB 60.18%），且两者异质 positive-count 组覆盖比例相近。因此“homomer 多，所以 RCSB 差”这个简单解释不成立。

## 7. 多构象监督为什么可能有用，而不一定有害？

设模型输入是 x，结构上下文是 c，标签为 y(x,c)。固定预测单体特征后，不同 c 可能有相同的 x。

对普通、不加权 binary cross-entropy，有精确恒等式：

$$
\frac{1}{m}\sum_{c=1}^{m}\mathrm{BCE}(p,y_c)
=\mathrm{BCE}(p,\bar y),\qquad
\bar y=\frac{1}{m}\sum_{c=1}^{m}y_c.
$$

同一个输入用多种 hard labels 训练，可以合法学习在被采样上下文中接触的频率。它不是原则上错误的监督，也不自动比 mean soft label 更差。仅把 hard masks 换成平均 soft labels，若其余权重相同，并不改变这个基础目标。

| 情况 | 可能作用 | 本项目需核查 |
|---|---|---|
| 输入坐标与相应标签一起变化 | 学习对构象变化的响应 | 按 UID 缓存单体后是否丢失输入变化 |
| 输入不变、标签随上下文变化 | 学习边际概率；避免过拟合单个界面 | 目标接受概率，还是要求特定 assembly 的精确 mask |
| 相同输入、相同标签大量重复 | 改变该模式的风险权重 | 频率是否代表目标分布，而非实验/装配采样偏好 |
| 同组不同 mask 频率不均 | 学习观测频率加权的 mixture | PDB 出现次数不等于热力学构象占有率 |

NMA 是产生/解释结构变化的工具，不天然等于有益正则化。若增强坐标作为输入、标签一致变化，它与“所有构象共用单体输入，但监督不同 mask”不是同一种实验。

当前 loss 也不是简单 BCE：per-chain positive-weighted CE（0.05 smoothing）加 0.25 ratio penalty，再对方向及 batch 聚合。仅在简化的 weighted BCE 中，最优概率为：

$$
p^*=\frac{wq}{wq+1-q}.
$$

这提示加权输出不应默认等于未加权接触频率，**不能把逆变换直接当作当前模型已验证的校准器**。Weighted-mean 的 chain 归一化、smoothing、ratio 项和动态 batch 权重都要纳入。

应区分统计权重不合适与真实上下文多样性，不能删掉后者来掩盖前者，也不能由多-mask 计数推导可达 AP 上限。

## 8. 已排查与未排除

| 排查 | 证据 | 不能据此声称 |
|---|---|---|
| Train/val exact PDB、sequence overlap | 当前导出为 0 | 已完成所有同源 SI 穷举证明 |
| PS / ESM-C 数值尺度 | 各取 128 条 train/val，有限值、尺度相近 | 全量语义、结构质量、映射均完美 |
| Partner 是否被忽略 | 256 val pairs，正确 partner AP 0.288602，替换后 0.277589 | 已充分利用复杂 partner interaction |
| PS 是否被利用 | 冻结模型 PS 置零后 AP 0.220479 | 等价于从头训练的结构消融 |
| ESM2 graph 缺失 | S5 graph/no-graph 20 epochs；各自 val 最优 AP 0.450803/0.445102 | Graph 完全无用或统计等价 |
| 长链是否唯一原因 | 普通长度区间也有差距 | 长度和结构质量无需继续控制 |
| 概率偏高 | 未加权 CE、mean(p)-prevalence 与 weighted loss 不一致 | Calibration 是全部 ranking gap 的原因 |

推理干预不是重训练消融。没发现粗大数值错误，不等于所有实现正确。

## 9. 原始文献的启发

| 来源 | 相关内容 | 适用边界 |
|---|---|---|
| [Abramson et al., AlphaFold 3, Nature 2024](https://www.nature.com/articles/s41586-024-07487-w) | 评估在 chain/interface clusters 内聚合后跨 cluster 汇总；不同数据类型采样也是训练设计的一部分 | 支持检查权重，不证明本项目被重复伤害，也不证明换 Pairformer 即可解决 |
| [Chen, Dobriban & Lee, JMLR 2020, A Group-Theoretic Framework for Data Augmentation](https://www.jmlr.org/papers/v21/20-163.html) | 在分布不变性等假设下，增强可通过平均降低估计方差 | 不支持任意改标签，也不否定多状态概率监督 |
| [Cui et al., CVPR 2019, Class-Balanced Loss Based on Effective Number of Samples](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html) | 重叠样本边际有效信息递减，提出重加权 | 原问题是类别不平衡；不能直接把公式套到 sequence-pair groups |
| [Menon et al., ICLR 2021, Long-tail Learning via Logit Adjustment](https://arxiv.org/abs/2007.07314) | 类别频率与目标风险影响 logits 和决策 | 提醒区分排序和概率含义，不是当前复合 loss 的校准证明 |
| [Müller, Kornblith & Hinton, NeurIPS 2019, When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) | Soft targets 影响泛化、校准和表示 | 应与 positive weight、ratio 项分开验证 |
| [Oliwa & Shen, Bioinformatics 2015, cNMA](https://pmc.ncbi.nlm.nih.gov/articles/PMC4765865/) | 用 encounter-complex normal modes 研究相互作用中的构象变化 | 是构象建模依据，不是相同输入多标签训练必然有益的证明 |

## 10. 最小判别实验建议，尚未启动

1. **Training-only 权重审计。** 统计 sequence pair、input+mask、PDB/assembly 重复；量化动态 batch 与 CE normalization 后权重。先确认 validation 现象是否也支配 training。
2. **保留上下文的组均衡。** 原始 instance sampling 对照“先抽 sequence pair，再抽 context”；保留全部 mask，不任取代表、不改 union。固定 optimizer steps/学习率进程，另报 wall time 和 pairs/residues，不能只比较 epoch。
3. **目标函数等价性控制。** 在简单 BCE 下验证 hard-mask sampling 与同权 soft target 等价，再单独加入 class weight、smoothing、ratio 项；结果违背预期时查归一化/采样。
4. **概率目标控制。** 同采样下分别调整 positive weight、smoothing、ratio；报告 AP/AUROC、未加权 log loss、概率偏差。校准只用 validation，不动 test。
5. **分离歧义与重复。** Single-mask / multi-mask 组按长度、阳性率、family、assembly size 匹配，比较预测和 mask 相似度；不能把组内频率当 held-out oracle 上限。
6. **补输入或换目标留到后续。** 特定装配预测需要合法可用的复杂体上下文；ensemble propensity 则需明确概率/评分合同。不能从目标接触标签反推输入姿态制造泄漏。

保留原生 instance-macro、sequence-pair-macro 和 pooled 指标，使用同一验证集，不删除难例；再做多 seed 和依赖组 bootstrap。

## 11. 文件、运行与数据位置

本地相关报告：

- [当前 split 合同](CURRENT_SPLIT_CONTRACT.md)
- [总体诊断、共享验证和特征干预](RCSB_GENERALIZATION_DIAGNOSTIC_20261018.md)
- [曲线与 graph control 更新](RCSB_TREND_UPDATE_20261019.md)
- [完整验证集分层](RCSB_STRATIFIED_VALIDATION_20261019.md)
- [SI30、去重、mask 审计](SI30_VS_REDUNDANCY_AND_LABEL_DIVERSITY_20261019.md)

仓库相对路径：

~~~text
artifacts/diagnostics/20261019/rcsb_epoch3/{summary.json,strata.json,directional_metrics.csv}
artifacts/diagnostics/20261019/pdbtm_epoch3/{summary.json,strata.json,directional_metrics.csv}
artifacts/diagnostics/20261019/label_diversity/{summary.json,ordered_sequence_groups.csv}
scripts/analysis/audit_repeated_pair_labels.py
scripts/analysis/summarize_validation_strata.py
scripts/training/export_validation_strata.py
scripts/training/train_esm_pair.py
src/ppcbind/data/rcsb_pair_dataset.py
src/ppcbind/models/esm_pair.py
~~~

HPC3 根目录：

~~~text
/data/user/shuang886/Folding/Dataset/derived/tminter3_rcsb_ppis_ge50_v1
~~~

相对此根目录：

~~~text
training/ppis_ge50_v1/ready_v1/ready_pair_manifest.csv
training/ppis_ge50_v1/ready_v1/normalization_stats.json
training/ppis_ge50_v1/pairs/labels/shard_*.jsonl
TMINTER3-ESMC/features/unique/
protenix/features/complete_raw_v1/
_hpc3_training/20261017_v1/code/
_hpc3_training/20261017_v1/runs/{full,large_batch4m,large_batch8m}/
~~~

HPC2 对照根目录：/hpc2hdd/home/shuang886/jhspoolers/PPC

运行：runs/pdbtm_S5_graph_control_{graph,no_graph}_seed49_20261018

关键 SHA256：

~~~text
RCSB native epoch-3 checkpoint:
e445f2a710ce676d10b2bca73659c8593963c61c78c5d40d83b9d80e098bb1f1
PDBTM graph-control epoch-3 checkpoint:
c7a1c2f496dc7c12a41a2afbf26bdfb6099cd79d9b1d40e0855786f177610375
Final validation-mask audit row hash:
8fe36cb7435d29e909a3372167d15db05d20ae8a8358e16c7a6326e8457298f0
~~~

公开复核优先需要配置、split/manifest schema、统计摘要和脚本；完整训练张量不适合普通 Git。各来源数据是否可再分发应按许可分别确认。

本次轻量包中的 smoke manifests 仅用于格式核对，不是正式训练 cohort。原始 split 为 251,267 条 >=50 aa unique sequences，但当前 feature-ready train/val/test 分别为 170,465 / 21,662 / 21,970 个 unique endpoints；特征准备造成的排除不能忽略。已评估的原生验证中没有 <100 aa 记录，不能把 ge50 源数据名理解为全长范围已覆盖。

## 12. 可直接转交独立研究者的 Prompt

> 请独立诊断 TMINTER3 的 RCSB pairwise PPIS 扩展实验。仓库：https://github.com/sunnylab-cb/TMINTER3
>
> 首先阅读本报告，不要预设“重复有害”或“扩充必然更好”。任务是给定两条序列及预测单体结构，预测 partner-specific 5.5 Å biological-assembly heavy-atom contact residues，不是单链 all-partner-union，也不是 P-L。
>
> RCSB 在完整 native validation 上 epoch-3 macro AP 为 0.2915，PDBTM S5 graph control 为 0.4323；equal unique-sequence-pair AP 为 0.3985/0.4412。但在 100 PDB 的相同标签共同 validation 上 RCSB 为 0.4231，PDBTM 为 0.3650。数据域、结构来源、graph、class weight 和训练步数均不同。
>
> RCSB validation 的 247,726 个方向记录中，87.2% 属于相同 query/partner exact sequences 但存在多个 mask 的组；不是错误率，也不是 training 完整审计结果。输入按 UID 共用 ESM-C 与单体结构，缺少 assembly pose。SI30 做跨 split 隔离，不做实例降采样。
>
> 请回答：
> 1. 什么条件下多构象监督是有效正则化或概率学习，什么条件下与实例级目标失配？与 NMA 增强的关键区别是什么？
> 2. 如何分离采样偏好、相同 input+label 重复、真实多界面、稀疏阳性率、结构质量和 class-weight calibration？
> 3. 普通 BCE 下 hard-mask 平均与 soft target 等价；实际 per-chain weighted normalization、smoothing、ratio loss 会如何改变结论？
> 4. 证据能排除什么，不能排除什么？有无更优先的实现或评估问题？
> 5. 给出最有区分力的 2-3 个受控实验，明确不同结果如何支持/反驳假设，优先复用现有输出。
>
> 引用原始论文/官方实现，分开事实、推断和假设。不要删除难例、改 test 或合并 union 来提高分数；不要把 validation mask 频率当泛化上限。只读报告列出的 loader、loss、sampler 和诊断脚本，避免扫描全部数据目录。先给解释与建议，不直接启动作业或修改训练。
