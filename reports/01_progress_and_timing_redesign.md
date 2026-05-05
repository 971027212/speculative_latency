# 阶段报告 01：从 RTT 验证转向阶段时间占比

## 本阶段目标

本阶段把实验目标从“RTT（Round Trip Time，往返时延）是否抵消 speculative decoding（推测解码）加速收益”，调整为“系统测量不同推理方法在完整解码过程中的阶段时间占比”。

大白话说：之前我们只是在问“网络等待会不会拖慢”；现在要问得更细：“每种方法到底把时间花在哪里，是草稿生成慢、目标模型验证慢、缓存更新慢，还是上传等待慢。”

## 已完成工作

- 已完成基础项目结构和远程服务器环境。
- 已跑通 `smollm-local` 本地模型路径。
- 已验证 Hugging Face `assistant_model（辅助模型）` sanity check 可以跑。
- 已实现并验证 `full-prefix vanilla speculative decoding（完整前缀普通推测解码）` 的正确性。
- 已跑出 RTT sweep（RTT 扫描）结果，确认 `upload_wait_share（上传等待占比）` 会随 RTT 增大而上升。
- 已新增方法对比主线：
  - `target-only（只用目标模型）`
  - `vanilla-spec（普通线性推测解码）`
  - `specinfer-simplified（简化 SpecInfer）`
  - `dsd-adaptive-draft（DSD 风格自适应草稿策略）`

## 技术改动说明

### 1. 统一计时桶

新增和统一以下 timing buckets（计时桶）：

| 字段 | 中文解释 | 用途 |
| --- | --- | --- |
| `prefill_time` | 预填充时间 | 处理 prompt（提示词）的初始前向计算 |
| `draft_generate_time` | 草稿生成时间 | draft model（草稿模型）生成候选 token 的时间 |
| `draft_structure_time` | 草稿结构构建时间 | SpecInfer 构建 token tree（token 树）的 Python/结构成本 |
| `upload_wait_time` | 上传等待时间 | 模拟端侧 draft tokens 上传到云端的等待 |
| `target_verify_time` | 目标模型验证时间 | target model（目标模型）验证候选 token 的时间 |
| `posterior_accept_time` | 后验接受时间 | 判断哪些 draft token 被接受的时间 |
| `cache_update_time` | 缓存更新时间 | 更新 KV cache（键值缓存）或输入 token 的时间 |
| `sampling_time` | 采样/argmax 时间 | 从 logits（输出分布）里取下一个 token 的时间 |
| `wasted_branch_time_or_tokens` | 无效分支成本 | SpecInfer 中没有进入最终输出的分支 token 数或成本 |
| `total_decode_time` | 总解码时间 | 整个 decode（解码）过程总时间 |

### 2. 新增统一方法注册表

新增 `edge_specdec/method_registry.py`。

作用：

- 每种方法统一通过一个 runner（运行器）调用。
- 后续增加论文方法时，不再为每个方法单独写一套实验脚本。
- 保证所有方法输出相同字段，方便横向比较。

### 3. 新增方法时间占比实验脚本

新增 `scripts/05_method_timing_experiment.py`。

它负责：

- 加载 target model（目标模型）和 draft model（草稿模型）。
- 对同一批 prompts（提示词）运行多个方法。
- 统一写出 raw CSV（原始结果表）。
- 自动检查 lossless decoding（无损解码）方法是否和 `target-only greedy（目标模型贪心解码）` 输出一致。

### 4. 新增分析和画图脚本

新增 `scripts/06_analyze_method_timing.py`。

它负责：

- 生成 summary CSV（汇总表）。
- 生成 stage share CSV（阶段占比表）。
- 生成 stacked bar chart（堆叠柱状图）。

## 当前方法口径

### target-only

`target-only（只用目标模型）` 是所有 lossless speculative methods（无损推测方法）的正确性基准。

### vanilla-spec

`vanilla speculative decoding（普通线性推测解码）` 保留原来的线性 draft tokens（草稿 token）流程：

1. draft model 生成一段候选 token。
2. target model 一次验证。
3. posterior accept（后验接受）决定接受多少 token。

### specinfer-simplified

`SpecInfer` 原论文的核心是 token tree（token 树）和 parallel verification（并行验证）。

第一版只做 simplified SpecInfer（简化 SpecInfer）：

- 构建一个小 token tree。
- 用 batched target verification（批量目标验证）验证多个候选路径。
- 统计 wasted branch tokens（无效分支 token）。

注意：这不是完整 SpecInfer serving system（服务系统）复现，不能直接声称复现论文速度，只用于第一阶段测“树状草稿方法的阶段成本结构”。

### dsd-adaptive-draft

`Decoding Speculative Decoding` 这里按 draft strategy（草稿策略）处理。

第一版实现为 adaptive draft length（自适应草稿长度）：

- 如果上一轮 draft tokens 全部被接受，就增大 `draft_k`。
- 如果接受很少，就减小 `draft_k`。

这是 DSD 风格的草稿使用策略，不是完整论文复现。

## 关键实验命令

快速跑第一版方法占比实验：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair smollm-local \
  --implementation full-prefix \
  --max-new-tokens 32 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 1 \
  --warmups 1 \
  --rtt-ms 0 20 100 \
  --output results/method_timing_quick.csv
```

生成汇总表和图：

```bash
python scripts/06_analyze_method_timing.py \
  --input results/method_timing_quick.csv \
  --summary-output results/method_timing_summary_quick.csv \
  --share-output results/method_timing_stage_shares_quick.csv \
  --plot-output results/method_timing_stage_shares_quick.png \
  --plot-rtt-ms 0
```

## 当前不足

- `specinfer-simplified` 是简化版，不是完整论文系统复现。
- `wasted_branch_time_or_tokens` 当前更接近 token count（token 数），不是严格时间。
- KV-cache（键值缓存）版本还需要继续验证稳定性，第一阶段建议先用 `full-prefix（完整前缀）` 跑通口径。
- 当前只用 `smollm-local`，后续还需要第二组模型增强说服力。

## 下一步计划

1. 在服务器上拉取最新代码。
2. 跑 `05_method_timing_experiment.py` 的 quick 版本。
3. 检查四种方法是否都和 target-only 输出一致。
4. 查看 `method_timing_stage_shares_quick.csv` 和堆叠柱状图。
5. 如果口径正确，再增加 repeats（重复次数）、RTT 档位和第二组模型。

## 附录：术语表

- `speculative decoding`：推测解码。
- `draft model`：草稿模型，小模型，负责先生成候选 token。
- `target model`：目标模型，大模型，负责验证候选 token。
- `target verification`：目标模型验证。
- `posterior acceptance`：后验接受，决定接受多少草稿 token。
- `KV cache`：键值缓存，Transformer 解码时缓存历史 token 的 key/value。
- `token tree`：token 树，多分支候选 token 结构。
- `parallel verification`：并行验证，一次验证多个候选路径。
- `RTT`：Round Trip Time，往返时延。
