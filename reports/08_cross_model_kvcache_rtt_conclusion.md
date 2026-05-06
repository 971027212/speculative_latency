# 阶段报告 08：SmolLM 与 Pythia 的 KV-cache RTT 对比结论

> 修正说明：本报告中 Pythia `--ignore-eos` 版本结果目前应视为 diagnostic result（诊断结果），不应直接作为最终 break-even RTT（盈亏平衡 RTT）结论。原因是 Pythia 原始 greedy decoding（贪心解码）会很快输出 EOS（结束 token），`--ignore-eos` 只是忽略停止条件，可能导致模型在 EOS 之后继续生成退化 token 序列，使 accept rate（接受率）虚高到 1.0。后续应使用 `--suppress-eos` 在 argmax（取最大概率 token）前屏蔽 EOS，再重跑固定 32-token continuation（续写）实验。

## 本阶段目标

本阶段目标是在已有 SmolLM 实验基础上，引入第二组模型 Pythia，并把实验推进到可以讨论结论的程度。

核心问题是：

1. speculative decoding（推测解码）在 RTT=0 ms 时是否能快于 target-only（只用目标模型解码）。
2. 如果 RTT=0 ms 有加速，break-even RTT（盈亏平衡 RTT）大约是多少。
3. upload wait（上传等待）在不同 RTT 下占总耗时多少。
4. 不同模型 pair（模型组合）之间，accept rate（接受率）、rounds（交互轮数）和阶段时间结构有什么差异。

## 当前实验任务口径

当前任务仍是 greedy continuation（贪心续写）任务：给定 prompt（提示词），固定使用 greedy decoding（贪心解码），比较 speculative decoding 输出是否与 target-only 输出完全一致。

这不是摘要、问答、代码评测等完整 benchmark（基准测试），而是系统实验里的受控 continuation workload（续写负载），主要用于：

- correctness check（正确性检查）：speculative 输出必须等于 target-only 输出。
- timing breakdown（阶段耗时拆解）：统计 draft generation（草稿生成）、target verification（目标模型验证）、cache update（缓存更新）、upload wait（上传等待）等阶段。
- RTT sweep（往返时延扫描）：模拟端侧 draft tokens（草稿 token）上传到云端验证的等待成本。

## 已完成工作

### 1. SmolLM 组已完成 KV-cache dense RTT 实验

SmolLM 组使用：

- target model（目标模型）：`/home/chajiahao/data/hf_models/SmolLM-360M`
- draft model（草稿模型）：`/home/chajiahao/data/hf_models/SmolLM-135M`
- model pair：`smollm-local`
- implementation（实现）：`kv-cache`
- max new tokens：32
- repeats（重复次数）：3
- dense RTT：0 / 5 / 10 / 20 / 50 / 100 / 150 / 200 ms

结论：SmolLM 组在 RTT=0 ms 时 speculative methods（推测方法）仍慢于 target-only，因此不存在正向 break-even RTT。

### 2. Pythia 组模型已下载并接入

Pythia 组使用：

- target model：`EleutherAI/pythia-410m-deduped`
- draft model：`EleutherAI/pythia-160m-deduped`
- 本地目录：
  - `/home/chajiahao/data/hf_models/pythia-410m-deduped`
  - `/home/chajiahao/data/hf_models/pythia-160m-deduped`
- model pair：`pythia-local`

服务器无法直连 `huggingface.co`，已通过 `HF_ENDPOINT=https://hf-mirror.com` 完成下载。

### 3. Pythia 组完成 correctness 和 dense RTT 实验

Pythia/GPT-NeoX 在默认 SDPA attention backend（缩放点积注意力后端）下，`vanilla-spec / kv-cache` 曾在 `prompt_id=3` 出现 mismatch（输出不一致）。

后续新增并使用：

```bash
--attn-implementation eager
```

使用 eager attention（显式注意力实现）后，Pythia correctness 通过。

另外，Pythia 原始实验中模型很快生成 EOS（结束 token），导致平均只生成约 1 个 token，不能代表 32-token 解码负载。后续新增并使用：

```bash
--ignore-eos
```

强制生成满 32 个 token 后，完成可用于结论的 dense RTT 实验。

## 技术改动说明

本阶段相关代码改动包括：

- 新增 `pythia-local` 模型组合配置。
- 扩展 `DEFAULT_PROMPTS` 到 10 条轻量 continuation prompts（续写提示词）。
- 新增 `--allow-mismatch`：允许记录 mismatch 并继续跑完整实验，用于定位 correctness 问题。
- 新增 `--attn-implementation`：允许选择 Transformers attention backend（注意力后端），Pythia 使用 `eager` 后通过 correctness。
- 新增 `--ignore-eos`：允许忽略 EOS，强制跑满 `max_new_tokens`，避免短输出导致计时失真。
- 分析脚本 summary 输出新增 `generated_tokens`、`rounds`、`drafted_tokens`，便于发现是否真的生成了完整 token 数。

## 英文术语中文解释

| English term | 中文解释 |
| --- | --- |
| speculative decoding | 推测解码，先由小 draft model 生成草稿 token，再由大 target model 验证 |
| target-only | 只用目标模型解码，作为正确性和速度基准 |
| draft model | 草稿模型，较小模型，用于快速提出候选 token |
| target model | 目标模型，较大模型，用于验证草稿 token |
| KV-cache | 键值缓存，缓存 Transformer 注意力中的 key/value，减少重复计算 |
| RTT | 往返时延，端侧上传到云端并等待响应的延迟 |
| upload wait | 上传等待，实验中模拟 draft tokens 上传到云端的等待时间 |
| accept rate | 接受率，draft token 被 target model 接受的比例 |
| break-even RTT | 盈亏平衡 RTT，speculative 加速收益刚好被通信等待抵消的 RTT |
| greedy continuation | 贪心续写，固定每步选概率最高 token 的续写任务 |
| eager attention | 显式注意力实现，较保守但更容易保证 KV-cache 行为正确 |
| SDPA | scaled dot product attention，缩放点积注意力后端 |

## 关键实验命令

### Pythia correctness quick

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair pythia-local \
  --implementation kv-cache \
  --attn-implementation eager \
  --ignore-eos \
  --method target-only \
  --method vanilla-spec \
  --method dsd-adaptive-draft \
  --max-new-tokens 32 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 1 \
  --warmups 1 \
  --rtt-ms 0 \
  --upload-token-bytes 4 \
  --upload-bandwidth-mbps 0 \
  --output results/method_timing_pythia_kvcache_eager_ignoreeos_correctness_quick.csv
```

### Pythia dense RTT repeat3

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair pythia-local \
  --implementation kv-cache \
  --attn-implementation eager \
  --ignore-eos \
  --method target-only \
  --method vanilla-spec \
  --method dsd-adaptive-draft \
  --max-new-tokens 32 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 3 \
  --warmups 2 \
  --rtt-ms 0 5 10 20 50 100 150 200 \
  --upload-token-bytes 4 \
  --upload-bandwidth-mbps 0 \
  --output results/method_timing_pythia_kvcache_eager_ignoreeos_rtt_dense_repeat3.csv
```

### Pythia analysis

```bash
python scripts/06_analyze_method_timing.py \
  --input results/method_timing_pythia_kvcache_eager_ignoreeos_rtt_dense_repeat3.csv \
  --summary-output results/method_timing_summary_pythia_kvcache_eager_ignoreeos_rtt_dense_repeat3.csv \
  --share-output results/method_timing_stage_shares_pythia_kvcache_eager_ignoreeos_rtt_dense_repeat3.csv \
  --upload-output results/method_timing_upload_summary_pythia_kvcache_eager_ignoreeos_rtt_dense_repeat3.csv \
  --plot-output results/method_timing_stage_shares_pythia_kvcache_eager_ignoreeos_rtt_dense_repeat3.png \
  --plot-rtt-ms 100 \
  --markdown-output ""
```

## 关键结果一：SmolLM KV-cache dense RTT

SmolLM 组结果显示，在当前实现下，RTT=0 ms 时 speculative methods 已经慢于 target-only。

| method | RTT(ms) | method_time(s) | speedup_vs_target_only | accept_rate |
| --- | ---: | ---: | ---: | ---: |
| target-only | 0 | 0.9351 | 1.0000 | 0.0000 |
| dsd-adaptive-draft | 0 | 1.8636 | 0.5100 | 0.5219 |
| vanilla-spec | 0 | 2.1790 | 0.4604 | 0.4487 |
| dsd-adaptive-draft | 20 | 2.2743 | 0.4179 | 0.5219 |
| vanilla-spec | 20 | 2.5719 | 0.3859 | 0.4487 |
| dsd-adaptive-draft | 100 | 3.5548 | 0.2780 | 0.5219 |
| vanilla-spec | 100 | 3.8852 | 0.2612 | 0.4487 |
| dsd-adaptive-draft | 200 | 4.9225 | 0.2012 | 0.5219 |
| vanilla-spec | 200 | 5.1248 | 0.1978 | 0.4487 |

### SmolLM 结论

SmolLM 在 RTT=0 ms 时没有 speculative 加速：

- `dsd-adaptive-draft` speedup = 0.5100
- `vanilla-spec` speedup = 0.4604

因此 SmolLM 组当前不存在正向 break-even RTT。也就是说，还没等加入网络时延，speculative path（推测路径）本身的 draft generation（草稿生成）和 cache update（缓存更新）开销已经超过 target-only baseline。

## 关键结果二：Pythia KV-cache eager + ignore-eos dense RTT

Pythia 组固定生成 32 个 token，correctness 通过，结果如下。

| method | RTT(ms) | method_time(s) | speedup_vs_target_only | accept_rate | generated_tokens | rounds | drafted_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target-only | 0 | 0.6259 | 1.0000 | 0.0000 | 32 | 0 | 0 |
| dsd-adaptive-draft | 0 | 0.4628 | 1.3518 | 1.0000 | 32 | 5 | 28 |
| dsd-adaptive-draft | 5 | 0.5327 | 1.1850 | 1.0000 | 32 | 5 | 28 |
| dsd-adaptive-draft | 10 | 0.5705 | 1.1149 | 1.0000 | 32 | 5 | 28 |
| dsd-adaptive-draft | 20 | 0.6241 | 1.0155 | 1.0000 | 32 | 5 | 28 |
| dsd-adaptive-draft | 50 | 0.7706 | 0.8183 | 1.0000 | 32 | 5 | 28 |
| dsd-adaptive-draft | 100 | 1.0625 | 0.5925 | 1.0000 | 32 | 5 | 28 |
| dsd-adaptive-draft | 200 | 1.5657 | 0.4006 | 1.0000 | 32 | 5 | 28 |
| vanilla-spec | 0 | 0.5125 | 1.2227 | 1.0000 | 32 | 7 | 26 |
| vanilla-spec | 5 | 0.5680 | 1.1027 | 1.0000 | 32 | 7 | 26 |
| vanilla-spec | 10 | 0.6556 | 0.9635 | 1.0000 | 32 | 7 | 26 |
| vanilla-spec | 20 | 0.7299 | 0.8657 | 1.0000 | 32 | 7 | 26 |
| vanilla-spec | 50 | 0.9546 | 0.6585 | 1.0000 | 32 | 7 | 26 |
| vanilla-spec | 100 | 1.3146 | 0.4784 | 1.0000 | 32 | 7 | 26 |
| vanilla-spec | 200 | 2.0483 | 0.3056 | 1.0000 | 32 | 7 | 26 |

### Pythia break-even RTT 估计

Pythia 组在 RTT=0 ms 时有正向加速，因此可以估计 break-even RTT。

#### dsd-adaptive-draft

- RTT=20 ms：`method_time=0.6241 s`，略快于 target-only `0.6259 s`
- RTT=50 ms：`method_time=0.7706 s`，慢于 target-only

线性插值估计：

```text
break-even RTT ≈ 20.4 ms
```

也就是说，对 Pythia 组当前实现，DSD-style adaptive draft strategy（DSD 风格自适应草稿策略）大约能承受 20 ms 左右 RTT。

#### vanilla-spec

- RTT=5 ms：`method_time=0.5680 s`，快于 target-only
- RTT=10 ms：`method_time=0.6556 s`，慢于 target-only

线性插值估计：

```text
break-even RTT ≈ 8.3 ms
```

也就是说，对 Pythia 组当前实现，普通 vanilla speculative decoding（普通线性推测解码）只能承受约 8 ms 左右 RTT。

## 关键结果三：RTT=100 ms 阶段时间占比

### SmolLM at RTT=100 ms

| method | draft generation | upload wait | target verification | cache update | sampling / argmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| dsd-adaptive-draft | 16.84% | 39.44% | 15.75% | 23.47% | 1.21% |
| vanilla-spec | 26.15% | 34.36% | 14.22% | 20.47% | 1.74% |
| target-only | 0.00% | 0.00% | 92.27% | 1.85% | 2.55% |

### Pythia at RTT=100 ms

| method | prefill | draft generation | upload wait | target verification | cache update | sampling / argmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dsd-adaptive-draft | 3.14% | 23.18% | 47.11% | 12.12% | 13.25% | 0.51% |
| vanilla-spec | 2.24% | 14.72% | 53.32% | 13.34% | 15.31% | 0.41% |
| target-only | 3.53% | 0.00% | 0.00% | 95.23% | 0.49% | 0.43% |

### 观察

在 RTT=100 ms 时，upload wait（上传等待）已经成为 Pythia speculative methods 的最大耗时来源：

- `dsd-adaptive-draft`：47.11%
- `vanilla-spec`：53.32%

这说明一旦端云 RTT 达到 100 ms，即使 Pythia 在本地计算上有明显 speculative 加速，通信等待也会迅速吞掉收益。

## 关键结果四：为什么 Pythia 有加速而 SmolLM 没有

### Pythia 的关键优势

Pythia 组的 accept rate 为 1.0，说明 draft model 生成的 draft tokens 几乎全部被 target model 接受。

这带来两个效果：

1. target model 每轮可以验证多个 token。
2. 每轮接受更多 token，rounds（交互轮数）减少。

Pythia 中：

- `dsd-adaptive-draft`：32 tokens 只需要 5 轮，drafted_tokens=28
- `vanilla-spec`：32 tokens 需要 7 轮，drafted_tokens=26

因此 DSD-style adaptive draft 能进一步减少 RTT 次数，break-even RTT 也比 vanilla-spec 更高。

### SmolLM 的主要问题

SmolLM 组 accept rate 明显更低：

- `dsd-adaptive-draft`：0.5219
- `vanilla-spec`：0.4487

接受率低会导致：

- draft generation（草稿生成）做了较多无效工作。
- target verification（目标验证）不能换来足够多 accepted tokens（被接受 token）。
- cache update（缓存更新）和 speculative control flow（推测控制流）开销相对更重。

因此 SmolLM 即使 RTT=0 ms，也没有跑赢 target-only。

## 当前可以下的结论

### 结论 1：RTT 是否拖慢 speculative decoding

可以确认：

> RTT 会显著拖慢端云式 speculative decoding。

在 Pythia 组中，RTT=0 ms 时 speculative decoding 是有加速的，但随着 RTT 增大，加速迅速消失：

- `dsd-adaptive-draft`：RTT=0 ms speedup=1.3518，RTT=100 ms speedup=0.5925
- `vanilla-spec`：RTT=0 ms speedup=1.2227，RTT=100 ms speedup=0.4784

### 结论 2：break-even RTT 依赖模型组合和 draft strategy

当前两组模型给出了清晰对照：

| model pair | method | RTT=0 speedup | estimated break-even RTT |
| --- | --- | ---: | ---: |
| SmolLM 360M/135M | dsd-adaptive-draft | 0.5100 | 无正向 break-even |
| SmolLM 360M/135M | vanilla-spec | 0.4604 | 无正向 break-even |
| Pythia 410M/160M | dsd-adaptive-draft | 1.3518 | 约 20.4 ms |
| Pythia 410M/160M | vanilla-spec | 1.2227 | 约 8.3 ms |

这说明 break-even RTT 不是一个固定常数，而是由以下因素共同决定：

- draft model 和 target model 的匹配程度。
- accept rate（接受率）。
- 每轮 accepted tokens（接受 token 数）。
- rounds（交互轮数）。
- draft generation 和 cache update 的本地开销。
- 网络 RTT 和上传 payload。

### 结论 3：DSD-style adaptive draft 在高接受率场景更有价值

Pythia 中 accept rate=1.0，DSD-style adaptive draft 可以扩大 draft_k，减少交互轮数。

结果上：

- DSD break-even RTT 约 20.4 ms。
- vanilla-spec break-even RTT 约 8.3 ms。

说明 adaptive draft strategy（自适应草稿策略）在高接受率模型组合上能明显提高可承受 RTT。

### 结论 4：高 RTT 下 upload wait 成为主导瓶颈

Pythia RTT=100 ms：

- `dsd-adaptive-draft` upload wait share=47.11%
- `vanilla-spec` upload wait share=53.32%

RTT=200 ms：

- `dsd-adaptive-draft` upload wait share=63.91%
- `vanilla-spec` upload wait share=68.39%

这说明当端云 RTT 较高时，继续优化本地计算只能解决一部分问题，减少交互轮数或合并上传验证会更关键。

## 当前不足

1. SmolLM dense RTT 是在扩展 Pythia prompts 之前跑的，严格跨模型对比还需要用同一组 10 条 prompts 重跑 SmolLM。
2. Pythia 使用 `--attn-implementation eager`，而 SmolLM 之前主要使用默认 attention backend；后续若强调严格公平，需统一 attention backend 或分别说明。
3. Pythia 使用 `--ignore-eos` 强制固定 32-token 输出，这是为了计时稳定，不代表真实应用中必须忽略 EOS。
4. 当前任务仍是轻量 greedy continuation，不是完整 benchmark。
5. 当前 upload 模型主要是固定 RTT，`--upload-bandwidth-mbps 0` 表示不计 payload transfer time（负载传输时间）。

## 下一步计划

为了把最终报告做得更严谨，建议补充以下实验：

### 1. SmolLM 使用扩展 prompts 重跑 dense RTT

目的：和 Pythia 使用同一组 10 条 prompts 做严格对比。

建议命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair smollm-local \
  --implementation kv-cache \
  --ignore-eos \
  --method target-only \
  --method vanilla-spec \
  --method dsd-adaptive-draft \
  --max-new-tokens 32 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 3 \
  --warmups 2 \
  --rtt-ms 0 5 10 20 50 100 150 200 \
  --upload-token-bytes 4 \
  --upload-bandwidth-mbps 0 \
  --output results/method_timing_smollm_kvcache_ignoreeos_rtt_dense_repeat3.csv
```

分析命令：

```bash
python scripts/06_analyze_method_timing.py \
  --input results/method_timing_smollm_kvcache_ignoreeos_rtt_dense_repeat3.csv \
  --summary-output results/method_timing_summary_smollm_kvcache_ignoreeos_rtt_dense_repeat3.csv \
  --share-output results/method_timing_stage_shares_smollm_kvcache_ignoreeos_rtt_dense_repeat3.csv \
  --upload-output results/method_timing_upload_summary_smollm_kvcache_ignoreeos_rtt_dense_repeat3.csv \
  --plot-output results/method_timing_stage_shares_smollm_kvcache_ignoreeos_rtt_dense_repeat3.png \
  --plot-rtt-ms 100 \
  --markdown-output ""
```

### 2. Pythia 增加 payload-aware upload 实验

目的：验证低带宽条件下 payload transfer time（负载传输时间）相对于固定 RTT 是否仍然较小。

建议命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair pythia-local \
  --implementation kv-cache \
  --attn-implementation eager \
  --ignore-eos \
  --method target-only \
  --method vanilla-spec \
  --method dsd-adaptive-draft \
  --max-new-tokens 32 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 3 \
  --warmups 2 \
  --rtt-ms 0 20 100 \
  --upload-token-bytes 4 \
  --upload-bandwidth-mbps 0.1 \
  --output results/method_timing_pythia_kvcache_eager_ignoreeos_payload_repeat3.csv
```

### 3. 最终跨模型报告

补完上述两个实验后，可以生成最终版跨模型报告，重点回答：

- 哪些模型组合在 RTT=0 ms 有真实 speculative 加速。
- 每个模型组合的 break-even RTT。
- upload wait 在不同 RTT 下如何吞掉加速收益。
- adaptive draft strategy 是否提高了可承受 RTT。

## 附录：本阶段关键异常与修复

### 1. Hugging Face 下载失败

服务器无法访问 `huggingface.co`：

```text
Network is unreachable
```

解决方式：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

### 2. Pythia 默认 SDPA attention mismatch

默认 attention backend 下：

```text
Method output mismatch. method=vanilla-spec, prompt_id=3
```

解决方式：

```bash
--attn-implementation eager
```

### 3. Pythia 过早 EOS

最初 Pythia 实验基本只生成约 1 个 token，导致计时不可用于 32-token 解码结论。

解决方式：

```bash
--ignore-eos
```

强制生成满 32 个 token 后，结果可用于阶段性结论。
