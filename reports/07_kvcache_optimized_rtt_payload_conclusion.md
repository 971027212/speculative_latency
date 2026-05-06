# 阶段报告 07：优化版 KV-cache 推测解码的 RTT 与上传负载结论

## 本阶段目标

本阶段目标是在 correctness（正确性）已经通过的前提下，对优化版 KV-cache speculative decoding（键值缓存推测解码）进行更稳定的 repeat3（重复 3 次）实验，并判断当前实现是否已经进入可以讨论 break-even RTT（盈亏平衡 RTT）的阶段。

重点问题是：

1. RTT=0 ms 时，speculative decoding（推测解码）是否已经快于 target-only（只用目标模型）。
2. RTT 增大时，upload wait（上传等待）占比如何变化。
3. 在 payload-aware upload model（考虑上传负载的网络模型）中，主要瓶颈来自 fixed RTT latency（固定往返时延）还是 payload transfer time（负载传输时间）。

## 当前代码状态

当前最新关键改动：

- KV-cache speculative path（键值缓存推测路径）已经通过 correctness check（正确性检查）。
- `vanilla-spec / kv-cache` 使用 incremental target verification（增量目标模型验证）。
- `dsd-adaptive-draft / kv-cache` 已接入 cached speculative path（缓存推测路径）。
- 为避免 mutable cache（可变缓存）污染，临时 draft / verify cache 使用 clone（拷贝）。
- 对 full accept round（整轮草稿全接受）保守复用 verified target cache（已验证目标缓存）。

当前需要注意：

- correctness 已稳定。
- 但当前实现仍未在 RTT=0 ms 超过 target-only。
- 因此当前还不存在正向 break-even RTT。

## 实验环境

| 项目 | 配置 |
| --- | --- |
| GPU | RTX 3090 |
| conda env | `edge-specdec` |
| target model | `/home/chajiahao/data/hf_models/SmolLM-360M` |
| draft model | `/home/chajiahao/data/hf_models/SmolLM-135M` |
| model pair | `smollm-local` |
| implementation | `kv-cache` |
| max new tokens | 32 |
| draft_k | 4 |
| tree_width | 2 |
| repeats | 3 |
| warmups | 2 |

## 实验 1：dense RTT sweep，无 payload transfer

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair smollm-local \
  --implementation kv-cache \
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
  --output results/method_timing_kvcache_optimized_rtt_dense_repeat3.csv
```

分析命令：

```bash
python scripts/06_analyze_method_timing.py \
  --input results/method_timing_kvcache_optimized_rtt_dense_repeat3.csv \
  --summary-output results/method_timing_summary_kvcache_optimized_rtt_dense_repeat3.csv \
  --share-output results/method_timing_stage_shares_kvcache_optimized_rtt_dense_repeat3.csv \
  --upload-output results/method_timing_upload_summary_kvcache_optimized_rtt_dense_repeat3.csv \
  --plot-output results/method_timing_stage_shares_kvcache_optimized_rtt_dense_repeat3.png \
  --plot-rtt-ms 100 \
  --markdown-output ""
```

### 方法耗时与加速比

| method | RTT(ms) | method_time(s) | speedup_vs_target_only | accept_rate |
| --- | ---: | ---: | ---: | ---: |
| target-only | 0 | 0.9351 | 1.0000 | 0.0000 |
| target-only | 5 | 0.9351 | 1.0000 | 0.0000 |
| target-only | 10 | 0.9351 | 1.0000 | 0.0000 |
| target-only | 20 | 0.9351 | 1.0000 | 0.0000 |
| target-only | 50 | 0.9351 | 1.0000 | 0.0000 |
| target-only | 100 | 0.9351 | 1.0000 | 0.0000 |
| target-only | 150 | 0.9351 | 1.0000 | 0.0000 |
| target-only | 200 | 0.9351 | 1.0000 | 0.0000 |
| dsd-adaptive-draft | 0 | 1.8636 | 0.5100 | 0.5219 |
| dsd-adaptive-draft | 5 | 2.0287 | 0.4719 | 0.5219 |
| dsd-adaptive-draft | 10 | 2.1618 | 0.4395 | 0.5219 |
| dsd-adaptive-draft | 20 | 2.2743 | 0.4179 | 0.5219 |
| dsd-adaptive-draft | 50 | 2.7377 | 0.3500 | 0.5219 |
| dsd-adaptive-draft | 100 | 3.5548 | 0.2780 | 0.5219 |
| dsd-adaptive-draft | 150 | 4.2539 | 0.2333 | 0.5219 |
| dsd-adaptive-draft | 200 | 4.9225 | 0.2012 | 0.5219 |
| vanilla-spec | 0 | 2.1790 | 0.4604 | 0.4487 |
| vanilla-spec | 5 | 2.3263 | 0.4333 | 0.4487 |
| vanilla-spec | 10 | 2.4161 | 0.4164 | 0.4487 |
| vanilla-spec | 20 | 2.5719 | 0.3859 | 0.4487 |
| vanilla-spec | 50 | 3.0799 | 0.3249 | 0.4487 |
| vanilla-spec | 100 | 3.8852 | 0.2612 | 0.4487 |
| vanilla-spec | 150 | 4.4653 | 0.2249 | 0.4487 |
| vanilla-spec | 200 | 5.1248 | 0.1978 | 0.4487 |

### 上传等待占比

| method | RTT(ms) | upload_wait_time(s) | upload_wait_share | upload_wait_per_generated_token |
| --- | ---: | ---: | ---: | ---: |
| dsd-adaptive-draft | 0 | 0.0000 | 0.00% | 0.00 ms |
| dsd-adaptive-draft | 20 | 0.2818 | 12.39% | 8.81 ms |
| dsd-adaptive-draft | 50 | 0.7028 | 25.67% | 21.96 ms |
| dsd-adaptive-draft | 100 | 1.4019 | 39.44% | 43.81 ms |
| dsd-adaptive-draft | 150 | 2.1019 | 49.41% | 65.68 ms |
| dsd-adaptive-draft | 200 | 2.8018 | 56.92% | 87.56 ms |
| vanilla-spec | 0 | 0.0000 | 0.00% | 0.00 ms |
| vanilla-spec | 20 | 0.2684 | 10.44% | 8.39 ms |
| vanilla-spec | 50 | 0.6685 | 21.70% | 20.89 ms |
| vanilla-spec | 100 | 1.3351 | 34.36% | 41.72 ms |
| vanilla-spec | 150 | 2.0023 | 44.84% | 62.57 ms |
| vanilla-spec | 200 | 2.6684 | 52.07% | 83.39 ms |

### RTT=100 ms 阶段占比

| method | draft generation | upload wait | target verification | cache update | sampling |
| --- | ---: | ---: | ---: | ---: | ---: |
| dsd-adaptive-draft | 16.84% | 39.44% | 15.75% | 23.47% | 1.21% |
| vanilla-spec | 26.15% | 34.36% | 14.22% | 20.47% | 1.74% |
| target-only | 0.00% | 0.00% | 92.27% | 1.85% | 2.55% |

### dense RTT 结论

在优化版 KV-cache 实现中：

- `dsd-adaptive-draft` 在 RTT=0 ms 时 speedup 为 0.5100。
- `vanilla-spec` 在 RTT=0 ms 时 speedup 为 0.4604。
- 二者都明显慢于 `target-only`。
- RTT 增大后，upload wait share（上传等待占比）持续上升，speedup 继续下降。
- RTT=200 ms 时，上传等待已经占 `dsd-adaptive-draft` 总时间的 56.92%，占 `vanilla-spec` 总时间的 52.07%。

因此，当前实现下没有正向 break-even RTT。

## 实验 2：payload-aware upload model

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair smollm-local \
  --implementation kv-cache \
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
  --output results/method_timing_kvcache_optimized_payload_repeat3.csv
```

分析命令：

```bash
python scripts/06_analyze_method_timing.py \
  --input results/method_timing_kvcache_optimized_payload_repeat3.csv \
  --summary-output results/method_timing_summary_kvcache_optimized_payload_repeat3.csv \
  --share-output results/method_timing_stage_shares_kvcache_optimized_payload_repeat3.csv \
  --upload-output results/method_timing_upload_summary_kvcache_optimized_payload_repeat3.csv \
  --plot-output results/method_timing_stage_shares_kvcache_optimized_payload_repeat3.png \
  --plot-rtt-ms 100 \
  --markdown-output ""
```

### payload-aware 方法耗时

| method | RTT(ms) | method_time(s) | speedup_vs_target_only | accept_rate |
| --- | ---: | ---: | ---: | ---: |
| target-only | 0 | 1.0760 | 1.0000 | 0.0000 |
| target-only | 20 | 1.0760 | 1.0000 | 0.0000 |
| target-only | 100 | 1.0760 | 1.0000 | 0.0000 |
| dsd-adaptive-draft | 0 | 2.2143 | 0.4982 | 0.5219 |
| dsd-adaptive-draft | 20 | 2.6545 | 0.4181 | 0.5219 |
| dsd-adaptive-draft | 100 | 3.8758 | 0.2877 | 0.5219 |
| vanilla-spec | 0 | 2.5460 | 0.4569 | 0.4487 |
| vanilla-spec | 20 | 2.9157 | 0.3984 | 0.4487 |
| vanilla-spec | 100 | 4.0990 | 0.2769 | 0.4487 |

### payload-aware 上传拆分

| method | RTT(ms) | payload bytes | latency time | transfer time | upload wait share |
| --- | ---: | ---: | ---: | ---: | ---: |
| dsd-adaptive-draft | 0 | 145 | 0.0000s | 0.0116s | 0.57% |
| dsd-adaptive-draft | 20 | 145 | 0.2800s | 0.0116s | 11.06% |
| dsd-adaptive-draft | 100 | 145 | 1.4000s | 0.0116s | 36.47% |
| vanilla-spec | 0 | 199 | 0.0000s | 0.0159s | 0.67% |
| vanilla-spec | 20 | 199 | 0.2667s | 0.0159s | 9.75% |
| vanilla-spec | 100 | 199 | 1.3333s | 0.0159s | 32.96% |

### payload-aware 结论

在 `upload_bandwidth_mbps=0.1` 的设置下：

- RTT=0 ms 时仍有少量 upload wait，因为 payload transfer time 不为 0。
- `dsd-adaptive-draft` 的 payload transfer time 为 0.0116s。
- `vanilla-spec` 的 payload transfer time 为 0.0159s。
- RTT=100 ms 时，fixed latency time（固定时延）分别为 1.4000s 和 1.3333s。

因此，当前 token payload 规模下，网络瓶颈主要来自 fixed RTT latency，而不是 payload bytes 传输。

## 总体结论

### 可以下的结论

1. 当前优化版 KV-cache speculative decoding 已经通过 correctness check。
2. 当前实现中，`dsd-adaptive-draft` 和 `vanilla-spec` 在 RTT=0 ms 时都慢于 `target-only`。
3. 因为 RTT=0 ms 时没有加速，所以当前不存在正向 break-even RTT。
4. RTT 增大时，upload wait share 显著上升，并且会成为主要时间组成。
5. 在 RTT=100 ms 时，upload wait share 达到约 34% 到 39%。
6. 在 RTT=200 ms 时，upload wait share 超过 52%。
7. payload-aware 实验显示，当前主要网络瓶颈是 fixed RTT latency，不是 token payload transfer。

### 不能下的结论

当前不能声称：

- speculative decoding 已经在本模型对上获得端到端加速。
- 已经找到 break-even RTT。
- 当前 simplified SpecInfer 或 DSD-style strategy 等价于完整论文复现。

原因是当前 speculative methods 在 RTT=0 ms 下仍慢于 target-only。

## 为什么 RTT=0 仍然慢

从 RTT=100 ms 阶段占比看，即使不考虑 upload wait，当前 speculative methods 仍有明显额外成本：

- `draft generation` 仍然较高。
- `cache update` 仍然较高。
- `target verification` 虽然低于 target-only，但节省不足以抵消 draft 和 cache update 成本。

以 dense RTT 实验的 RTT=100 ms 为例：

| method | draft generation | target verification | cache update |
| --- | ---: | ---: | ---: |
| dsd-adaptive-draft | 16.84% | 15.75% | 23.47% |
| vanilla-spec | 26.15% | 14.22% | 20.47% |

这说明下一阶段优化重点不是继续扫 RTT，而是降低 draft generation 和 cache update。

## 下一步技术路线

### 1. 优化 cache update

当前 cache update 仍然占 20% 到 23%。下一步需要更精细地复用 verified cache 和 draft cache，但必须保证 correctness。

建议方向：

- 对 full accept round（全接受轮次）复用 target verify cache。
- 对 rejected round（拒绝轮次）避免不安全 cache crop。
- 为 cache reuse 增加单元测试，逐 token 比较 logits 和 output。

### 2. 优化 draft generation

当前 draft generation 占比仍高。建议：

- 检查 draft-side KV cache 是否真正减少了重复前向。
- 对比 full-prefix draft generation 和 cached draft generation 的每 token 时间。
- 尝试更小 draft model 或更高接受率模型对。

### 3. 更长 generation 长度

当前 `max_new_tokens=32` 偏短。可以补：

```text
max_new_tokens = 64 / 128
```

长生成可能更有利于 speculative decoding 摊薄 prefill 和固定开销。

### 4. 第二组模型对

当前只用了 SmolLM-360M / SmolLM-135M。后续需要第二组模型对验证结论是否普遍。

## 推荐下一组命令

当前不建议继续做更多 RTT sweep。建议先做定位实验：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair smollm-local \
  --implementation kv-cache \
  --method target-only \
  --method vanilla-spec \
  --method dsd-adaptive-draft \
  --max-new-tokens 64 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 3 \
  --warmups 2 \
  --rtt-ms 0 \
  --upload-token-bytes 4 \
  --upload-bandwidth-mbps 0 \
  --output results/method_timing_kvcache_optimized_len64_repeat3.csv
```

如果 len64 仍然没有 RTT=0 加速，则说明当前模型对和实现组合下，主要瓶颈不是网络，而是 speculative path 自身开销。

## 英文术语中文解释

| English | 中文解释 |
| --- | --- |
| speculative decoding | 推测解码 |
| target-only | 只用目标模型解码 |
| draft model | 草稿模型，小模型，负责生成候选 token |
| target model | 目标模型，大模型，负责验证候选 token |
| KV cache | 键值缓存，缓存 Transformer 历史 key/value |
| RTT | Round Trip Time，往返时延 |
| break-even RTT | 盈亏平衡 RTT，加速收益刚好被网络等待抵消的时延 |
| upload wait | 上传等待，端侧 draft tokens 到云端验证前的等待 |
| payload bytes | 上传负载字节数 |
| payload transfer time | 负载传输时间 |
| fixed latency | 固定时延 |
| cache update | 缓存更新 |
| draft generation | 草稿生成 |
| target verification | 目标模型验证 |
| correctness check | 正确性检查，验证输出是否和 target-only 一致 |

