# 阶段总报告 06：speculative decoding 端云时延实验汇总与下一步数据需求

## 本报告目标

本报告汇总目前已经完成的 speculative decoding（推测解码）实验，包括：

- full-prefix（完整前缀）版本的方法阶段时间占比实验。
- upload wait（上传等待）占比实验。
- payload-aware upload model（考虑上传负载的网络模型）实验。
- kv-cache（键值缓存）版本 correctness（正确性）验证。
- kv-cache + DSD-style adaptive draft（DSD 风格自适应草稿策略）的 RTT sweep（往返时延扫描）。

同时，本报告明确下一阶段还需要补充哪些数据，才能继续回答 break-even RTT（盈亏平衡 RTT）和端云式 speculative decoding 是否真正加速的问题。

## 当前研究问题

最初问题是：

> 在端云式 speculative decoding 中，端侧 draft tokens（草稿 token）上传到云端 target model（目标模型）验证的 RTT（Round Trip Time，往返时延），是否会拖慢整体推理，并在什么 RTT 条件下抵消 speculative decoding 的加速收益。

现在问题已经细化为：

> 不同解码方法在完整 decode（解码）过程中，时间分别花在 draft generation（草稿生成）、target verification（目标模型验证）、cache update（缓存更新）、upload wait（上传等待）、payload transfer（负载传输）等阶段的比例是多少。

## 实验环境

- GPU：RTX 3090
- conda 环境：`edge-specdec`
- torch：`2.6.0+cu124`
- transformers：`4.51.3`
- target model（目标模型）：`/home/chajiahao/data/hf_models/SmolLM-360M`
- draft model（草稿模型）：`/home/chajiahao/data/hf_models/SmolLM-135M`
- model pair name（模型对名称）：`smollm-local`

## 已完成的基础验证

### 环境和官方路径

已经完成：

- `pip install -r requirements.txt`
- CUDA 可用性检查。
- Hugging Face official assisted generation（官方辅助生成）路径验证：

```python
target_model.generate(..., assistant_model=draft_model)
```

说明 target model 和 draft model 都能正常加载和推理。

### 自写 full-prefix speculative decoding correctness

自写 vanilla speculative decoding（普通线性推测解码）的 full-prefix 版本已经通过 correctness check（正确性检查）：

- speculative output（推测解码输出）
- target-only greedy output（只用目标模型的贪心解码输出）

二者一致。

## 实验 1：full-prefix quick method timing

### 实验命令

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

分析命令：

```bash
python scripts/06_analyze_method_timing.py \
  --input results/method_timing_quick.csv \
  --summary-output results/method_timing_summary_quick.csv \
  --share-output results/method_timing_stage_shares_quick.csv \
  --plot-output results/method_timing_stage_shares_quick.png \
  --plot-rtt-ms 0
```

### 方法汇总

| method | RTT(ms) | method_time(s) | speedup_vs_target_only | accept_rate | wasted_branch_time_or_tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| target-only | 0 | 0.9258 | 1.0000 | 0.0000 | 0.0000 |
| target-only | 20 | 0.9258 | 1.0000 | 0.0000 | 0.0000 |
| target-only | 100 | 0.9258 | 1.0000 | 0.0000 | 0.0000 |
| vanilla-spec | 0 | 1.7095 | 0.5905 | 0.4487 | 0.0000 |
| vanilla-spec | 20 | 2.1274 | 0.4823 | 0.4487 | 0.0000 |
| vanilla-spec | 100 | 3.4628 | 0.2827 | 0.4487 | 0.0000 |
| specinfer-simplified | 0 | 2.6773 | 0.3735 | 0.2243 | 79.6667 |
| specinfer-simplified | 20 | 3.0681 | 0.3285 | 0.2243 | 79.6667 |
| specinfer-simplified | 100 | 4.3889 | 0.2298 | 0.2243 | 79.6667 |
| dsd-adaptive-draft | 0 | 1.3614 | 0.6844 | 0.5219 | 0.0000 |
| dsd-adaptive-draft | 20 | 1.8896 | 0.4899 | 0.5219 | 0.0000 |
| dsd-adaptive-draft | 100 | 3.1036 | 0.3098 | 0.5219 | 0.0000 |

### RTT=0 ms 阶段占比

| method | draft generation | target verification | cache update | sampling | prefill | structure | upload | accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dsd-adaptive-draft | 69.41% | 28.82% | 0.41% | 0.88% | 0.00% | 0.00% | 0.00% | 0.01% |
| specinfer-simplified | 84.27% | 13.99% | 0.08% | 0.87% | 0.00% | 0.05% | 0.00% | 0.00% |
| target-only | 0.00% | 95.06% | 0.43% | 0.75% | 3.51% | 0.00% | 0.00% | 0.00% |
| vanilla-spec | 75.80% | 22.33% | 0.42% | 0.96% | 0.00% | 0.00% | 0.00% | 0.01% |

### full-prefix 实验结论

在 full-prefix 实现下，三个 speculative methods（推测方法）在 RTT=0 ms 时都没有超过 target-only。主要原因是 draft generation（草稿生成）占比过高。

因此，full-prefix 结果可以用于验证 timing framework（计时框架）和上传等待趋势，但不能用于给出真正的 break-even RTT。

## 实验 2：full-prefix 上传等待占比

### 上传等待占比结果

| method | RTT=20ms upload share | RTT=100ms upload share |
| --- | ---: | ---: |
| dsd-adaptive-draft | 14.92% | 45.16% |
| vanilla-spec | 12.62% | 38.55% |
| specinfer-simplified | 8.75% | 30.42% |
| target-only | 0.00% | 0.00% |

### 结论

RTT 增大会显著增加 upload wait（上传等待）占比。RTT=100 ms 时，上传等待已经占 speculative methods 总耗时的 30% 到 45%。

这说明：

> 端侧 draft tokens 上传到云端 target model 验证的交互时延，会明显拖慢 speculative decoding。

## 实验 3：payload-aware upload model

### 模型定义

后续将上传等待从固定 RTT 扩展为：

```text
upload_time = fixed_latency + payload_bytes / bandwidth
```

代码口径：

```python
upload_time = rtt_ms / 1000 + payload_bytes * 8 / (bandwidth_mbps * 1_000_000)
```

其中：

- `upload_latency_time`：固定 RTT 或固定网络时延。
- `upload_transfer_time`：payload bytes（上传负载字节数）除以 bandwidth（带宽）得到的传输时间。
- `upload_payload_bytes`：draft token payload（草稿 token 上传负载）。
- `upload_token_bytes`：每个 draft token 估算占用字节数，本轮为 4 bytes。
- `upload_bandwidth_mbps`：上传带宽，本轮为 0.1 Mbps。

### full-prefix payload-aware quick 结果

| method | RTT | upload share | latency | transfer | payload |
| --- | ---: | ---: | ---: | ---: | ---: |
| dsd-adaptive-draft | 100 ms | 45.26% | 1.4000s | 0.0116s | 145 bytes |
| vanilla-spec | 100 ms | 38.58% | 1.3333s | 0.0159s | 199 bytes |
| specinfer-simplified | 100 ms | 31.15% | 1.3333s | 0.0318s | 397 bytes |

### 结论

在当前 `upload_bandwidth_mbps=0.1` 的设置下，`upload_latency_time` 远大于 `upload_transfer_time`。

也就是说，当前实验里的主要网络瓶颈仍然是固定 RTT，而不是 token payload 的字节数传输。

SpecInfer-style token tree（SpecInfer 风格 token 树）上传 payload 最大，因为它包含更多候选分支。

## 实验 4：kv-cache correctness quick

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair smollm-local \
  --implementation kv-cache \
  --max-new-tokens 32 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 1 \
  --warmups 1 \
  --rtt-ms 0 \
  --upload-token-bytes 4 \
  --upload-bandwidth-mbps 0.1 \
  --output results/method_timing_kvcache_correctness_quick.csv
```

分析报告：

```text
reports/04_kvcache_correctness_timing_analysis.md
```

### 结果

| method | method_time(s) | speedup | accept_rate | 主要观察 |
| --- | ---: | ---: | ---: | --- |
| target-only kv-cache | 0.8936 | 1.0000 | 0.0000 | 正常 baseline |
| dsd-adaptive-draft kv-cache | 1.4173 | 0.6386 | 0.5219 | correctness 通过，但仍慢 |
| specinfer-simplified kv-cache | 2.7222 | 0.3548 | 0.2243 | correctness 通过，但仍慢 |
| vanilla-spec kv-cache | 3.4999 | 0.2581 | 0.1183 | accept_rate 偏低，cache update 占比异常高 |

### 阶段占比

| method | draft generation | target verification | cache update | upload wait | sampling |
| --- | ---: | ---: | ---: | ---: | ---: |
| dsd-adaptive-draft kv-cache | 67.73% | 28.56% | 0.65% | 0.91% | 1.59% |
| specinfer-simplified kv-cache | 81.73% | 14.61% | 0.11% | 1.22% | 1.25% |
| target-only kv-cache | 0.00% | 94.25% | 0.81% | 0.00% | 1.44% |
| vanilla-spec kv-cache | 44.11% | 18.25% | 33.76% | 0.83% | 1.03% |

### 修复过程

KV-cache 版本经历了两个关键修复：

1. 修复 attention mask（注意力掩码）和 past_key_values（历史键值缓存）长度不匹配问题。
2. 将 `speculative_greedy_cached` 的 target verification 暂时改为 full-prefix verification（完整前缀验证），作为 conservative correctness path（保守正确性路径）。

当前状态：

- correctness 已经通过。
- 但性能不是最终版本。
- 尤其 `vanilla-spec kv-cache` 仍有明显异常，不适合作为最终性能结论。

## 实验 5：kv-cache + DSD RTT quick

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair smollm-local \
  --implementation kv-cache \
  --method target-only \
  --method dsd-adaptive-draft \
  --max-new-tokens 32 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 1 \
  --warmups 1 \
  --rtt-ms 0 20 100 \
  --upload-token-bytes 4 \
  --upload-bandwidth-mbps 0.1 \
  --output results/method_timing_kvcache_dsd_rtt_quick.csv
```

分析报告：

```text
reports/05_kvcache_dsd_rtt_analysis.md
```

### 结果

| RTT | method_time | speedup | upload_wait | upload_share |
| ---: | ---: | ---: | ---: | ---: |
| 0 ms | 1.4397s | 0.6764x | 0.0125s | 0.87% |
| 20 ms | 1.7614s | 0.5641x | 0.2935s | 16.66% |
| 100 ms | 3.0251s | 0.3427x | 1.4133s | 46.72% |

target-only kv-cache baseline：

| method | method_time | speedup |
| --- | ---: | ---: |
| target-only kv-cache | 0.9689s | 1.0000x |

### RTT=100 ms 阶段占比

| method | draft generation | upload wait | target verification | cache update | sampling |
| --- | ---: | ---: | ---: | ---: | ---: |
| dsd-adaptive-draft kv-cache | 35.12% | 46.72% | 17.01% | 0.28% | 0.45% |
| target-only kv-cache | 0.00% | 0.00% | 94.04% | 0.54% | 1.08% |

### 结论

在当前 conservative KV-cache 实现中，DSD-style adaptive draft 在 RTT=100 ms 时：

- upload wait share（上传等待占比）：46.72%
- speedup_vs_target_only（相对 target-only 加速比）：0.3427x
- upload latency time（固定时延）：1.4000s
- upload transfer time（负载传输时间）：0.0116s

这说明 RTT 固定等待是主要网络瓶颈。

但由于 RTT=0 ms 时 speculative method 仍然慢于 target-only，因此当前还不能计算真正的正向 break-even RTT。

## 当前总判断

### 已经可以确定的结论

1. 端侧 draft tokens 上传到云端 target verification 的等待会显著拖慢 speculative decoding。
2. RTT 从 0 ms 增加到 100 ms 时，upload wait share 可以从约 1% 上升到约 47%。
3. 当前 payload-aware model 下，固定 RTT 是主要网络瓶颈，payload transfer time 远小于 latency time。
4. SpecInfer-style token tree 上传 payload 更多，但当前 simplified implementation（简化实现）不能代表完整 SpecInfer 论文性能。
5. DSD-style adaptive draft 是当前 speculative methods 中较稳定的一个，但仍没有超过 target-only。

### 还不能下的结论

目前不能声称已经找到了真正的 break-even RTT。

原因是：

- full-prefix 版本没有 KV-cache 优化。
- 当前 conservative KV-cache 版本为了 correctness，target verification 仍有 full-prefix 成分。
- speculative methods 在 RTT=0 ms 时仍慢于 target-only。

只有当某个 optimized speculative method（优化后的推测方法）在 RTT=0 ms 时超过 target-only，才有意义计算：

```text
break-even RTT = speedup 被上传等待抵消时的 RTT
```

## 当前需要的数据

下一步我最需要以下数据。

### 1. 稳定性数据：repeats 提高到 3 或 5

当前很多结果是 `repeats=1` quick run（快速单次实验），适合看趋势，但不适合作最终表格。

建议先跑：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair smollm-local \
  --implementation kv-cache \
  --method target-only \
  --method dsd-adaptive-draft \
  --max-new-tokens 32 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 3 \
  --warmups 2 \
  --rtt-ms 0 20 100 \
  --upload-token-bytes 4 \
  --upload-bandwidth-mbps 0.1 \
  --output results/method_timing_kvcache_dsd_rtt_repeat3.csv
```

### 2. RTT 更密集扫描

为了观察 upload wait 对 speedup 的曲线影响，需要更多 RTT 档位：

```text
0 / 5 / 10 / 20 / 50 / 100 / 150 / 200 ms
```

建议命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/05_method_timing_experiment.py \
  --pair smollm-local \
  --implementation kv-cache \
  --method target-only \
  --method dsd-adaptive-draft \
  --max-new-tokens 32 \
  --draft-k 4 \
  --tree-width 2 \
  --repeats 3 \
  --warmups 2 \
  --rtt-ms 0 5 10 20 50 100 150 200 \
  --upload-token-bytes 4 \
  --upload-bandwidth-mbps 0.1 \
  --output results/method_timing_kvcache_dsd_rtt_dense.csv
```

### 3. 带宽扫描

当前使用 `0.1 Mbps` 是为了让 payload transfer time 能被观察到。后续需要区分：

- RTT dominated（固定时延主导）
- bandwidth dominated（带宽主导）

建议扫：

```text
upload_bandwidth_mbps = 0 / 0.1 / 1 / 10 / 100
```

其中 `0` 表示只模拟固定 RTT，不加 payload transfer。

### 4. 更长输出长度

`max_new_tokens=32` 偏短。端云交互成本在长生成中可能更明显。

建议补：

```text
max_new_tokens = 64 / 128
```

### 5. 优化版 KV-cache 数据

最关键的数据是 optimized KV-cache speculative decoding（优化版 KV-cache 推测解码）：

- target verification 不再 full-prefix。
- cache update 不再重复跑 target/draft。
- vanilla-spec accept_rate 恢复正常。
- RTT=0 ms 时至少有一个 speculative method 超过 target-only。

这一步完成后，才能真正计算 break-even RTT。

### 6. 每次实验需要保存的文件

每次实验请保存并发回：

```text
results/*raw*.csv
results/*summary*.csv
results/*stage_shares*.csv
results/*upload_summary*.csv
reports/*.md
```

同时建议记录：

```bash
git log --oneline -3
nvidia-smi
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
```

## 下一阶段建议

下一阶段优先级如下：

1. 先跑 `repeats=3` 的 kv-cache + dsd-adaptive-draft RTT 实验，确认当前结论稳定。
2. 再做 RTT dense sweep（密集 RTT 扫描）。
3. 然后优化 KV-cache speculative 内部实现，目标是让 RTT=0 ms 出现真实加速。
4. 最后再做 bandwidth sweep（带宽扫描）和 longer generation（长生成）实验。

## 英文术语中文解释

| English | 中文解释 |
| --- | --- |
| speculative decoding | 推测解码 |
| draft model | 草稿模型，小模型，负责先生成候选 token |
| target model | 目标模型，大模型，负责验证候选 token |
| draft generation | 草稿生成 |
| target verification | 目标模型验证 |
| posterior acceptance | 后验接受，决定接受多少草稿 token |
| upload wait | 上传等待，端侧草稿 token 到云端目标模型验证前的等待 |
| RTT | Round Trip Time，往返时延 |
| payload bytes | 上传负载字节数 |
| bandwidth | 带宽，单位时间内可传输的数据量 |
| KV cache | 键值缓存，用来复用历史 token 的 key/value |
| full-prefix | 完整前缀，每次前向重新输入完整上下文 |
| break-even RTT | 盈亏平衡 RTT，加速收益刚好被网络等待抵消的时延 |
| token tree | token 树，多分支候选 token 结构 |
| wasted branch tokens | 无效分支 token，没有进入最终输出的候选分支 |

