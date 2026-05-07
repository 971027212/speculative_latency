# 阶段报告 03：更精细的方法阶段时间与上传等待分析

## 本阶段目标

本阶段在 quick method timing experiment（方法阶段时间快速实验）的基础上，进一步拆解 upload wait（上传等待）对端云式 speculative decoding（推测解码）的影响。

重点不只是看总耗时，而是把上传等待拆成三个更细指标：

- `upload_wait_time`：累计上传等待时间。
- `upload_latency_time`：固定 RTT 或固定网络时延部分。
- `upload_transfer_time`：由 payload bytes（上传负载字节数）和 bandwidth（带宽）决定的传输时间。
- `upload_payload_bytes`：本轮实验累计上传的 draft token payload（草稿 token 负载）字节数。
- `upload_wait_share_percent`：上传等待占整个方法耗时的比例。
- `upload_wait_per_round_ms`：平均每轮 draft-to-target verification（草稿到目标验证）等待多少毫秒。
- `upload_wait_per_generated_token_ms`：平均每生成一个 token 承担多少上传等待。

## 输入与输出

- Raw input CSV（原始输入表）：`results/method_timing_network_quick.csv`
- Plot RTT（画图使用的 RTT）：`100.0` ms
- Plot output（图像输出）：`results/method_timing_stage_shares_network_quick.png`
- Report generated at（报告生成时间）：`2026-05-05 17:16:56`

## 方法汇总

| method | implementation | RTT(ms) | time(s) | speedup | accept rate | rounds | generated | drafted | wasted branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dsd-adaptive-draft | full-prefix | 0.0000 | 1.5179 | 0.6336 | 0.5219 | 14.0000 | 32.0000 | 36.3333 | 0.0000 |
| dsd-adaptive-draft | full-prefix | 20.0000 | 1.8698 | 0.5066 | 0.5219 | 14.0000 | 32.0000 | 36.3333 | 0.0000 |
| dsd-adaptive-draft | full-prefix | 100.0000 | 3.1227 | 0.3120 | 0.5219 | 14.0000 | 32.0000 | 36.3333 | 0.0000 |
| specinfer-simplified | full-prefix | 0.0000 | 2.8167 | 0.3544 | 0.2243 | 13.3333 | 32.0000 | 99.3333 | 79.6667 |
| specinfer-simplified | full-prefix | 20.0000 | 3.2343 | 0.3129 | 0.2243 | 13.3333 | 32.0000 | 99.3333 | 79.6667 |
| specinfer-simplified | full-prefix | 100.0000 | 4.3883 | 0.2309 | 0.2243 | 13.3333 | 32.0000 | 99.3333 | 79.6667 |
| target-only | full-prefix | 0.0000 | 0.9425 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 | 0.0000 |
| target-only | full-prefix | 20.0000 | 0.9425 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 | 0.0000 |
| target-only | full-prefix | 100.0000 | 0.9425 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 | 0.0000 |
| vanilla-spec | full-prefix | 0.0000 | 1.8288 | 0.5626 | 0.4487 | 13.3333 | 32.0000 | 49.6667 | 0.0000 |
| vanilla-spec | full-prefix | 20.0000 | 2.4504 | 0.4444 | 0.4487 | 13.3333 | 32.0000 | 49.6667 | 0.0000 |
| vanilla-spec | full-prefix | 100.0000 | 3.5013 | 0.2868 | 0.4487 | 13.3333 | 32.0000 | 49.6667 | 0.0000 |

## 上传等待精细分析

| method | RTT(ms) | time(s) | upload wait(s) | latency(s) | transfer(s) | payload(bytes) | upload share(%) | upload/round(ms) | upload/generated token(ms) | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dsd-adaptive-draft / full-prefix | 0.0000 | 1.5179 | 0.0126 | 0.0000 | 0.0116 | 145.3333 | 0.8306 | 0.9006 | 0.3940 | 0.6336 |
| dsd-adaptive-draft / full-prefix | 20.0000 | 1.8698 | 0.2933 | 0.2800 | 0.0116 | 145.3333 | 15.6881 | 20.9522 | 9.1666 | 0.5066 |
| dsd-adaptive-draft / full-prefix | 100.0000 | 3.1227 | 1.4133 | 1.4000 | 0.0116 | 145.3333 | 45.2594 | 100.9512 | 44.1661 | 0.3120 |
| specinfer-simplified / full-prefix | 0.0000 | 2.8167 | 0.0330 | 0.0000 | 0.0318 | 397.3333 | 1.1725 | 2.4769 | 1.0320 | 0.3544 |
| specinfer-simplified / full-prefix | 20.0000 | 3.2343 | 0.3008 | 0.2667 | 0.0318 | 397.3333 | 9.3010 | 22.5615 | 9.4006 | 0.3129 |
| specinfer-simplified / full-prefix | 100.0000 | 4.3883 | 1.3668 | 1.3333 | 0.0318 | 397.3333 | 31.1461 | 102.5082 | 42.7117 | 0.2309 |
| target-only / full-prefix | 0.0000 | 0.9425 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| target-only / full-prefix | 20.0000 | 0.9425 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| target-only / full-prefix | 100.0000 | 0.9425 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| vanilla-spec / full-prefix | 0.0000 | 1.8288 | 0.0171 | 0.0000 | 0.0159 | 198.6667 | 0.9334 | 1.2802 | 0.5334 | 0.5626 |
| vanilla-spec / full-prefix | 20.0000 | 2.4504 | 0.2842 | 0.2667 | 0.0159 | 198.6667 | 11.5991 | 21.3170 | 8.8821 | 0.4444 |
| vanilla-spec / full-prefix | 100.0000 | 3.5013 | 1.3508 | 1.3333 | 0.0159 | 198.6667 | 38.5803 | 101.3117 | 42.2132 | 0.2868 |

## 阶段占比

下表展示 RTT=100.0 ms 时各阶段占比。

| method | prefill(%) | draft generation(%) | draft structure(%) | upload wait(%) | target verification(%) | posterior accept(%) | cache update(%) | sampling(%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dsd-adaptive-draft / full-prefix | 0.00 | 36.79 | 0.00 | 45.26 | 16.84 | 0.00 | 0.26 | 0.48 |
| specinfer-simplified / full-prefix | 0.00 | 56.85 | 0.03 | 31.15 | 10.50 | 0.00 | 0.05 | 0.72 |
| target-only / full-prefix | 3.43 | 0.00 | 0.00 | 0.00 | 95.49 | 0.00 | 0.31 | 0.55 |
| vanilla-spec / full-prefix | 0.00 | 45.53 | 0.00 | 38.58 | 14.39 | 0.00 | 0.33 | 0.75 |

## 关键观察

- 当前结果中上传等待占比最高的是 `dsd-adaptive-draft / full-prefix` 在 RTT=100.0 ms 时，upload wait share（上传等待占比）为 45.26%。
- 该点的总耗时为 3.1227 s，其中 upload wait（上传等待）为 1.4133 s，speedup_vs_target_only（相对 target-only 加速比）为 0.3120。
- 在当前 full-prefix（完整前缀）实现下，RTT=0 ms 时 speculative methods（推测方法）仍未超过 target-only，主要原因是 draft generation（草稿生成）占比过高。
- 当 RTT 增大时，upload wait（上传等待）会快速变成主要成本之一，尤其是轮数较多的方法。
- target-only（只用目标模型）没有 draft token 上传，因此 upload wait 始终为 0。

## 当前口径说明

当前 upload wait（上传等待）不是实际网络测速，而是每轮 draft tokens（草稿 token）生成完成后模拟一次网络等待：

```python
upload_time = rtt_ms / 1000 + payload_bytes * 8 / (bandwidth_mbps * 1_000_000)
time.sleep(upload_time)
```

当 `upload_bandwidth_mbps <= 0` 时，只模拟固定 RTT，保持旧版实验口径；当 bandwidth（带宽）为正时，会额外加入 payload transfer time（负载传输时间）。它仍然没有建模 serialization（序列化）、协议头、拥塞和 batching（批处理）等真实网络因素。

## 下一步计划

1. 用 `--repeats 3` 或更高重复次数重跑实验，降低 quick run（快速实验）的随机波动。
2. 验证 `--implementation kv-cache` 的 correctness（正确性），因为 break-even RTT（盈亏平衡 RTT）必须基于 KV cache 版本判断。
3. 使用不同 `--upload-bandwidth-mbps` 档位重跑实验，观察 bandwidth（带宽）是否会在低带宽下成为主要瓶颈。
4. 对 SpecInfer-style token tree（SpecInfer 风格 token 树）进一步拆分 wasted branch tokens（无效分支 token）和实际 compute waste（计算浪费）。

## 英文术语中文解释

| English | 中文解释 |
| --- | --- |
| upload wait | 上传等待，端侧草稿 token 到云端目标模型验证前的等待 |
| RTT | Round Trip Time，往返时延 |
| draft generation | 草稿生成，小模型生成候选 token 的过程 |
| target verification | 目标模型验证，大模型验证候选 token 的过程 |
| speedup | 加速比，相对 target-only 的速度比例 |
| full-prefix | 完整前缀，每次前向重新输入完整上下文 |
| KV cache | 键值缓存，用来复用历史 token 的 key/value |
| break-even RTT | 盈亏平衡 RTT，加速收益刚好被网络等待抵消的时延 |
| bandwidth | 带宽，单位时间可传输的数据量 |
| payload bytes | 上传负载字节数 |
