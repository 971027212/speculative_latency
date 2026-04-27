# Edge-Cloud Speculative Decoding RTT Experiment

这个项目用来验证一个具体问题：

> 传统端云式 speculative decoding 中，端侧 draft tokens 上传到云端 target verify 的交互时延，是否会拖慢整体推理，并在什么 RTT 条件下抵消 speculative decoding 的加速收益。

第一版代码故意写成“学习友好”的形式：先不用 KV cache，而是用完整 prefix forward 来实现 greedy speculative decoding。这样每一步 token 是怎么 draft、怎么 verify、怎么 accept/reject 都能直接读懂。

## 1. 环境准备

在远程 Linux GPU 服务器上新建独立 conda 环境：

```bash
conda create -n edge-specdec python=3.11 -y
conda activate edge-specdec
pip install -r requirements.txt
```

`requirements.txt` 固定了 PyTorch CUDA 12.4 wheel 源，避免误装到和老驱动不兼容的 CUDA 13 wheel。

## 2. 官方 API Sanity Check

先确认 Hugging Face 的 `assistant_model` assisted generation 可以正常跑：

```bash
python scripts/01_hf_assisted_sanity.py --pair smollm --max-new-tokens 32
python scripts/01_hf_assisted_sanity.py --pair pythia --max-new-tokens 32
```

这一步不做细粒度计时，只确认模型组合、tokenizer 和 generate 路径没问题。

## 3. 自写 Greedy Speculative Decoding

跑自写最小实现，并检查输出是否和 target-only greedy 完全一致：

```bash
python scripts/02_greedy_spec_decode.py --pair smollm --max-new-tokens 32 --draft-k 4
```

如果 speculative 输出和 target-only greedy 不一致，脚本会直接报错。

核心逻辑在 `edge_specdec/decoding.py`：

1. draft model 连续生成 `draft_k` 个候选 token。
2. 模拟上传等待，可设置 `--rtt-ms`。
3. target model 一次 forward verify 这些 draft token。
4. 从左到右比较 draft token 和 target greedy argmax。
5. 连续相等则 accept；第一个不相等处用 target token 替换。
6. 如果全部 accept，则额外追加 target 给出的 bonus token。

## 4. RTT Sweep

按计划注入 RTT 阶梯：

```bash
python scripts/03_rtt_sweep.py \
  --pair smollm \
  --max-new-tokens 32 \
  --draft-k 4 \
  --repeats 3 \
  --rtt-ms 0 5 10 20 50 100 \
  --output results/rtt_sweep_smollm.csv
```

两组模型都跑：

```bash
python scripts/03_rtt_sweep.py \
  --max-new-tokens 32 \
  --draft-k 4 \
  --repeats 3 \
  --rtt-ms 0 5 10 20 50 100 \
  --output results/rtt_sweep.csv
```

每条记录会输出这些计时桶：

- `prefill_time`
- `draft_generate_time`
- `upload_wait_time`
- `target_verify_time`
- `posterior_accept_time`
- `kv_or_input_update_time`
- `sampling_time`
- `total_decode_time`

注意：当前 full-prefix 学习版没有真正复用 KV cache，所以 speculative 路径里的 `prefill_time` 通常为 0；主要时间会落在 `draft_generate_time` 和 `target_verify_time`。等升级成 KV-cache 版本后，`prefill_time` 会变成实际 prompt prefill 成本。

## 5. 分析 Break-even RTT

```bash
python scripts/04_analyze_results.py \
  --input results/rtt_sweep.csv \
  --output results/rtt_summary.csv
```

重点看三件事：

1. `speedup` 是否随 RTT 增大而下降。
2. `upload_wait_share` 是否明显上升。
3. 哪个 RTT 首次让 `speedup <= 1.0`，这就是本实验里的 break-even RTT。

## 重要说明

这一版是 correctness-first 的学习实现，不是最快实现。因为没有复用 KV cache，绝对速度不会代表工业实现；但 RTT 注入对总时延和 speedup 的影响趋势仍然可以先被观察出来。下一步可以把 `edge_specdec/decoding.py` 升级成 cache-aware 版本，再比较 full-prefix 和 KV-cache 两种实现下的 break-even RTT。
