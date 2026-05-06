# Group Meeting Timing Data

This file contains data snippets only. Use it as input for the final group-meeting report.

## Break-Even Notes

- Qwen2.5-1.5B / dsd-adaptive-draft: no positive break-even; RTT=0 speedup=0.3498x.
- Qwen2.5-1.5B / vanilla-spec: no positive break-even; RTT=0 speedup=0.2725x.
- SmolLM / dsd-adaptive-draft: no positive break-even; RTT=0 speedup=0.3143x.
- SmolLM / vanilla-spec: no positive break-even; RTT=0 speedup=0.2807x.

## Stage Highlights at RTT=100 ms

- Qwen2.5-1.5B / dsd-adaptive-draft / kv-cache: top stages at RTT=100 ms are network_wait=35.00%, cache_update=29.82%, cloud_verification=22.73%.
- Qwen2.5-1.5B / target-only / kv-cache: top stages at RTT=100 ms are cloud_verification=95.37%, prefill=3.22%, sampling_argmax=0.66%.
- Qwen2.5-1.5B / vanilla-spec / kv-cache: top stages at RTT=100 ms are cloud_verification=29.02%, network_wait=27.79%, cache_update=24.25%.
- SmolLM / dsd-adaptive-draft / kv-cache: top stages at RTT=100 ms are network_wait=30.50%, cache_update=28.98%, cloud_verification=24.25%.
- SmolLM / target-only / kv-cache: top stages at RTT=100 ms are cloud_verification=95.38%, prefill=3.47%, sampling_argmax=0.46%.
- SmolLM / vanilla-spec / kv-cache: top stages at RTT=100 ms are cache_update=27.11%, cloud_verification=26.95%, network_wait=26.09%.

## Stage Share Table at RTT=100 ms

| group | method | rtt_ms | method_time | speedup_vs_target_only | draft_generation_percent | network_wait_percent | cloud_verification_percent | cache_update_percent | sampling_argmax_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | dsd-adaptive-draft / kv-cache | 100.0000 | 3.9148 | 0.2155 | 10.4986 | 34.9953 | 22.7345 | 29.8248 | 0.2707 |
| Qwen2.5-1.5B | target-only / kv-cache | 100.0000 | 0.8302 | 1.0000 | 0.0000 | 0.0000 | 95.3704 | 0.4804 | 0.6636 |
| Qwen2.5-1.5B | vanilla-spec / kv-cache | 100.0000 | 4.6067 | 0.1904 | 17.1965 | 27.7859 | 29.0199 | 24.2472 | 0.3314 |
| SmolLM | dsd-adaptive-draft / kv-cache | 100.0000 | 4.2616 | 0.2121 | 14.1680 | 30.5047 | 24.2522 | 28.9791 | 0.2753 |
| SmolLM | target-only / kv-cache | 100.0000 | 0.8976 | 1.0000 | 0.0000 | 0.0000 | 95.3832 | 0.4417 | 0.4567 |
| SmolLM | vanilla-spec / kv-cache | 100.0000 | 4.4848 | 0.2029 | 17.9605 | 26.0883 | 26.9465 | 27.1070 | 0.2928 |

## Method Summary

| group | method_name | target_verify_mode | rtt_ms | method_time | speedup_vs_target_only | accept_rate | rounds | generated_tokens | drafted_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | dsd-adaptive-draft | sequential | 0.0000 | 2.3743 | 0.3498 | 0.5907 | 13.7000 | 32.0000 | 32.9000 |
| Qwen2.5-1.5B | dsd-adaptive-draft | sequential | 5.0000 | 2.5072 | 0.3316 | 0.5907 | 13.7000 | 32.0000 | 32.9000 |
| Qwen2.5-1.5B | dsd-adaptive-draft | sequential | 10.0000 | 2.6471 | 0.3142 | 0.5907 | 13.7000 | 32.0000 | 32.9000 |
| Qwen2.5-1.5B | dsd-adaptive-draft | sequential | 20.0000 | 2.9225 | 0.2862 | 0.5907 | 13.7000 | 32.0000 | 32.9000 |
| Qwen2.5-1.5B | dsd-adaptive-draft | sequential | 50.0000 | 3.1491 | 0.2653 | 0.5907 | 13.7000 | 32.0000 | 32.9000 |
| Qwen2.5-1.5B | dsd-adaptive-draft | sequential | 100.0000 | 3.9148 | 0.2155 | 0.5907 | 13.7000 | 32.0000 | 32.9000 |
| Qwen2.5-1.5B | dsd-adaptive-draft | sequential | 150.0000 | 4.6364 | 0.1827 | 0.5907 | 13.7000 | 32.0000 | 32.9000 |
| Qwen2.5-1.5B | dsd-adaptive-draft | sequential | 200.0000 | 5.3107 | 0.1605 | 0.5907 | 13.7000 | 32.0000 | 32.9000 |
| Qwen2.5-1.5B | target-only | sequential | 0.0000 | 0.8302 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only | sequential | 5.0000 | 0.8302 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only | sequential | 10.0000 | 0.8302 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only | sequential | 20.0000 | 0.8302 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only | sequential | 50.0000 | 0.8302 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only | sequential | 100.0000 | 0.8302 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only | sequential | 150.0000 | 0.8302 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only | sequential | 200.0000 | 0.8302 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | vanilla-spec | sequential | 0.0000 | 3.1899 | 0.2725 | 0.4540 | 12.8000 | 32.0000 | 50.5000 |
| Qwen2.5-1.5B | vanilla-spec | sequential | 5.0000 | 3.3002 | 0.2633 | 0.4540 | 12.8000 | 32.0000 | 50.5000 |
| Qwen2.5-1.5B | vanilla-spec | sequential | 10.0000 | 3.4326 | 0.2540 | 0.4540 | 12.8000 | 32.0000 | 50.5000 |
| Qwen2.5-1.5B | vanilla-spec | sequential | 20.0000 | 3.6285 | 0.2410 | 0.4540 | 12.8000 | 32.0000 | 50.5000 |
| Qwen2.5-1.5B | vanilla-spec | sequential | 50.0000 | 3.9264 | 0.2228 | 0.4540 | 12.8000 | 32.0000 | 50.5000 |
| Qwen2.5-1.5B | vanilla-spec | sequential | 100.0000 | 4.6067 | 0.1904 | 0.4540 | 12.8000 | 32.0000 | 50.5000 |
| Qwen2.5-1.5B | vanilla-spec | sequential | 150.0000 | 5.3019 | 0.1664 | 0.4540 | 12.8000 | 32.0000 | 50.5000 |
| Qwen2.5-1.5B | vanilla-spec | sequential | 200.0000 | 5.9143 | 0.1493 | 0.4540 | 12.8000 | 32.0000 | 50.5000 |
| SmolLM | dsd-adaptive-draft | sequential | 0.0000 | 2.8671 | 0.3143 | 0.5382 | 13.0000 | 32.0000 | 37.3000 |
| SmolLM | dsd-adaptive-draft | sequential | 5.0000 | 2.9657 | 0.3037 | 0.5382 | 13.0000 | 32.0000 | 37.3000 |
| SmolLM | dsd-adaptive-draft | sequential | 10.0000 | 3.1025 | 0.2906 | 0.5382 | 13.0000 | 32.0000 | 37.3000 |
| SmolLM | dsd-adaptive-draft | sequential | 20.0000 | 3.3188 | 0.2712 | 0.5382 | 13.0000 | 32.0000 | 37.3000 |
| SmolLM | dsd-adaptive-draft | sequential | 50.0000 | 3.5931 | 0.2510 | 0.5382 | 13.0000 | 32.0000 | 37.3000 |
| SmolLM | dsd-adaptive-draft | sequential | 100.0000 | 4.2616 | 0.2121 | 0.5382 | 13.0000 | 32.0000 | 37.3000 |
| SmolLM | dsd-adaptive-draft | sequential | 150.0000 | 4.9433 | 0.1829 | 0.5382 | 13.0000 | 32.0000 | 37.3000 |
| SmolLM | dsd-adaptive-draft | sequential | 200.0000 | 5.6385 | 0.1606 | 0.5382 | 13.0000 | 32.0000 | 37.3000 |
| SmolLM | target-only | sequential | 0.0000 | 0.8976 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| SmolLM | target-only | sequential | 5.0000 | 0.8976 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| SmolLM | target-only | sequential | 10.0000 | 0.8976 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| SmolLM | target-only | sequential | 20.0000 | 0.8976 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| SmolLM | target-only | sequential | 50.0000 | 0.8976 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| SmolLM | target-only | sequential | 100.0000 | 0.8976 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| SmolLM | target-only | sequential | 150.0000 | 0.8976 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| SmolLM | target-only | sequential | 200.0000 | 0.8976 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| SmolLM | vanilla-spec | sequential | 0.0000 | 3.2355 | 0.2807 | 0.4986 | 11.7000 | 32.0000 | 44.2000 |
| SmolLM | vanilla-spec | sequential | 5.0000 | 3.3131 | 0.2735 | 0.4986 | 11.7000 | 32.0000 | 44.2000 |
| SmolLM | vanilla-spec | sequential | 10.0000 | 3.3994 | 0.2668 | 0.4986 | 11.7000 | 32.0000 | 44.2000 |
| SmolLM | vanilla-spec | sequential | 20.0000 | 3.5846 | 0.2531 | 0.4986 | 11.7000 | 32.0000 | 44.2000 |
| SmolLM | vanilla-spec | sequential | 50.0000 | 3.8746 | 0.2344 | 0.4986 | 11.7000 | 32.0000 | 44.2000 |
| SmolLM | vanilla-spec | sequential | 100.0000 | 4.4848 | 0.2029 | 0.4986 | 11.7000 | 32.0000 | 44.2000 |
| SmolLM | vanilla-spec | sequential | 150.0000 | 5.0688 | 0.1794 | 0.4986 | 11.7000 | 32.0000 | 44.2000 |
| SmolLM | vanilla-spec | sequential | 200.0000 | 5.7133 | 0.1591 | 0.4986 | 11.7000 | 32.0000 | 44.2000 |

## Edge-Cloud Cycle at RTT=100 ms

| group | method | rtt_ms | edge_cloud_cycle_time | cloud_verify_share_of_cycle_percent | wait_share_of_cycle_percent | network_cycle_share_of_method_time_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | dsd-adaptive-draft / kv-cache | 100.0000 | 2.2602 | 39.3775 | 60.6140 | 57.7347 |
| Qwen2.5-1.5B | target-only / kv-cache | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen2.5-1.5B | vanilla-spec / kv-cache | 100.0000 | 2.6171 | 51.0814 | 48.9093 | 56.8111 |
| SmolLM | dsd-adaptive-draft / kv-cache | 100.0000 | 2.3337 | 44.2869 | 55.7044 | 54.7616 |
| SmolLM | target-only / kv-cache | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SmolLM | vanilla-spec / kv-cache | 100.0000 | 2.3787 | 50.8044 | 49.1865 | 53.0396 |
