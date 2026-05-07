# Group Meeting Timing Data

This file contains data snippets only. Use it as input for the final group-meeting report.

## Break-Even Notes

- Qwen2.5-1.5B / target-only-sampling: no positive break-even; RTT=0 speedup=1.0000x.
- Qwen2.5-1.5B / traditional-spec-sampling: no positive break-even; RTT=0 speedup=0.1303x.

## Stage Highlights at RTT=100 ms

- Qwen2.5-1.5B / target-only-sampling / kv-cache: top stages at RTT=100 ms are cloud_verification=93.07%, prefill=3.32%, probability_normalize=2.41%.
- Qwen2.5-1.5B / traditional-spec-sampling / kv-cache: top stages at RTT=100 ms are cloud_verification=32.32%, network_wait=29.50%, draft_generation=19.09%.

## Stage Share Table at RTT=100 ms

| group | method | rtt_ms | method_time | speedup_vs_target_only | draft_generation_percent | network_wait_percent | cloud_verification_percent | probability_normalize_percent | random_sample_percent | accept_reject_percent | resample_percent | cache_update_percent | sampling_argmax_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | target-only-sampling / kv-cache | 100.0000 | 0.9318 | 1.0000 | 0.0000 | 0.0000 | 93.0667 | 2.4058 | 0.6842 | 0.0000 | 0.0000 | 0.2149 | 0.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling / kv-cache | 100.0000 | 10.8471 | 0.0860 | 19.0922 | 29.5009 | 32.3202 | 2.1208 | 0.3145 | 0.0527 | 0.0473 | 15.6483 | 0.0000 |

## Method Summary

| group | method_name | target_verify_mode | temperature | top_k | top_p | stochastic_seed | rtt_ms | method_time | speedup_vs_target_only | accept_rate | rounds | generated_tokens | drafted_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | target-only-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 0.0000 | 0.9318 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 5.0000 | 0.9318 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 10.0000 | 0.9318 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 20.0000 | 0.9318 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 50.0000 | 0.9318 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 100.0000 | 0.9318 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 150.0000 | 0.9318 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | target-only-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 200.0000 | 0.9318 | 1.0000 | 0.0000 | 0.0000 | 32.0000 | 0.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 0.0000 | 7.1484 | 0.1303 | 0.0000 | 32.0000 | 32.0000 | 122.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 5.0000 | 7.5042 | 0.1242 | 0.0000 | 32.0000 | 32.0000 | 122.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 10.0000 | 7.6961 | 0.1211 | 0.0000 | 32.0000 | 32.0000 | 122.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 20.0000 | 8.3524 | 0.1116 | 0.0000 | 32.0000 | 32.0000 | 122.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 50.0000 | 9.0769 | 0.1027 | 0.0000 | 32.0000 | 32.0000 | 122.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 100.0000 | 10.8471 | 0.0860 | 0.0000 | 32.0000 | 32.0000 | 122.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 150.0000 | 12.6475 | 0.0737 | 0.0000 | 32.0000 | 32.0000 | 122.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling | sequential | 1.0000 | 20 | 0.9000 | 123 | 200.0000 | 14.3520 | 0.0649 | 0.0000 | 32.0000 | 32.0000 | 122.0000 |

## Edge-Cloud Cycle at RTT=100 ms

| group | method | rtt_ms | edge_cloud_cycle_time | cloud_verify_share_of_cycle_percent | wait_share_of_cycle_percent | network_cycle_share_of_method_time_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | target-only-sampling / kv-cache | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen2.5-1.5B | traditional-spec-sampling / kv-cache | 100.0000 | 6.7064 | 52.2756 | 47.7155 | 61.8266 |
