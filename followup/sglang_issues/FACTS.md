## A1. Last passing rate under a 500 ms p95 TTFT bar (median of 3 repeats; 32 adapters, 16 slots, open loop 256 in / 128 out)
| Workload | vLLM 0.34.0 | SGLang 0.5.23 |
|---|---|---|
| Zipf popularity | 20 rps | 20 rps |
| uniform popularity | 10 rps | 10 rps |
files: results/raw/main/vllm/vllm_{zipf32,uni32}_r*_rep*.json ; results/raw/main_rerun1/sglang/sglang_{zipf32,uni32}_r*_rep*.json

## A2. p95 TTFT in ms, median (min-max) over 3 repeats, 32 adapters, 16 slots
| Workload | Rate | vLLM | SGLang | vLLM tok/s | SGLang tok/s |
|---|---|---|---|---|---|
| zipf32 | 5 rps | 47 (44-47) | 51 (49-52) | 651 | 654 |
| zipf32 | 10 rps | 44 (43-44) | 56 (54-67) | 1291 | 1297 |
| zipf32 | 20 rps | 50 (48-72) | 84 (73-163) | 2532 | 2551 |
| zipf32 | 40 rps | 1456 (958-1565) | 12142 (11841-13517) | 4620 | 3035 |
| zipf32 | 80 rps | 4663 (3316-5629) | 9560 (9441-9656) | 7108 | 4158 |
| uni32 | 5 rps | 51 (50-60) | 66 (65-67) | 650 | 653 |
| uni32 | 10 rps | 51 (46-51) | 81 (72-84) | 1286 | 1295 |
| uni32 | 20 rps | 684 (496-938) | 755 (591-1138) | 2482 | 2499 |
| uni32 | 40 rps | 3075 (2277-3081) | 13058 (12590-13529) | 4443 | 2975 |
files: as A1

## A3. 32 adapters, 32 slots (Zipf): p95 TTFT ms
| Rate | vLLM max_loras 8 | vLLM 16 | vLLM 32 | SGLang 16 slots | SGLang 32 slots |
|---|---|---|---|---|---|
| 20 rps | 1868 (1535-2819) | 50 (48-72) | 95 (94-98) | 97 (76-146) | 46 (40-48) |
| 40 rps | 4675 (4669-5794) | 1456 (958-1565) | 142 (136-146) | 12031 (11807-13764) | 101 (94-108) |
| 80 rps | - | 4663 (3316-5629) | 2381 (2121-2793) | 9672 (9656-9796) | 130 (126-131) |
files: results/raw/sweep_m8/vllm, results/raw/main/vllm, results/raw/sweep_m32/vllm, results/raw/sgl_flags/control, results/raw/sgl_flags/slots32 (the SGLang 16-slot control here is a fresh same-session run; the main-grid value is in A2)

## A4. SGLang at 16 slots, Zipf, 32 adapters: scheduling options (p95 TTFT ms, median (min-max); tok/s)
| Variant | 20 rps | 40 rps | 80 rps | tok/s at 40 | tok/s at 80 |
|---|---|---|---|---|---|
| defaults | 97 (76-146) | 12031 (11807-13764) | 9672 (9656-9796) | 3022 | 4111 |
| --lora-drain-wait-threshold 1.0 | 85 (71-146) | 10111 (9026-10597) | 9279 (9037-9441) | 2960 | 3968 |
| --enable-lora-overlap-loading --max-loaded-loras 32 | 112 (64-130) | 11977 (11420-13422) | 9639 (9490-9685) | 3033 | 4157 |
| --lora-eviction-policy fifo | 175 (107-6680) | 12168 (11874-13709) | 9640 (9453-9815) | 3017 | 4120 |
| --max-loras-per-batch 32 (32 slots) | 46 (40-48) | 101 (94-108) | 130 (126-131) | 4040 | 6081 |
files: results/raw/sgl_flags/<variant>/sglang_zipf32_r*_rep*.json ; first overlap attempt failed at startup (assertion on --max-loaded-loras) and was rerun as overlap2

## A5. 128 adapters on 32 slots (exploratory), p95 TTFT ms median (min-max)
| Workload | Rate | vLLM (max_loras 32) | SGLang (max_loras_per_batch 32) |
|---|---|---|---|
| zipf128 | 5 rps | 50 (48-62) | 50 (49-53) |
| zipf128 | 10 rps | 70 (67-72) | 64 (62-70) |
| zipf128 | 20 rps | 114 (98-115) | 83 (73-5806) |
| zipf128 | 40 rps | 3011 (2424-3057) | 11662 (11200-13399) |
| zipf128 | 80 rps | 5537 (5077-5628) | 12891 (11856-13170) |
| uni128 | 5 rps | 57 (56-70) | 69 (67-71) |
| uni128 | 10 rps | 81 (78-89) | 88 (76-91) |
| uni128 | 20 rps | 816 (564-1520) | 352 (251-1033) |
| uni128 | 40 rps | 5440 (4817-5481) | 17408 (16593-17742) |
| uni128 | 80 rps | - | 17544 (17331-17898) |
files: results/raw/exp1/{vllm,sglang}/*_{zipf128,uni128}_r*_rep*.json

## B1. csgmv vs triton, closed loop, 32 Zipf adapters, 16 slots, 3 repeats, 500 requests per run (50 warm-up excluded)
| Concurrency | Metric | triton median (min-max) | csgmv median (min-max) | change in median | ranges overlap |
|---|---|---|---|---|---|
| 8 | TTFT p95 ms | 146.4 (140.8-152.9) | 140.4 (135.5-143.7) | -4.1% | yes |
| 8 | TTFT p50 ms | 99.7 (95.7-101.9) | 95.9 (87.8-98.5) | -3.8% | yes |
| 8 | TPOT p95 ms | 6.4 (6.4-6.4) | 6.4 (6.4-6.4) | -0.1% | yes |
| 8 | end-to-end p95 ms | 918.9 (913.8-921.9) | 913.1 (911.7-918.2) | -0.6% | yes |
| 8 | output tokens/s (mean) | 1159 | 1162 | +0.3% | |
| 32 | TTFT p95 ms | 391.0 (357.1-1176.3) | 347.3 (333.4-1141.0) | -11.2% | yes |
| 32 | TTFT p50 ms | 275.3 (273.8-283.7) | 252.3 (239.2-259.3) | -8.4% | no |
| 32 | TPOT p95 ms | 7.8 (7.7-7.9) | 7.4 (7.3-7.5) | -5.6% | no |
| 32 | end-to-end p95 ms | 1216.3 (1188.6-2181.6) | 1161.9 (1146.3-2099.9) | -4.5% | yes |
| 32 | output tokens/s (mean) | 3361 | 3500 | +4.1% | |
| 128 | TTFT p95 ms | 5437.3 (5401.4-5488.0) | 5183.3 (5168.7-5187.4) | -4.7% | no |
| 128 | TTFT p50 ms | 761.9 (725.5-781.6) | 632.9 (616.8-642.4) | -16.9% | no |
| 128 | TPOT p95 ms | 37.9 (37.7-38.5) | 36.1 (35.8-36.2) | -4.8% | no |
| 128 | end-to-end p95 ms | 9991.7 (9935.0-10171.7) | 9632.1 (9589.0-9663.5) | -3.6% | no |
| 128 | output tokens/s (mean) | 2269 | 2371 | +4.5% | |
files: results/raw/repA/sglang_{triton,csgmv}_c{8,32,128}_rep{0,1,2}.json ; references: results/raw/repA/ref_{triton,csgmv}_{start,end}.json
reference run drift, triton: e2e p50 start 794 ms, end 795 ms (+0.1%)
reference run drift, csgmv: e2e p50 start 808 ms, end 814 ms (+0.8%)
