# Qwen3.8-27B 复现 SPORK 核心实验计划 V2

版本：2026-08-31
项目：`self-speculation`
模型：`Qwen/Qwen3.8-27B`
硬件：8 x NVIDIA A100 40GB
论文：[SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference](https://arxiv.org/html/2607.03333v1)

## 1. 为什么重写计划

第一次 core-E2E 运行证明当前 harness 会减速，但没有实现论文最核心的收益路径：

```text
probe 只生成了工具调用文本
-> 没有提前执行工具
-> strict match 后没有复用投机结果
-> main 重新执行工具
```

同时，控制器在 main 完成后继续等待 fork/draft，D2 没有产生可观测的筛选行为，D3
accepted token 无法与 ngram proposer 分开归因，tau2 `success` 也只是 completed rate，不是质量
grader。旧 full-155 因此只证明“当前不完整实现显著变慢”，不能用于判定论文中的完整 SPORK。

V2 的目标不是立刻重跑大样本，而是先证明每一个收益项真实存在，再逐级扩大实验。

### 1.1 本轮执行范围：Quick10 时间诊断

当前轮次不执行完整 Gate 2-5、airline-43、floor sweep 或 full-155，只回答一个问题：修正后的完整
SPORK 在当前实现上，端到端时间点估计是变快还是变慢。

固定设计：

| 项目 | 设置 |
|---|---|
| 数据集 | `tau2-bench` 的 `airline` domain |
| 唯一任务数 | 从既有 airline-43 manifest 固定抽取10个，运行前冻结 task ID |
| 抽样规则 | 对43个 task ID 排序后使用 `seed=42` 无放回抽10个；不得人工换题 |
| 数据要求 | 每条至少有一个可投机的只读工具机会；不得按运行后的速度挑选 |
| baseline | `baseline-ngram` |
| treatment | `d1-d2-d3` |
| server | 两者使用相同 composite ngram+boundary server 配置 |
| tool floor | 2秒 |
| repeat | 1 |
| measured runs | 10 x 2 = 20次任务运行 |
| warmup | 每个配置10次丢弃请求，共20次，不计入延迟 |
| task concurrency | 1 |

必须同时记录：task wall time、main/fork/tool/draft 时间线、真实工具 overlap、结果复用、fork 尾部、
D3/ngram source-specific accepted token、质量 grader、server metrics 和 GPU utilization。

执行命令只有一条：

```bash
cp scripts/qwen38_v2/server.quick10.env.example scripts/qwen38_v2/server.env
# 填写全部 /ABS/... 路径后：
scripts/qwen38_v2/run_v2.sh quick
```

结果写入：

```text
runs/v2/quick10/<timestamp>/analysis/QUICK10_SUMMARY.json
runs/v2/quick10/<timestamp>/analysis/QUICK10_REPORT.md
```

Quick10 的 `ratio_of_means = mean(baseline) / mean(treatment)` 仍是主时间指标。10条任务的 CI 只作
诊断，不用于宣称论文复现。若机制门失败，脚本先写报告再返回非零；若明显变慢，停止扩大样本并根据
时间线优化；只有出现正向点估计且机制、质量均通过，才讨论增加 repeat 或任务数。

## 2. 研究目标与最终结论等级

V2 优先复现论文的核心结论：

1. D1 能利用 prefix cache 以较低成本产生早期工具预测。
2. 对只读工具，probe 能提前执行工具；严格匹配时结果被 main 真实复用。
3. D2 能在有区分度的 workload 上过滤低置信 probe 或触发后续 retry。
4. D3 能在 matched-ngram baseline 上回收 probe token，且不改变权威 main 输出。
5. D1/D2/D3 在保持任务质量的条件下改善端到端 mean 或 P95 latency。

结论分四级，禁止越级：

| 等级 | 可以声明的结论 | 必要条件 |
|---|---|---|
| L1 | early-tool-intent reproduced | 协议审计和 R1 raw audit 通过 |
| L2 | SPORK execution mechanism works | 提前执行、strict match、结果复用和 real overlap 均大于0 |
| L3 | D3 mechanism works | matched-ngram、source-specific accepted token、输出等价门禁通过 |
| L4 | core end-to-end speedup reproduced | 质量门通过，预注册速度指标 CI 通过 |

只有达到 L4，报告标题才可使用：

```text
SPORK core end-to-end reproduction on Qwen3.8-27B
```

否则必须明确写成 mechanism reproduction、partial reproduction 或 negative result。

## 3. 已有结果如何处理

以下结果不直接重跑，但必须先同步原始产物并重新审计：

- Phase 1 JSON 协议资格测试。
- Phase 1 forced-prefix、stop sequence 和数据隔离 audit。
- R1 full-155 replay-pos0：name 92.8%，args-exact 89.3%。

以下结果保留为历史诊断，不进入 V2 的论文结论：

- `runs/core-e2e/20260829T050024Z`：正确性 bug，标记 invalid。
- `runs/core-e2e/20260829T174900Z`：不完整 SPORK 实现，mean speedup 0.842x。

旧数据不得删除或覆盖。V2 使用全新的目录：

```text
runs/v2/
results/v2/
```

## 4. 冻结实验环境

正式实验统一使用：

| 项目 | 设置 |
|---|---|
| GPU | GPU 0,1，其他 GPU 在正式 latency run 中空闲 |
| tensor parallel | TP=2 |
| dtype | BF16 |
| temperature | 0 |
| seed | 42 |
| max model length | 32768 |
| max sequences | 8 |
| prefix caching | enabled |
| built-in MTP | disabled |
| task concurrency | 1，除非另建 throughput 实验 |
| warmup | 每次 server restart 后10个丢弃请求 |

必须冻结并记录：

- 模型 revision、tokenizer revision 和权重 SHA256/目录清单。
- Python、PyTorch、Transformers、vLLM、CUDA 和 driver 版本。
- `self-speculation`、实验 harness 和 tau2-bench commit。
- 完整 server command、环境变量和 GPU 拓扑。
- 每套 task manifest 及 SHA256。

任何影响模型输出或调度的版本变化都必须生成新的 run group，不能与旧 run 合并。

## 5. Baseline 与实验配置

论文没有一个跨所有 serving mode 通用的 baseline。V2 分成两条独立对照，禁止跨组计算加速比。

### 5.1 Plain HTTP 对照：验证 D1/D2 工具重叠

使用 plain vLLM server，不启用 token-level speculative decoding：

| 配置 | main decode | fork | 提前执行工具 | D2 | D3 |
|---|---|---|---|---|---|
| `baseline-serial` | plain | 无 | 无 | 无 | 无 |
| `d1` | plain | first-token cached fork | 是 | 无 | 无 |
| `d1-d2` | plain | cached fork + retry | 是 | 是 | 无 |

该组回答：提前执行只读工具是否产生真实 overlap，以及 D1/D2 的净收益。

### 5.2 Composite engine 对照：验证完整 D1/D2/D3

使用同一个 composite ngram + SPORK-boundary proposer server：

| 配置 | ngram proposer | SPORK fork | boundary draft |
|---|---|---|---|
| `baseline-ngram` | 开启 | 无 | 不注册 |
| `d1-ngram` | 开启 | D1 | 不注册 |
| `d1-d2-ngram` | 开启 | D1+D2 | 不注册 |
| `d1-d2-d3` | 开启 | D1+D2 | 注册并注入 |

其中决定性 D3 对照是：

```text
baseline-ngram vs d1-d2-d3
```

两者必须使用相同 server binary、proposer、TP、GPU、KV cache 设置和 ngram 参数。baseline 只是不
注册 SPORK boundary draft。禁止使用 plain baseline 与 composite treatment 直接计算 D3 speedup。

## 6. SPORK 工具执行的权威语义

### 6.1 只允许投机只读工具

为所有工具建立冻结 manifest：

```json
{
  "tool_name": {
    "read_only": true,
    "idempotent": true,
    "speculation_allowed": true
  }
}
```

write、non-idempotent 或未分类工具一律走 serial fallback。不得为了提高接受率把写工具标为只读。

### 6.2 正确控制流

每个 agent turn 必须遵守：

```text
1. 启动权威 main stream。
2. 收到 main 第一个严格 token 后启动 fork。
3. fork 解码出合法工具调用，且通过 D2 gate。
4. 若工具允许投机，立即在隔离 executor 中启动 speculative tool future。
5. main 继续生成，不受 fork 结果控制。
6. main 输出最终工具调用后，规范化比较完整 name + arguments。
7. strict match：复用 speculative future/result，不再次执行工具。
8. mismatch：丢弃或取消 speculative future，执行 main 的 canonical tool call。
9. 只有 main 的 canonical call/result 可以写入 conversation history。
```

canonical arguments 使用 JSON 语义规范化：对象 key 排序、数值/布尔类型保留、禁止只做字符串或
tool-name 比较。

### 6.3 取消和尾部等待

main 完成时：

- fork 尚未形成合法调用：立即取消 fork，执行 main canonical call。
- fork 已形成调用但不匹配：取消/丢弃 speculative future，不等待其完成。
- strict match 且 speculative tool 仍在运行：只等待工具剩余时间，这是允许的 residual wait。
- D3 注入窗口已关闭：取消未提交 draft；已注册 draft 进行一次 request-scoped clear。

不得为了收集完整 `fork_text` 而把过期 fork 排空到结束。不得重复 clear。

### 6.4 隔离与副作用

tau2 每个 baseline/SPORK episode 使用新的环境实例。投机只读工具使用独立快照或明确线程安全的
只读 executor。mismatch 的结果不得改变权威环境。对无法真正取消的线程/网络请求，允许其后台
结束，但必须隔离、记录为 wasted execution，并在资源账本中计数。

## 7. D3 的权威语义

D3 只做 token-level draft verification：

```text
probe tool-call body
-> tokenize(add_special_tokens=False)
-> request-scoped registration
-> main 到达锁定的 tool-call boundary
-> proposer 提供最多20个 draft tokens
-> target model 接受最长匹配前缀
-> 第一个不匹配位置恢复正常 decode
```

要求：

- D3 不复制 probe KV cache。
- D3 不直接相信 probe token。
- target verifier 决定每个 token 是否接受。
- temperature=0 时，D3 的权威 main 工具调用必须与 no-boundary-draft 对照一致。
- composite proposer 在没有 boundary draft 时必须退回完全相同的 ngram proposer。

必须分别记录 `source=ngram` 和 `source=spork_boundary` 的 proposed/accepted tokens，禁止用混合的
vLLM 全局 counter 证明 SPORK D3 的收益。

## 8. 必须新增的逐任务和逐 turn 记录

所有时间戳来自同一个 monotonic clock。每个 task 至少记录：

```text
run_id, config, repeat, task_id, domain
task_start, task_end, task_wall_s
status, error, timeout
quality_reward, action_match, final_answer
n_turns, n_tool_turns
```

每个 turn 至少记录：

```text
main_start
main_first_token
main_tool_call_decoded
main_completed
fork_start
fork_tool_call_decoded
fork_completed | fork_cancel_requested | fork_cancelled
probe_confidence, probe_attempt, d2_threshold
speculative_tool_start, speculative_tool_end
canonical_tool_start, canonical_tool_end
strict_match
speculative_result_reused
speculative_execution_wasted
duplicate_tool_execution
real_overlap_ms
residual_tool_wait_ms
obsolete_fork_tail_ms
draft_submit_start, draft_submit_end
draft_boundary_hit
d3_proposed_tokens, d3_accepted_tokens
ngram_proposed_tokens, ngram_accepted_tokens
draft_clear_start, draft_clear_end, draft_clear_count
```

`real_overlap_ms` 使用实际时间区间交集计算：

```text
intersection(
  [speculative_tool_start, speculative_tool_end],
  [speculative_tool_start, main_tool_call_decoded]
)
```

它不能用工具 floor、probe accuracy 或整轮 wall time代替。

同时报告两个完成时间：

- `task_wall_s`：安全返回最终答案所需的关键路径，包含必要取消和状态 fencing。
- `resource_drained_s`：所有后台投机执行和远端 draft 清理真正结束的时间。

主加速指标使用 `task_wall_s`，但必须同步报告 `resource_drained_s`，防止通过隐藏后台资源消耗制造
虚假低延迟。

## 9. 正式运行前的硬门禁

### Gate 0：原始产物和协议审计

同步并核验 Phase 1/R1 raw、manifest、bootstrap JSON、locked protocol 和 server command。

通过条件：

- forced prefix 的 UTF-8 bytes 和 token IDs 正确。
- prefix 只存在于 fork。
- stop sequence 不截断参数。
- dev/smoke/formal manifests 无泄漏。
- R1 数字可从 raw 独立重算。

### Gate 1：纯 Python 正确性测试

使用 fake main、fake fork 和 deterministic tool executor 覆盖：

1. strict match 复用结果且工具只执行一次。
2. 同名不同参数拒绝投机结果。
3. fork 有调用、main 无调用时不提交。
4. write 工具永不提前执行。
5. mismatch 不改变权威环境。
6. main 先结束时取消过期 fork。
7. accepted future 未完成时只等待 residual tool time。
8. draft clear 恰好一次。
9. D3 不改变 greedy main tokens。
10. 多轮状态只由 main 驱动。

任一测试失败，禁止启动 GPU 正式实验。

### Gate 2：在线 timeline smoke

先使用一个合成只读工具案例，再使用冻结 smoke10：

- `speculative_tool_start < main_tool_call_decoded` 至少出现一次。
- `speculative_result_reused=true` 至少出现一次。
- accepted turn 的 `real_overlap_ms > 0`。
- accepted turn 无 canonical duplicate execution。
- mismatch turn 正确 fallback。
- obsolete fork 收到 cancel，不被 drain。
- `draft_clear_count <= 1`。
- main/fork 原始文本和完整事件时间线已落盘。

如果 smoke10 没有只读 strict-match turn，扩大到 airline-43，但不得降低门槛。

### Gate 3：D1 prefix-cache microbenchmark

对同一批固定 prompt 比较：

```text
A. main only
B. main 与 cold fork 同时启动
C. main 首 token 后启动 cached fork
```

至少记录 prompt tokens、cached tokens、prefill wall、fork decode wall 和 main TPOT。

通过条件：

- C 的共享 prefix token 数与预期一致。
- cached fork prefill 不超过 cold fork 的50%。
- C 相对 A 的 main TPOT 回归不超过1%。
- B/C 的差异方向与 prefix cache 生效一致。

达不到时先修 KV cache 命中，不能用增加工具 floor 掩盖 probe 开销。

### Gate 4：D2 行为门禁

在冻结 calibration set 上记录每次 probe 的 name-span confidence、正确性和 retry。阈值只由
calibration set 选择，正式集不得调参。

如果 tau2 中 confidence 全部饱和，必须写：

```text
D2 not identifiable on this workload
```

此时 `d1-d2` 可以作为等价配置运行，但不能声明 D2 有收益。后续使用 HotpotQA 或 GAIA-like
workload 单独验证 D2。

### Gate 5：D3 matched-ngram 门禁

固定20条任务依次验证：

1. stock ngram 与 composite/no-draft 输出完全一致。
2. 两者 ngram proposed/accepted 统计在预注册容差内。
3. pure-boundary smoke 的 SPORK accepted tokens 大于0。
4. 正式 workload 中 `spork_boundary_accepted_tokens > 0`。
5. 对捕获的相同 main request，在固定调度的等价性测试中，D3 与 no-boundary-draft 的
   greedy tool-call tokens 一致。
6. active requests 最终为0，无 request-ID 串线和 worker 泄漏。

如果固定请求的等价性测试失败，判 Gate 5 失败。正式 E2E 中若出现系统性的 treatment-only
action/turn 膨胀，必须先暂停并区分 D3 bug、请求路由错误和 vLLM batching 数值差异，不能直接继续
扩大样本。

## 10. 数据集和运行阶段

### Stage A：smoke10，单次

目的：只验证正确性和时间线，不作性能结论。

```text
plain:     baseline-serial, d1, d1-d2
composite: baseline-ngram, d1-d2-d3
```

### Stage B：airline-43，2秒 floor，单次

目的：确认真实 agent loop 中有 accepted tool reuse、positive overlap、零质量回归和零轨迹异常。

只有以下全部满足才进入正式测量：

- reused speculative results > 0。
- total real overlap > 0。
- authority/safety violation = 0，任务质量 grader 不低于 baseline。
- treatment-only max_turns = 0。
- duplicate accepted tool execution = 0。
- write-tool speculative execution = 0。

Stage B 只作为预检，不与正式 repeats 合并。

### Stage C：airline-43 latency floor sweep

工具 floor：

```text
0.5s, 1s, 2s, 5s
```

每个 floor 运行3个 repeats。plain 和 composite 分开报告：

```text
plain:     baseline-serial vs d1 vs d1-d2
composite: baseline-ngram vs d1-d2-d3
```

目标：验证工具越慢，真实 overlap 和端到端收益是否总体增加。相邻点允许噪声反转，但不得选择性
删除不利 floor。

### Stage D：full-155，2秒 floor，3 repeats

决定性对照：

```text
plain primary:     baseline-serial vs locked best D1/D2
composite primary: baseline-ngram vs d1-d2-d3
```

必须同时按 airline/retail、turn count、CoT wall time、工具数、strict acceptance 和 overlap 分层。

### Stage E：HotpotQA 扩展

在 tau2 核心链路通过后再运行，用于：

- 验证 D2 confidence/retry 是否有区分度。
- 使用真实工具检测外部有效性。
- 报告 EM/F1 和真实网络延迟。

真实 API 组与 deterministic record/replay 组分开：真实组回答质量和外部有效性，replay 组回答
因果 latency。两组不得合并。

## 11. 重复、顺序和 cache 控制

论文公开代码默认按 config block 跑任务，没有充分说明正式结果的 repeat 和顺序平衡。V2 明确
预注册以下控制：

1. 每个正式配置3个 repeats。
2. repeat 间轮换配置顺序。
3. 每个配置 block 前重启相同 server，清空跨配置 prefix cache。
4. server restart 后执行相同的10个丢弃 warmup。
5. 每个 task/config 使用新的 agent/tool environment。
6. 同一 repeat 内使用相同 task manifest 和固定 task 顺序。
7. 正式 latency 期间 GPU 2-7 空闲，记录每秒 GPU utilization 和显存。

推荐顺序：

```text
repeat 1: baseline -> treatment
repeat 2: treatment -> baseline
repeat 3: baseline -> treatment
```

如果一次 server restart 后同时跑多个配置，必须证明跨配置 cache 不会让后运行配置获益，否则该
run 只能作为诊断。

## 12. 质量指标

不能再用 `status=completed` 冒充任务质量。

tau2 至少报告：

- action-match reward 或冻结任务定义的实际 grader。
- golden actions matched / total。
- 最终 DB state 或适用的环境检查。
- completed、max_turns、timeout 和 error 作为独立运行状态。

对于 deterministic tau2，baseline/SPORK canonical action trajectory 的一致率必须作为诊断指标，
但不能替代任务 grader。独立模型执行即使 temperature=0，也可能因 serving batching 的浮点数值差异
产生少量不同但同样正确的 action。真正的硬约束是：agent 状态只由 main 驱动、投机结果必须与
main 的完整 name+arguments 严格匹配、任务 grader 不退化。若 treatment 出现系统性额外 turns、
重复 action 或 grader 下降，必须暂停并审计，不能归因于正常噪声。

HotpotQA 报告 aggregate EM/F1。真实网络返回具有随机性，因此不要求 token-identical，但必须使用
相同任务、相同时间窗口和相同工具 backend。

## 13. 加速比和统计口径

### 13.1 原始延迟

每个配置都必须先报告：

```text
N, errors, timeouts
mean, P50, P95
每个 repeat 的 mean/P50/P95
```

### 13.2 主指标：均值之比

V2 预注册的 mean speedup：

```text
ratio_of_means = mean(B_i) / mean(S_i)
```

大于1表示 SPORK 更快。该指标回答总平均任务延迟是否下降，是 V2 mean verdict 的唯一主指标。

### 13.3 论文兼容指标

论文正文明确将主要 tail speedup 定义为：

```text
P95_speedup = P95(B_i) / P95(S_i)
```

同时报告延迟降幅：

```text
P95_reduction = 1 - P95(S_i) / P95(B_i)
```

论文 Figure 14 使用“mean per-task speedup”措辞，但没有公开完整聚合代码。为兼容该图，V2 将下列
指标作为 secondary，同时明确标签：

```text
mean_paired_ratio = mean(B_i / S_i)
geometric_mean_ratio = exp(mean(log(B_i / S_i)))
```

secondary 指标不得覆盖或替代主 `ratio_of_means` verdict。如果两者方向相反，结论必须写成
estimator-sensitive，并展示导致翻转的长短任务分层。

### 13.4 Bootstrap

使用至少10,000次 task-cluster paired bootstrap：

- 重采样单位是 task ID。
- 一个 task 的全部 repeats 和 turns 一起带入。
- 每次重采样后重新计算 ratio-of-means、mean latency difference、P50/P95 和质量差。
- 禁止把 turn、probe 或 repeat 当作独立样本。

报告：

```text
ratio_of_means 95% CI
mean(S-B) 95% CI
P95_speedup 95% CI
quality difference 95% CI
```

### 13.5 错误和缺失

不得静默排除 error/timeout。主 paired latency 可在双方均有效的 task 上计算，但必须同时报告：

- 完整 manifest 数量。
- 每个配置有效任务数量。
- 非对称 error/timeout。
- 将 timeout 计为预注册上限的 sensitivity analysis。

任一 treatment-only correctness error、未隔离副作用或数据泄漏都优先触发 NEGATIVE/BLOCKED，不能
靠删除任务恢复速度结论。

## 14. Verdict 规则

分别给出，不再只写一个模糊 overall：

### Mechanism verdict

`PASS` 需要：

- speculative tool dispatch > 0。
- strict accepted turns > 0。
- reused results > 0。
- real overlap > 0。
- duplicate accepted executions = 0。
- authority/safety violations = 0。

### Mean latency verdict

- `PASS`：ratio-of-means > 1 且95% CI下界 > 1，质量门通过。
- `INCONCLUSIVE`：点估计 > 1 但 CI 跨1，质量门通过。
- `NEGATIVE`：ratio-of-means <= 1，或质量门失败。

### Tail latency verdict

- `PASS`：P95 speedup > 1 且 bootstrap CI下界 > 1，质量门通过。
- `INCONCLUSIVE`：点估计 > 1 但 CI 跨1。
- `NEGATIVE`：P95 speedup <= 1，或质量门失败。

### D3 verdict

`PASS` 还要求：

- SPORK-boundary accepted tokens > 0，且可与 ngram 分开归因。
- no-draft equivalence 通过。
- 固定请求的 D3/no-draft greedy 输出等价门禁通过。
- 正式 workload 无未解释的系统性 action/turn 膨胀。
- matched-ngram P95 或 mean 至少一个预注册指标通过。

## 15. 必须产出的文件

每个 run group：

```text
runs/v2/<stage>/<timestamp>/
  commands.txt
  provenance/
    input-fingerprint.json
    plain-server-launcher.sh
    composite-server-launcher.sh
    locked-protocol.json
    manifest-*.json
    git-head.txt
    git-status.txt
    python-version.txt
    nvidia-smi.txt
  raw/tasks.jsonl
  raw/turns.jsonl
  raw/events.jsonl
  raw/warmups.jsonl
  logs/server-*.log
  logs/driver-*.log
  metrics-*-before.prom
  metrics-*-after.prom
  gpu-*.csv
  analysis/contract.json
  analysis/*-gate.json
  analysis/*-analysis.json
```

`raw/warmups.jsonl` 每个 `(config, repeat)` 必须恰好有10条成功且丢弃的请求。`events.jsonl`
必须能回溯每个 turn 的 main、fork、tool、draft 时间线。脚本会交叉检查 task 声明的 turn 数、event
归属和 warmup 数量；server metrics 或 GPU 采样缺失也会使当前 block 失败。

总报告：

```text
results/v2/CORE_E2E_V2_REPORT.md
results/v2/CORE_E2E_V2_SUMMARY.json
results/v2/FAILURES_AND_DEVIATIONS.md
results/v2/FAILURES_AND_DEVIATIONS.json
```

报告必须同时保留不利结果、所有失败任务和偏差，不得只保留最优 repeat。

## 16. 执行顺序和停止条件

本轮只执行上面的 `quick`，不要执行 `qualify`、`unlock` 或 `formal`。后续决定恢复完整复现时，才按
本节后面的完整流程运行。

本计划的可执行入口为：

```text
scripts/qwen38_v2/run_v2.sh
```

服务器一次性适配接口、字段契约和配置模板见：

```text
scripts/qwen38_v2/README.md
scripts/qwen38_v2/server.env.example
experiments/qwen38_v2/contract.py
```

执行器是 fail-closed：它不读取 Markdown 中的 PASS，不允许跳过前置 marker，不允许 plain 与
composite baseline 混比。每个 marker 同时绑定代码、server adapter、测试、协议、manifest、环境文件、
启动命令和完整阶段产物的 SHA256；其中任一内容变化，旧 marker 自动失效。

严格按以下顺序：

```text
同步旧 raw 并完成 Gate 0
-> 实现并测试正确工具执行语义
-> Gate 1 纯 Python 测试
-> Gate 2 在线 timeline smoke
-> Gate 3 D1 prefix-cache microbenchmark
-> Gate 4 D2 audit
-> Gate 5 D3 matched-ngram gate
-> Stage B airline-43 单次预检
-> 人工检查产物并执行 unlock
-> Stage C floor sweep
-> Stage D full-155
-> Stage E HotpotQA
```

核心 tau2 执行命令固定为：

```bash
scripts/qwen38_v2/run_v2.sh qualify
# 人工检查 runs/v2 下全部 gate JSON 和 raw 后：
scripts/qwen38_v2/run_v2.sh unlock
scripts/qwen38_v2/run_v2.sh formal
```

如需定位失败，可以改用展开的单阶段命令：

```bash
scripts/qwen38_v2/run_v2.sh preflight
scripts/qwen38_v2/run_v2.sh gate0
scripts/qwen38_v2/run_v2.sh gate1
scripts/qwen38_v2/run_v2.sh smoke
scripts/qwen38_v2/run_v2.sh cache
scripts/qwen38_v2/run_v2.sh d2
scripts/qwen38_v2/run_v2.sh d3
scripts/qwen38_v2/run_v2.sh precheck
# 人工检查 runs/v2 下全部 gate JSON 和 raw 后：
scripts/qwen38_v2/run_v2.sh unlock
scripts/qwen38_v2/run_v2.sh floor
scripts/qwen38_v2/run_v2.sh full
scripts/qwen38_v2/run_v2.sh report
```

当前脚本故意不伪造本机缺失的 tau2 driver。服务器现有 `experiments.qwen38` 必须按 README 暴露
`v2-e2e`、`v2-cache-gate`、`v2-d2-gate`、`v2-d3-gate` 四个子命令，并使用真实 tau2 环境和 grader。
接口或 `tests/test_core_e2e_v2_correctness.py` 未实现时，preflight/Gate 1 会停止，禁止开始 GPU 正式实验。

立即停止大规模实验的条件：

- `speculative_result_reused` 始终为0。
- accepted turn 出现 duplicate tool execution。
- write/non-idempotent 工具被提前执行。
- main 完成后仍无条件 drain 过期 fork。
- 固定请求的 D3/no-draft 等价性失败，或正式 workload 出现未解释的系统性 action/turn 膨胀。
- D3 accepted token 无法与 ngram 分开归因。
- 质量 grader 缺失。
- baseline/treatment task set 不相同。
- 统计器没有按预注册估计量计算 bootstrap。

出现这些问题时只允许修代码和跑 smoke，不允许继续 full-155。

## 17. V2 完成标准

本计划完成不等于所有结果必须为正。以下产物齐全即可完成实验：

1. 完整 SPORK 工具提前执行、strict match 和结果复用链路。
2. 可核验的 real overlap、tail wait、D3 source-specific token 账本。
3. plain 与 composite 两套匹配 baseline。
4. 正确的 tau2 quality grader 和运行状态。
5. floor sweep 与 full-155 的3-repeat paired raw 数据。
6. ratio-of-means、P95 ratio、mean paired ratio 和 bootstrap CI。
7. 明确的 mechanism、mean、tail、D3 四项 verdict。

如果修正后的完整 SPORK 仍然显著变慢，应报告为可信的 Qwen3.8/A100 negative result；如果通过，
则可以声明复现了论文的核心端到端结论。两种结果都比继续运行不完整 harness 有价值。
