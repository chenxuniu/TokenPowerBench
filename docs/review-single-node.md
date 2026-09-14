# 单机功耗与 Prefill/Decode 审查记录

审查基线：Git 提交 `ec00f52`。以下行号均指该提交，避免修改后的行号变化影响复核。
范围是权限与功耗计算，以及首 token 和阶段时间窗；本次不评价多机性能，也不据此
推断论文实验数据必然存在同样问题。历史代码能够完成推理和传感器采集，但采集成功
与阶段归因正确是两个不同的验收条件。

## 发现与处理

| 级别 | 基线证据 | 对复现的影响 | 本次处理 |
| --- | --- | --- | --- |
| P1 | [`run_single_node.py:110–123`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/run_single_node.py#L110-L123)；[`SingleNode/llm_benchmark.py:150–171`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/SingleNode/llm_benchmark.py#L150-L171) | 单机入口只拿整段 inference 的起止时间，没有首 token 时间，因此不能支持 prefill/decode 功耗分解。 | 在维护中的单机 vLLM 入口增加逐请求、逐 step 的首 token 观测和两个阶段的时间窗；要求 batch size 1。 |
| P1 | [`MultipleNode/vllm_engine.py:481–494`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/MultipleNode/vllm_engine.py#L481-L494) | `first_token_generated` 在阻塞式 `generate()` 返回以后才记录，实际已经生成完整个响应；该字段不是 TTFT。 | 将其作为旧实现的明确限制记录；新单机路径在首次返回 token ID 时记录。旧多机脚本不宣称已修复。 |
| P1 | [`tokenpowerbench/energy/gpu_monitor.py:23–34, 87–109`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/gpu_monitor.py#L87-L109)；[`SingleNode/power_monitor.py:377–394`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/SingleNode/power_monitor.py#L377-L394) | 样本未保留时间戳，所谓按时间过滤实际上删首尾 10%，再用均值乘时长。它无法精确排除初始化/空闲尾部，也无法对齐首 token。 | 功率样本保留单调时钟时间戳，按真实区间积分；阶段采样不足返回 null，并保留原始轨迹。 |
| P1 | [`tokenpowerbench/energy/full_node_monitor.py:136–139, 294–298`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/full_node_monitor.py#L294-L298) | IPMI 使用默认上限 1000 W 的 `_robust_mean`，多卡服务器正常超过 1000 W 的节点功率会被全部过滤，最终可能变成 0。 | 不使用这个通用硬上限过滤节点读数。传感器失败与实际 0 W 分开表示。 |
| P1 | [`tokenpowerbench/energy/full_node_monitor.py:122–128, 191–212`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/full_node_monitor.py#L122-L128) | RAPL 同时发现 package 和 core/uncore 子域，却把所有非 DRAM 域相加，可能重复计入 CPU 能耗。 | CPU 汇总只取 package 域；独立 DRAM 域单列，按读取到的真实计数器范围处理回绕。 |
| P2 | [`tokenpowerbench/energy/__init__.py:65–72`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/__init__.py#L65-L72)；[`tokenpowerbench/energy/full_node_monitor.py:117–139, 237–240`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/full_node_monitor.py#L117-L139) | auto 只由 RAPL 决定，错过 IPMI 可读但 RAPL 不可读的情况；full_node 可以静默缺传感器，读取失败写为 0。 | 先报告进程身份，再独立探测 RAPL/IPMI；gpu_only 不探测这两者；full_node 仅要求 IPMI 成功，CPU/DRAM 独立可选；缺失值统一 null，并记录能力与原因。 |
| P2 | [`tokenpowerbench/energy/gpu_monitor.py:49–51`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/tokenpowerbench/energy/gpu_monitor.py#L49-L51) | NVML 枚举全机 GPU，而推理使用 CUDA 可见设备；限制 `CUDA_VISIBLE_DEVICES` 后可能把其他作业的 GPU 计入。 | 根据推理进程可见设备的 UUID 选择 NVML 设备，并保存设备身份。 |
| P2 | [`SingleNode/llm_benchmark.py:88, 147–163`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/SingleNode/llm_benchmark.py#L147-L163)；[`SingleNode/power_monitor.py:307–318`](https://github.com/chenxuniu/TokenPowerBench/blob/ec00f52/SingleNode/power_monitor.py#L307-L318) | 旧脚本复用监控对象，但每次 stop 都关闭 NVML，下一批次未重新初始化，读失败后又写 0，后续批次可能得到错误 GPU 能耗。 | 维护中的监控器支持重复 start/stop，最终关闭与停止采样分离；旧单机入口也通过兼容适配器使用修正后的监控器。 |

P1 表示会使主要测量含义或数值失真，应在作为复现依据前修复；P2 表示特定权限、
设备选择或重复运行条件下会影响正确性。上表修复范围是 `run_single_node.py`、`tokenpowerbench/` 和旧单机监控适配器，
不是历史四引擎代码的全面改写。

## 对两个核心问题的结论

权限说明应直接告诉使用者：作者当时用 IPMI 每秒读取整机功率，用 Intel RAPL
sysfs 读取 CPU 能量计数器。这两类访问通常需要 root，或管理员事先授予相应权限。
如果两者都没有授权，普通用户只得到 NVML GPU 结果；不能用 GPU 或组件之和冒充
整机 IPMI，也不能把不可读的 CPU/node 数值写成 0。IPMI 已覆盖的节点能耗不能再
加上 GPU/CPU/DRAM，否则又会重复计算。

每次启动（包括 `--check-monitor`）先报告真实 UID/GID、有效 UID/GID 和
`is_root`，正式运行保存为 `runtime.json`。`is_root` 仅依据有效 UID 是否为 0；
实际传感器能力另行探测和保存，不能由 root 身份推断。GH200 的 Grace ARM CPU
没有 Intel RAPL 接口，即使 root 运行也没有这套 CPU/DRAM 读数；但只要 IPMI
读取成功，仍可使用 `full_node` 记录整机能耗。缺 RAPL 不应阻塞有效的整机测量。

Prefill/decode 的原始单机实现缺少可审查的分界；另一个目录中虽然存在
`first_token_generated` 字段，记录位置也不正确。修复后用真正逐步返回的 token ID
定义首 token，不依赖文本非空。`submitted_s` 在调用 `add_request` 之前记录，
并同时作为 `prefill_start_s`：这时到首 token 的同一窗口既是 TTFT，也是 prefill
proxy；首 token 到完成为 decode。`dispatch_completed_s` 记录 `add_request`
返回，仅作辅助诊断。后台引擎可能在提交调用返回之前已经执行，因此不能把返回后
的时间称为 GPU 执行起点，也不能据此排除提交阶段的能耗。该窗口包含提交、排队、
调度、主机处理和首 token 采样开销，不能声称是纯 GPU prefill kernel 时间。

阶段模式限制 batch size 1，完成当前请求后才提交下一请求，避免不同请求的
prefill/decode 重叠。构造引擎时请求关闭 prefix caching 与 chunked prefill，并将
`max_num_seqs` 设为 1；GH200 验证使用的 vLLM V1 实际将 chunked prefill 改为
开启，prefix caching 关闭、`max_num_seqs=1` 保持生效。因此 `engine_config.json`
分别记录 `requested`、`effective` 和 `engine_class`，不能只凭传入参数宣称
chunked prefill 已禁用。串行请求的提交到首 token 窗口仍能包含分块 prefill，且
没有不同请求之间的阶段混叠；这不证明引擎内部不存在抢占或重计算。原有批处理
吞吐测试仍可以使用，但不输出未经支持的阶段归因。

## 传感器分辨率仍是实验限制

时间边界修对以后，IPMI 每秒一次仍然可能看不到很短的 prefill。本次对每种传感器
独立检查阶段内的有效样本数，少于两个就不给阶段能耗。这个检查只说明存在读数，
不能证明底层传感器真的解析了阶段内的变化。

尤其需要说明：NVML 的查询间隔不是所有 GPU 的物理功率分辨率。NVIDIA 文档说明，
Ampere（GA100 除外）及之后架构的 `nvmlDeviceGetPowerUsage` 返回 1 秒平均功率；
GA100 和更早架构返回瞬时值。因此即使每 100 ms 读取一次，短阶段也可能被传感器
平均窗口混合。RAPL 的两次能量差同样对应一个区间平均值。
[NVIDIA 官方说明](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)

当前 GPU 监控器采用保守策略：未验证具体架构有效分辨率之前，所有不足 1 秒的
GPU 能耗窗口都返回 null，并在能力报告中说明该策略；即使窗口内有多个样本也
不例外。这不是声称 GA100 硬件也一定使用 1 秒平均。阶段时间戳仍照常保留。

建议新结果同时保留阶段时长、样本数、传感器类型与采样配置，展示哪些阶段估计
不可用。重复实验能减少随机波动，但不能恢复硬件未观测到的时间变化。长 prompt
可以让 prefill 更容易被 IPMI 覆盖，同时必须说明它改变了实验负载。

## 验证状态

本地模拟测试覆盖引擎事件、进程身份、传感器权限、能量计数器回绕、时间窗积分和
缺失值等逻辑。2026-09-14 UTC 已在 GH200 完成真实非 root 的 vLLM 阶段推理，
并通过权限矩阵：非 root 的 `auto` 正常返回 GPU 能力、`full_node` 因 IPMI 权限
不足明确失败；root 的 `full_node` 可读 IPMI，同时 CPU/DRAM RAPL 保持缺失。
短 prefill 的时间戳可用，GPU 阶段能耗按分辨率规则返回 null。

root 的长输入阶段运行及普通 batch 1、2 回归已完成，原始轨迹的独立积分复算已通过。
长输入 prefill 约 1.03 秒，IPMI 窗内样本仍不足，且该次整段 IPMI 均为 515 W；
因此这些功能检查不能证明物理上的整机阶段功率差异。实测环境、结果及传感器限制见
[GH200 验证报告](gh200-validation.md)。Intel RAPL 的实机行为仍未验证，因为
Grace 不提供该接口；首 token 主机观测与精确 GPU kernel profiler 边界的等价性
也未验证。已通过的功能检查不能消除传感器平均窗口带来的阶段分辨率限制。

可直接执行的单机验收步骤和完整测量定义见
[measurement.md](measurement.md)。当前修改不等于已经重现论文图表，也不能从
仓库基线的这些问题反推作者当时实际采用的脚本、记录或后处理流程。
