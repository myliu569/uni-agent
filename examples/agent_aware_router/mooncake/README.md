# Mooncake Store 接入与远端验收

本轮只接入外部 CPU 缓存；Router 仍使用 GPU KV events，忽略明确为 CPU 的
Stored/Removed。GPU prefix caching 保持开启。存储查询、保存、恢复由
`MooncakeStoreConnector` 完成，不增加 Router 的 L2 目录或评分。

本机完成代码及 CPU 测试，真实 GPU 验收在目标机器执行。以下是部署模板，
不是已经验证的版本组合或性能结果。首轮使用 Full Attention、固定权重、
同模型/TP/缓存布局的副本，不包含 hybrid attention、Ascend、SSD 或异构 TP。

## 1. 固定目标环境

使用与当前 verl 兼容的独立环境或容器；其中的 vLLM 必须支持
`MooncakeStoreConnector`、`mode=standalone-store` 和 `save_decode_cache`。
Mooncake 必须提供 `mooncake_master`、`mooncake_client` 及 Python bindings。
这些依赖只在启用时需要，不添加到 UniAgent 的必装依赖。

在目标环境安装完成后，保存以下输出，验收期间不要升级：

```bash
git rev-parse HEAD
git submodule status verl
python -m pip freeze > mooncake-acceptance-requirements.txt
python -m pip show vllm mooncake-transfer-engine ray torch
nvidia-smi
mooncake_master --help > mooncake-master-help.txt
mooncake_client --help > mooncake-client-help.txt
```

对照**所安装版本**的帮助和源码核对下面的参数；旧版本不支持时先解决兼容性，
不能删掉 `mode` 后以 embedded 冒充 standalone-store。版本锁定和 GPU 验收均待目标机器完成。

接口参考：[vLLM Connector 配置](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)、
[Mooncake 服务部署](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html)。
链接指向滚动文档，不能代替已安装版本的核对。

## 2. 启动独立 CPU 池

启动顺序：master → 贡献内存的 `mooncake_client` → vLLM/UniAgent。
所有进程在目标单机上；示例地址假定它们共享网络命名空间。容器隔离网络时应使用
相互可达的地址，同时调整服务参数和 JSON，不能直接照搬 loopback。

在独立终端以前台方式运行 master（以下 50051/50053 端口须空闲）：

```bash
mooncake_master --port=50051 --enable_offload=false
```

在另一个终端**显式设置**本次池容量 `STORE_POOL_BYTES`（字节数，正整数），然后运行 owner：

```bash
: "${STORE_POOL_BYTES:?Set the CPU pool budget in bytes before starting the owner}"
mooncake_client \
  --host=127.0.0.1 --port=50053 \
  --master_server_address=127.0.0.1:50051 \
  --metadata_server=P2PHANDSHAKE --protocol=tcp \
  --global_segment_size="$STORE_POOL_BYTES" \
  --enable_offload=false
```

先确认 master 正常监听、owner 成功注册且容量符合预算，再启动推理。
可用 `ss -ltnp` 检查监听；监听端口本身不证明注册或 KV 读写成功。
保留两个进程的启动日志。停止时先退出本次推理作业，再在各服务终端按 Ctrl-C；
不使用全局 `pkill`，也不停止其他实验共享的服务。

复制 [配置模板](standalone-cpu.example.json) 为本次实验配置，并记录绝对路径。
`global_segment_size=0` 表示 vLLM ranks 不贡献池容量；CPU 池由 owner 分配。
模板 `local_buffer_size=1073741824` 是**每个 rank** 的 1 GiB 私有缓冲示例，
须按目标环境显式调整。总预算还要计入 owner 缓冲、各 rank 私有缓冲和其他进程开销。
TCP 用于首轮功能验收；将来更换 RDMA 时应重新验证，性能结果需记录传输协议。

## 3. 开启 UniAgent

在同一运行环境、仓库根目录执行。模型、数据、GPU 数、TP、上下文长度和并发等
继续使用已经能够跑通的 `run_infer.sh` 参数，增加：

```text
bash examples/agent_aware_router/run_infer.sh \
  <已有的模型、数据、task-config 和资源参数> \
  --device gpu --kv-events \
  --enable-mooncake \
  --mooncake-config-path /absolute/path/mooncake.json \
  --no-mooncake-save-decode-cache \
  --result-path /absolute/path/acceptance-result.json
```

在启动 Python **之前**，为共享 Store 的各实例设置同一个种子：

```bash
export PYTHONHASHSEED=42
```

decode 保存默认关闭；第二轮功能检查改为 `--mooncake-save-decode-cache`。
不传 `--enable-mooncake` 就走原有路径，不读取配置文件、不附加 Connector。
启动日志和结果 JSON 的 `mooncake` 字段记录提交给引擎的参数；它们不是加载成功的证据。

入口会把配置路径解析为绝对路径并校验 JSON 对象，通过 Ray job 的 `runtime_env.env_vars`
传递路径。配置文件必须在实际引擎进程中以相同路径可读，路径传递不会上传文件。
如果设置 `RAY_ADDRESS`，入口连接已有集群，否则初始化 Ray。
同一 Python 进程已初始化 Ray 但缺少对应 job 环境时会报错，应重新启动作业。

除配置路径外，入口还转发调用者显式设置的 `MOONCAKE_PREFERRED_SEGMENT`、
`MOONCAKE_REQUESTER_LOCAL_HOSTNAME`、`VLLM_MOONCAKE_STORE_TIER_LOG` 和 `PYTHONHASHSEED`。
不批量转发任意环境变量；额外传输参数需在目标环境的 worker 启动配置中设置。

## 4. 验证实际配置与恢复

先用单副本，再用两个相同配置副本。所有进程使用相同权重、tokenizer、
block size、dtype、TP、缓存布局、哈希算法和种子；保存这些实际生效值及 GPU block 数。

| 检查 | 执行方法 | 必须保留的证据 |
| --- | --- | --- |
| 配置传递 | 分别使用新建 Ray 和 `RAY_ADDRESS=auto` 的已有集群；在实际 vLLM worker 核对 `MOONCAKE_CONFIG_PATH` 与文件可读性 | worker 环境/配置路径及 Connector 初始化日志；仅 driver 日志不够 |
| 冷启动写入 | 独立空 Store，发送包含多个完整 KV block 的固定长前缀 P | Connector/Store PUT 完成或成功保存指标 |
| GPU 热复用 | 在同一副本再次请求 P | GPU prefix 命中和正确输出；不要求发生 Store GET |
| GPU 淘汰后恢复 | 注入不同前缀的干扰请求直到 P 从 GPU 淘汰，再请求 P；CPU 池须能保留 P | GPU 淘汰证据、外部匹配与成功加载 block/字节数，以及输出 |
| 跨副本恢复 | A 完成保存后，让从未处理 P 的 B 直接接收 P | B 的成功加载记录；先直接访问副本控制请求分配，排除 Router 的分配影响 |
| decode 保存 | 使用足够长输出形成完整 block；开/关保存各跑一次，再复用对应会话前缀 | 生效开关、额外保存与后续加载证据；考虑末尾不足整块的情况 |
| UniAgent 链路 | 使用同一配置跑多轮 agent 请求和两个副本 | 请求成功、GPU 路由指标正常、外部复用证据，以及轮次切换后的 Store 状态 |
| 正确性对照 | 固定输入与采样设置，比较 Mooncake 关闭和开启的输出 | 无恢复错误；确定性场景输出对照，若有差异先排查再验收 |

worker 环境可通过所用容器/进程管理工具检查；只读取所需配置项，不导出全部环境或凭据。
可先用普通 Ray actor 检查 job 环境继承，但它不能替代实际 vLLM worker 检查。
获取缓存证据时使用所选版本的引擎/Connector/Store 日志或指标，不依赖 Router CPU 计数。
lookup 命中不等于加载完成；没有加载证据时不能通过验收。

用真实缓存压力制造 GPU 淘汰，不调用可能清除 Connector 的通用 reset。
当前 verl 部分生命周期路径使用 `reset_prefix_cache(reset_connector=True)`；
应核对本次 rollout mode 和多轮生命周期，不能全局禁用权重变化后的必要失效。
冷启动检查使用本次专用的 master/owner；不要清空共享实验的 Store。

验收记录包含版本清单、完整命令、配置副本、PUT/GET 成功证据、对照输出、失败信息及
上述各行结论。全部完成后才进入四组性能实验；本轮不以启动成功或单次加速宣称性能收益。
