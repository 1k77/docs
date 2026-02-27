核心指南
========

.. contents::
   :local:
   :depth: 2

----------------------------------
在昇腾设备上对齐 verl 和 vLLM 两个框架下的推理结果
----------------------------------


在昇腾设备上对齐verl和vLLM两个框架下的推理结果。

Last updated: 11/17/2025.

这是一份在昇腾设备上对齐verl和vLLM两个框架下推理结果的教程。

环境变量配置
~~~~~~~~~~~~

在多卡通信情况下：

- HCCL通信下(默认场景):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export HCCL_DETERMINISTIC="true"
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

- LCCL通信下(通过export HCCL_OP_EXPANSION_MODE="AIV"使能）:

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export LCCL_DETERMINISTIC=1
  -  export ATB_LLM_LCOC_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

在单卡无通信情况下：

- HCCL和LCCL通信下:

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

vLLM初始化参数
~~~~~~~~~~~~

需要对 SamplingParams 参数里单独设置seed, 保持vLLM和verl推理结果一致, 举例修改如下：

.. code:: yaml

   sampling_params = SamplingParams(n=1,
                                    logprobs=0,  # can be set to 0 and let actor to recompute
                                    max_tokens=config.response_length,
                                    repetition_penalty=config.get("repetition_penalty", 1.0),
                                    seed=1234)


----------------------------------
profiling分析
----------------------------------

Last updated: 02/24/2026.

背景介绍
~~~~~~~~

随着DeepSeek-R1的发布，大模型强化学习（RL）训练受到广泛关注。在昇腾NPU环境下，verl框架已积累了丰富的性能调优经验。本文系统总结了包括性能数据采集与分析在内的方法论，旨在帮助开发者更高效地运用MindStudio工具链，实现强化学习场景下的性能优化。

强化学习计算流程概述
~~~~~~~~~~~~~~~~~~~~

1. **Rollout**：策略（actor）模型基于输入的prompt序列，推理生成回答（response序列）
2. **ref logprob**：基于prompt和生成的response，reference模型计算ref logprob用于KL散度计算
3. **logprob**：基于prompt和生成的response，actor模型计算logprob用于重要性采样
4. **reward**：基于prompt和生成的response，奖励模型评估奖励值R_N。
5. **update**：基于计算得到的R_N、ref logprob、logprob计算优化函数和策略梯度，对actor模型进行更新

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/rl_data_stream.png
   :alt: rl_data_stream

profilling工具使能
~~~~~~~~~~~~~~~~~~

使能方法
^^^^^^^^

使能和配置教程可参考：
[verl/docs/ascend_tutorial/ascend_profiling_zh.rst at main · verl-project/verl](https://github.com/verl-project/verl/raw/main/docs/ascend_tutorial/ascend_profiling_zh.rst)

性能分析方法论
~~~~~~~~~~~~~~

整体性能概览分析
^^^^^^^^^^^^^^^^

1. 长耗时任务与资源空泡分析
""""""""""""""""""""""""""""

- **操作**：使用MindStudio Insight加载profiling数据，自动识别不同计算阶段，通过RL页签流水图定位长耗时任务与NPU资源空泡
- **价值**：快速掌握不同阶段耗时占比
- **效果展示**：

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Bubble_analysis.png
   :alt: Bubble_analysis

2. 负载均衡分析
""""""""""""""""

- **操作**：通过MindStudio Insight直接查看MSTX打点数据，观察Rollout阶段不同DP Rank的负载均衡情况
- **价值**：快速识别负载不均问题
- **效果展示：**

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Load_Balancing_Analysis.gif
   :alt: Load_Balancing_Analysis

3. 集群整体性能分析
""""""""""""""""""""

- **操作**：结合MSTT的rl_analysis功能，生成集群Timeline缩略图，观察各阶段整体耗时
- **价值**：宏观掌握集群性能瓶颈
- **操作指南**：[rl_analysis使用文档](https://gitcode.com/Ascend/mstt/raw/pre-research/profiler/msprof_analyze/docs/features/rl_analysis.md)
- **效果展示**：

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Cluster%20Performance%20Analysis.png
   :alt: Cluster Performance Analysis

细粒度分析
^^^^^^^^^^

性能分析
""""""""

- **操作**：可通过 MindStudio Insight Windows 或 Linux 版本加载 Profiling 数据
- **价值**：MindStudio Insight 支持分析任务调度效率、算子执行性能、计算资源利用率、集合通信性能等。其 Timeline 视图具备任务拆解与 Overlap 分析功能（**为 MindStudio 独有核心特性，在 NV 及其他竞品中不具备，是 AI 调优的必备工具**），并支持鼠标交互式分析。
- **效果展示**：

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/performance%20analysis.png
   :alt: performance analysis

内存分析
""""""""

通过 Profiling 结合调用栈分析系统内存变化
''''''''''''''''''''''''''''''''''''''''''

- **操作**：采集数据时开启调用栈和内存视图功能。
- **价值**：观察框架、CANN内存申请释放情况，可结合调用栈跟踪到前端python代码。
- **效果展示**：结合调用栈进行内存变化分析。效果如下所示：

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/in-memory%20analytics.gif
   :alt: in-memory analytics

使用 msleaks 工具进行深层次内存分析
''''''''''''''''''''''''''''''''''''

- **操作步骤**：参考 [msleaks 工具使用指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/msleaks/atlas_msleaks_0001.html)。
- **价值**：可以查看框架内存申请总量折线图/内存块图，并直接对应调用栈，可深层次分析框架内存使用情况。
- **效果展示**：

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/msleaks.gif
   :alt: msleaks

性能分析案例
~~~~~~~~~~~~

要做具体的性能分析，profiling要开启**level1**，否则算子的关键信息会缺失。

1.host bound诊断
^^^^^^^^^^^^^^^^

host bound是指CPU任务量综合大于NPU，导致NPU执行出现空泡的现象。可以通过看Host2Device的同步连线来判断，如果连线都是歪的，那证明这里的set信号早于wait信号，NPU一ready就执行了，那也是device bound：

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_1.png
   :alt: host_bound_1

如果确诊为host bound，那么我们可以打开CPU侧，找出各算子的下发耗时。注意找的时候需要找出所有CPU耗时的累加值，而不能找单层，因为首次调用的耗时是很长的。例如下图的GmmSwigluQuant，CPU上首次调用需要1ms，后续每次只需要200us。

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_2.png
   :alt: host_bound_2

此时有的算子在负重前行，有的算子拖了后腿，后者多于了前者。我们优先**找出来host耗时大于device的top算子，这些算子是拖后腿的**，可以交予算子团队重点分析。

2.组网合理性分析
^^^^^^^^^^^^^^^^

有的时候，模型组网没有按照最高效的方式来，这一点在profiling中是非常易于识别的，下面会介绍一下分析思路并给出例子。

通常来讲，LLM中的大的热点算子是Attention和FFN中的矩阵乘计算，二者加起来在prefill下可能达到计算耗时的70%+，decode下可能达到50%+。如果整体的耗时比例不符合预期，或者profiling中出现了一些新面孔，或者拼接类算子太多了，这都值得我们去分析一下模型组网，是不是使用算子的方式错了？尤其是拼接类算子，是值得我们逐一分析的。

对于slice/split/concat这样的拼接类算子，还有transpose/cast这种转换算子，他们的存在往往是前后算子不直接配套造成的。如果前一个算子可以直接对输出做好尾处理，往往可以节省一个算子的启动开销和一次冗余读写。但这样的改变不一定符合算子的基本设计原则。

举一个正例，对于某次Matmul的输出shape为[m, n0 + n1]，在这后面我们接了两个slice，输入均为这个[m, n0 + n1]的tensor，输出分别为[m, n0]和[m, n1]。第一个优化的思路是将两个slice改为一个split，这样耗时可以基本减半，[m, n0 + n1]的显存也可以尽早释放。进一步优化的思路是将矩阵乘的权重从[k, n0 + n1]分割为[k, n0]和[k, n1]，将原来的矩阵乘任务分成两个（前提是这两个的耗时加起来不比之前的劣化太多，分核策略不能出问题），从而彻底消除这个slice/split操作。

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_1.png
   :alt: network_1

举一个反例，Rmsnorm(fp16)+Cast(fp16->fp32)+Matmul(fp32)，Rmsnorm虽然输入输出都是fp16，但考虑到累加运算的精度，内部是fp32做计算的。如果将Cast融到Rmsnorm内，本就内部使用fp32做计算的Rmsnorm就可以省去一个末尾fp32->fp16的cast，加上我们干掉的Cast，总共节省两个cast的同时避免了一次精度丢失。虽然这样看起来精度性能双收了，但fp16进，fp32出的Rmsnorm是反原则的（核心输入和输出需要是同数据类型），除非我们能在广大开源模型中频繁找到这样的结构，证明它的普适性，否则算子团队是不允许做这样的算子的。

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_2.png
   :alt: network_2

3.算子性能初诊
^^^^^^^^^^^^^^

需要利用 ``".\ASCEND_PROFILER_OUTPUT\operator_details.csv"`` 来做分析，从而判断算子识否有性能问题。

Profiling工具会统计这些流水线在不同核上的平均繁忙时间（xxx_time），与最慢核的完整kernel耗时（task_duration）做除法，得到流水线利用率（xxx_ratio）。这些流水线之间虽然互有依赖，且搬运类流水线会互抢带宽，但算子只要设计得当，是可以做到互相掩盖的。因此我们可以初步认为，**当算子的执行耗时大到一定程度上，算子应当在某一条流水线上形成bound**，即利用率要高到一定程度。经验上，在单算子耗时达到50μ时，就可以认为算子应当在bound流水线上，达成80%+的占用率了。

以下图为例，第一行是一个FA算子，第二行是一个Matmul算子，FA在vec流水线上达到了88.1%的利用率，Matmul算子在mac流水线上达到了89.8%的利用率，他们的性能可以认为是合格的。

.. image:: https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Operator%20performance.png
   :alt: Operator performance

4.亲和shape调整
^^^^^^^^^^^^^^^

对于一个模型而言，超参是我们控制不了的，但我们可以控制并发度、权重格式、切分策略等因素来迎合算子，使其发挥出最大的性能，这一节主要从算子搬运效率和负载均衡两个方面出发，讨论模型侧值得尝试的调整方向。

4.1 搬运效率亲和的shape
""""""""""""""""""""""""

mte2是一个自身效率严重受shape影响的流水线。要想让mte2保证最大搬运效率，我们需要保障如下两个条件至少满足其一：

**（1）被搬运的矩阵使用nz作为format（最优）
（2）被搬运的矩阵的尾轴512B对齐，且不为16KB的整数倍（近似最优）**

对于权重矩阵来说，推理阶段尤其是decode，我们通常满足（1），训练阶段我们通常满足（2）。**如果我们做不到（1），我们就要迎合（2）**。典型的手段有：

1，如果没达成B的矩阵的首轴是亲和的而尾轴不亲和，那么对它做transpose
2，调整TP切分策略，避免出现不亲和的尾轴

4.2 负载均衡亲和的shape
""""""""""""""""""""""""

在算子shape不大时，受制于算子语义，我们有可能不能把所有核都利用起来，或者即使开满核，负载均衡却很差。这一小节主要是对decode阶段的小shape做分析。

首先，我们明确出当前NPU卡是多少核的，如果不清楚，跑出来的profiling里都是20，40这样的数，就说明是20核，反之是24核。这里我的24核其实是代表了一个cube和两个vector组成的小组，我们可以认为是一个cube作为主核，带了两个vector作为从核。如果一个算子是纯vector算子，那么就不再有组的概念，40或48个vector核会作为主核直接独立去拿逻辑任务。

对于LLM中的vector算子，它的一种常见分核策略有可能是分在最高维，也就是batch维，常见于对低维（也叫尾轴）有规约操作的norm类、动态量化类等算子；另一种是整体拍平，允许算子切分的非常细的算子，如elementwse算子。对于第一种，我们就可以在模型侧关注它的负载均衡问题。例如我们打48batch，而硬件却是个40个vector核，那这40个核会循环2次，第二次有多数的核会无事可做，这个batch数就可以认为是不友好的。如果将batch打到64或80，性能可以预见会是无损的。同样的情况下，如果是48核的卡，那我们可以认为这就是个非常友好的batch数。

对于cube类算子，它常见的分核策略是以base快去切分M和N（K轴是累加轴，对它分核会引入确定性问题）。最常见的分块是baseM=128，baseN=256。在decode阶段，我们的耗时基本可以看做都是在搬权重，这是因为激活的M极小，M方向大概率只分了一块，那么右矩阵就只需要搬一次。所以我们在M≤128的范围内可以尽情提高M，对性能都基本是无损的，如果M大于128，可以认为(128, 256]是下一个性能分档。

除了M外，N轴切分的任务也影响算子亲和性，以deepseekR1中的MLA预处理为例，它会使用同一个激活（shape为[batch_size, 7168]）与两个权重做矩阵乘(shape为[7168, 1536]和[7168, 576])。在batch_size打不大的情况下，即使baseN缩短为128，N轴都不能用满核数，所以此时这两个矩阵乘各自的耗时，会约等于将他们权重N轴拼起来乘(shape为[7168, 2112])的矩阵乘的耗时。如果仅考虑模型竞争力，我们更希望对这两个权重做合并，否则两个小的矩阵乘带宽利用率都会非常差。

对于Attention算子，它常见的分核策略是q_seqlen、batch_size和kv_headnum。增量阶段q_seqlen会以MTP和GQA倍数做合并，但是通常也不会大过128，划分不出第二个任务，那么并行度基本就是batch_size * kv_headnum。

总的来说，我们可以依据shape信息和算子类别，对算子是否有负载均衡问题作出识别，从而对我们切分策略选择，最高吞吐量的batch策略作出预判。


----------------------------------
在昇腾设备上基于 FSDP 或 MindSpeed (Megatron) 后端进行性能数据采集
----------------------------------

Performance data collection based on FSDP or MindSpeed(Megatron) on Ascend devices(zh)
在昇腾设备上基于 FSDP 或 MindSpeed (Megatron) 后端进行性能数据采集
Last updated: 12/20/2025.

这是一份在昇腾设备上基于FSDP或MindSpeed(Megatron)后端，使用GRPO或DAPO算法进行数据采集的教程。

配置
~~~~

使用两级profile设置来控制数据采集

全局采集控制：使用verl/trainer/config/ppo_trainer.yaml(FSDP)，或verl/trainer/config/ppo_megatron_trainer.yaml(MindSpeed)中的配置项控制采集的模式和步数。

角色profile控制：通过每个角色中的配置项控制等参数。

全局采集控制
~~~~~~~~~~~~

通过 ppo_trainer.yaml 中的参数控制采集步数和模式：

global_profiler: 控制采集的rank和模式

tool: 使用的采集工具，选项有 nsys、npu、torch、torch_memory。

steps: 此参数可以设置为包含采集步数的列表，例如 [2, 4]，表示将采集第2步和第4步。如果设置为 null，则不进行采集。

save_path: 保存采集数据的路径。默认值为 “outputs/profile”。

角色profiler控制
~~~~~~~~~~~~~~~~

在每个角色的 profiler 字段中，您可以控制该角色的采集模式。

enable: 是否为此角色启用性能分析。

all_ranks: 是否从所有rank收集数据。

ranks: 要收集数据的rank列表。如果为空，则不收集数据。

tool_config: 此角色使用的性能分析工具的配置。

通过每个角色的 profiler.tool_config.npu 中的参数控制具体采集行为：

level: 采集级别—选项有 level_none、level0、level1 和 level2

level_none: 禁用所有基于级别的数据采集（关闭 profiler_level）。

level0: 采集高级应用数据、底层NPU数据和NPU上的算子执行详情。在权衡数据量和分析能力后，level0是推荐的默认配置。

level1: 在level0基础上增加CANN层AscendCL数据和NPU上的AI Core性能指标。

level2: 在level1基础上增加CANN层Runtime数据和AI CPU指标。

contents: 控制采集内容的选项列表，例如 npu、cpu、memory、shapes、module、stack。

npu: 是否采集设备端性能数据。

cpu: 是否采集主机端性能数据。

memory: 是否启用内存分析。

shapes: 是否记录张量形状。

module: 是否记录框架层Python调用栈信息。相较于stack，更推荐使用module记录调用栈信息，因其产生的性能膨胀更低。

stack: 是否记录算子调用栈信息。

analysis: 启用自动数据解析。

discrete: 使用离散模式。

示例
~~~~

禁用采集
^^^^^^^^

.. code:: yaml

   global_profiler:
      steps: null # disable profile

端到端采集
^^^^^^^^^^

.. code:: yaml

   global_profiler:
      steps: [1, 2, 5]
      save_path: ./outputs/profile
   actor_rollout_ref:
      actor:  # 设置 actor role 的 profiler 采集配置参数
         profiler:
            enable: True
            all_ranks: True
            tool_config:
               npu:
                  discrete: False
                  contents: [npu, cpu]  # 控制采集列表，默认cpu、npu，可配置memory、shapes、module等

   # rollout & ref follow actor settings

离散模式采集
^^^^^^^^^^^^

.. code:: yaml

   global_profiler:
      steps: [1, 2, 5]
      save_path: ./outputs/profile
   actor_rollout_ref:
      actor:
         profiler:
            enable: True  # 设置为 True 以采集训练阶段
            all_ranks: False
            ranks: [0]  # 全局 Rank 0
            tool_config:
               npu:
                  discrete: True
                  contents: [npu, cpu]
      rollout:
         profiler:
            enable: True  # 设置为 True 以采集推理阶段
            all_ranks: False
            ranks: [0]  # 在 Agent Loop 模式下，此处指推理实例的 Replica Rank (例如第 0 个实例)
            tool_config:
               npu:
                  discrete: True  # Agent Loop 模式下必须开启离散模式
   # ref follow actor settings

Agent Loop 模式说明
~~~~~~~~~~~~~~~~~~

在 Agent Loop 模式下，Rollout 阶段的性能数据 必须使用离散模式 采集，此时 Profiler 由推理引擎后端触发。

Rank 定义：Rollout 配置中的 ranks 指代 Replica Rank（推理实例索引），而非全局 Rank。

推理引擎支持：当前支持vLLM和SGLang引擎，无需额外设置。具体说明如下：

vLLM 引擎：自动采集 AsyncLLM 调度栈及推理进程性能数据。不支持设置 analysis（默认不解析，需离线解析）和 profiler_level（默认 level1）。

SGLang 引擎：自动采集推理进程性能数据。不支持 contents 中的 memory 配置项。不支持设置 analysis（默认解析）和 profiler_level（默认 level0）。

可视化
~~~~~~

采集后的数据存放在用户设置的save_path下，可通过 MindStudio Insight 工具进行可视化。

另外在Linux环境下，MindStudio Insight工具提供了 JupyterLab插件 形态，提供更直观和交互式强的操作界面。JupyterLab插件优势如下：

无缝集成：支持在Jupyter环境中直接运行MindStudio Insight工具，无需切换平台，无需拷贝服务器上的数据，实现数据即采即用。

快速启动：通过JupyterLab的命令行或图形界面，可快速启动MindStudio Insight工具。

运行流畅：在Linux环境下，通过JupyterLab环境启动MindStudio Insight，相较于整包通信，有效解决了运行卡顿问题，操作体验显著提升。

远程访问：支持远程启动MindStudio Insight，可通过本地浏览器远程连接服务直接进行可视化分析，缓解了大模型训练或推理数据上传和下载的困难。

如果analysis参数设置为False，采集之后需要进行离线解析：

.. code:: python

   import torch_npu
   # profiler_path请设置为"localhost.localdomain_<PID>_<timestamp>_ascend_pt"目录的上一级目录
   torch_npu.profiler.profiler.analyse(profiler_path=profiler_path)

进阶指南：精细化采集
~~~~~~~~~~~~~~~~~~~~

背景与挑战
^^^^^^^^^^

上述基于配置文件的采集方式虽然便捷，但在 长序列 (Long Context) 或 大全局批量 (Large Global Batch Size) 的训练场景中面临挑战。 在一个完整的训练步 (Step) 内，模型计算呈现出高频次、重复性的特征：

Rollout 阶段：序列生成 (Generate Sequence) 是一个自回归过程，涉及成千上万次 Decoder 模型的前向计算。

Training 阶段：为了控制显存峰值，verl 通常采用 Micro-Batch 策略，将庞大的数据流切分为多个微批次进行计算。

compute_log_prob (Actor/Ref)：涉及多轮纯前向传播。

update_policy (Actor/Critic)：涉及多轮前向与反向传播。

这种特性会导致全量 Profiling 产生海量且重复的算子记录。如下图所示：

https://raw.githubusercontent.com/mengchengTang/verl-data/master/verl_ascend_profiler.png

即使使用了 discrete 模式，单个阶段的性能数据文件仍可能达到数 TB，导致 解析失败 或 可视化工具卡顿 。

解决方案：关键路径采样
^^^^^^^^^^^^^^^^^^^^^^

为了解决上述问题，我们可以采用 关键路径采样 策略：基于 torch_npu.profiler 提供的API接口，直接修改 Python 源码，仅采集具有代表性的数据片段（如特定 Decode Step 或首个 Micro-Batch）。

重要提示
^^^^^^^^

本章节涉及直接修改源码。建议修改前备份文件，调试完成后恢复。

使用代码插桩采集时，请务必在 ppo_trainer.yaml 或 ppo_megatron_trainer.yaml 中**禁用全局采集** (global_profiler: steps: null)，以避免 Profiler 冲突。

1. Rollout 阶段精细化采集
^^^^^^^^^^^^^^^^^^^^^^^^^

对于 vLLM 或 SGLang 推理引擎，我们可以通过控制 schedule 参数来控制采集模型在特定token的前向传播性能数据。

vLLM 引擎
""""""""""

参考版本：vLLM v0.11.0, vLLM-Ascend v0.11.0rc1

修改文件：vllm-ascend/vllm_ascend/worker/worker_v1.py

.. code:: python

   class NPUWorker(WorkerBase):

       def __init__(self, *args, **kwargs):
           # ... existing code ...

           # Initialize profiler
           import torch_npu
           experimental_config = torch_npu.profiler._ExperimentalConfig(
               profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
               export_type=torch_npu.profiler.ExportType.Db,  # 可选择torch_npu.profiler.ExportType.Text格式
           )
           self.profiler_npu = torch_npu.profiler.profile(
               activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
               with_modules=False,  # 采集调用栈
               profile_memory=False,  # 采集内存
               experimental_config=experimental_config,
               # 跳过第一步，warmup一步，采集3步，重复1次。如果想采集第30~70个decode step，可以设置为schedule=torch_npu.profiler.schedule(wait=29, warmup=1, active=30, repeat=1)
               schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./outputs/vllm_profile", analyse_flag=True)  # 采集数据保存路径，是否在线解析
           )
           self.profiler_npu.start()

           # ... existing code ...

       def execute_model(self, scheduler_output=None, intermediate_tensors=None, **kwargs):
           # ... existing code ...
           output = self.model_runner.execute_model(scheduler_output,
                                               intermediate_tensors)

           self.profiler_npu.step()  # 驱动 schedule，对部分decode step进行采集

           # ... existing code ...

SGLang 引擎
""""""""""""

参考版本：SGLang master 分支

修改文件：sglang/python/sglang/srt/model_executor/model_runner.py

.. code:: python

   # ... existing imports ...
   import torch_npu

   class ModelRunner:

       def __init__(self, *args, **kwargs):
           # ... existing init code ...

           # Initialize profiler (配置同上，略)
           experimental_config = torch_npu.profiler._ExperimentalConfig(...)
           self.profiler_npu = torch_npu.profiler.profile(
               # ...
               # 跳过第一步，warmup一步，采集3步，重复1次。
               schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./outputs/sglang_profile", analyse_flag=True)
           )
           self.profiler_npu.start()

       def forward(self, forward_batch, **kwargs):
           # ... existing code ...

           self.profiler_npu.step()  # 驱动 schedule，对部分decode step进行采集
           return output

2. compute_log_prob (Actor & Ref) 阶段精细化采集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该阶段计算新旧策略的概率分布。

FSDP 后端
""""""""""

FSDP 后端允许在 Micro-Batch 级别进行精细控制。

修改文件：verl/workers/actor/dp_actor.py

.. code:: python

   # ... 引入依赖 ...
   import torch_npu

   class DataParallelPPOActor(BasePPOActor):

       def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:

           role = "Ref" if self.actor_optimizer is None else "Actor"
           # 准备 profiler (配置同上，略)
           experimental_config = torch_npu.profiler._ExperimentalConfig(...)
           self.prof_npu = torch_npu.profiler.profile(
               # ...
               # wait=0, warmup=0, active=1: 直接采集第一个 micro-batch
               schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"./outputs/{role}_compute_log_prob", analyse_flag=True)
           )


           # 此函数ref和actor共用，设置role标志位来区分。如果想采集actor_compute_log_prob，可设置if role=="Actor":
           if role=="Ref":
               self.prof_npu.start()

           for micro_batch in micro_batches:

               # ... 原始计算逻辑 ...
               with torch.no_grad():
                   entropy, log_probs = self._forward_micro_batch(...)

                   # 驱动 schedule，对micro batch进行采集
                   if role=="Ref":
                       self.prof_npu.step()

               # ...

Megatron 后端
""""""""""""""

Megatron 后端的 Micro-Batch 调度由框架内部管理，暂不支持通过简单的代码插桩进行 Micro-Batch 级别的精细化采集。建议使用全局配置进行采集。

3. update_policy (Actor & Critic) 阶段精细化采集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Update 阶段包含前向和反向传播。

FSDP 后端
""""""""""

FSDP 后端支持设置对 Mini-Batch 和 Micro-Batch 的粒度进行采集。

修改文件：verl/workers/actor/dp_actor.py

.. code:: python

   # ... 引入依赖 ...
   import torch_npu

   class DataParallelPPOActor(BasePPOActor):

       def update_policy(self, data: DataProto):

           # 准备 profiler (配置同上，略)
           experimental_config = torch_npu.profiler._ExperimentalConfig(...)
           self.prof_npu = torch_npu.profiler.profile(
               # ...
               # 仅采集第一个 Mini Batch（包含所有 Micro-Batch 的计算和一次优化器更新）
               schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./outputs/fsdp_actor_update_profile", analyse_flag=True)
           )
           self.prof_npu.start()

           # ... PPO Epochs 循环 ...
           for _ in range(self.config.ppo_epochs):
               # ... Mini Batch 循环 ...
               for batch_idx, mini_batch in enumerate(mini_batches):
                   # ... mini_batches 切分 ...

                   for i, micro_batch in enumerate(micro_batches):
                       # ... 原始 Forward & Backward 逻辑 ...
                       # ... loss.backward() ...
                       pass

                   grad_norm = self._optimizer_step()

                   # 驱动 schedule，对mini batch进行采集，如果想对micro batch进行，则将self.prof_npu.step()移动到micro_batch的循环内
                   self.prof_npu.step()

Megatron 后端
""""""""""""""

Megatron 后端支持以 Mini-Batch 的粒度进行采集。

修改文件：verl/workers/actor/megatron_actor.py

.. code:: python

   class MegatronPPOActor(BasePPOActor):

       def update_policy(self, dataloader: Iterable[DataProto]) -> dict:
           # ...
           # 准备 profiler (配置同上，略)
           experimental_config = torch_npu.profiler._ExperimentalConfig(...)
           self.prof_npu = torch_npu.profiler.profile(
               # ...
               # 仅采集第一个 Mini Batch 的计算（含所有 Micro-Batch）和一次优化器更新
               schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./outputs/megatron_actor_update_profile", analyse_flag=True)
           )
           self.prof_npu.start()

           for data in dataloader:
               # ... 内部会调用 self.forward_backward_batch 进行计算 ...
               # ... metric_micro_batch = self.forward_backward_batch(...)

               # ... self.actor_optimizer.step() ...

               # 驱动 schedule，对mini batch进行采集
               self.prof_npu.step()
