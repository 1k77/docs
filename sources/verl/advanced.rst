进阶案例
========

.. contents::
   :local:
   :depth: 2

----------------------------------
NPU Qwen3-32B GSPO Optimization Practice
----------------------------------

Last updated: 02/26/2026.

本文章对应脚本地址：`qwen3_32b_gspo_npu <https://github.com/volcengine/verl/blob/main/examples/gspo_trainer/run_qwen3_32b_gspo_npu.sh>`_

算法适配
----------------------------------

GSPO通过将优化颗粒度从**token级**提升到**sequence级**，规避了GRPO会遇到的**方差急剧增大**导致训练不稳定的情况，增加了训练的稳定性，同时该算法也在一定程度上提升了算法的收敛速度。

想要成功在verl仓库中成功调用到GSPO算法，需要进行如下的必要配置

.. code-block:: python

  # 核心算法配置  
  algorithm.adv_estimator=grpo \                    # 使用GRPO优势估计器    
  algorithm.use_kl_in_reward=False \                # 不在奖励中添加KL惩罚    
  # GSPO策略损失模式  
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \ # 启用GSPO策略损失
  # 极小裁剪范围（GSPO特色）  
  actor_rollout_ref.actor.clip_ratio_low=0.0003 \   # 裁剪下界，论文推荐值    
  actor_rollout_ref.actor.clip_ratio_high=0.0004 \  # 裁剪上界，论文推荐值    
  # KL配置（GSPO不使用KL loss）  
  actor_rollout_ref.actor.use_kl_loss=False \       # 禁用KL损失    
  actor_rollout_ref.actor.kl_loss_coef=0.0 \        # KL损失系数设为0    
  # 序列级损失聚合模式（GSPO核心）  
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \ # 序列级平均，GSPO论文推荐    
  # 批次配置  
  actor_rollout_ref.rollout.n=16 \                  # 每个prompt生成16个响应（组采样）

一般选择入口函数为 ``verl.trainer.main_ppo``

基础环境
----------------------------------

当前支持Atlas 800T A3 与 Atlas 900 A3 SuperPoD。完成跑完本次最佳实践需要 4台Atlas 800T A3。关键软件版本可以参考：`Ascend Quickstart <https://github.com/volcengine/verl/blob/main/docs/ascend_tutorial/ascend_quick_start.rst>`_

安装基础环境
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - software
     - version
   * - Python
     - >= 3.10, <3.12
   * - CANN
     - == 8.3.RC1
   * - torch
     - == 2.7.1
   * - torch_npu
     - == 2.7.1
   * - verl
     - main分支 commitId=252d76908b903ad8fb6969eb3a5e5f873c95ea2b
   * - vllm
     - v0.11.0
   * - vllm-ascend
     - v0.11.0-dev
   * - transformers
     - 4.57.3

在本实践中, 我们通过指定 verl 的commit id 以避免引入其他问题

.. code-block:: bash

  cd verl
  git checkout 252d76908b903ad8fb6969eb3a5e5f873c95ea2b
  # 指定相应的recipe版本
  git submodule update --init --recursive recipe

权重获取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

从Hugging Face库下载对应的模型权重：`Qwen/Qwen3-32B · Hugging Face <https://huggingface.co/Qwen/Qwen3-32B>`_

数据集准备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  # 下载math-17k数据集
  git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

  # 下载AIME_2024测试数据集
  git clone https://huggingface.co/datasets/Maxwell-Jia/AIME_2024

jemalloc安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。

Ubuntu 操作系统
""""""""""""""""

通过操作系统源安装jemalloc（注意： 要求ubuntu版本>=20.04）：

.. code-block:: shell

  sudo apt install libjemalloc2

在启动任务前执行如下命令通过环境变量导入jemalloc，需先通过 **find /usr -name libjemalloc.so.2** 确认文件是否存在 ：

.. code-block:: shell

  # arm64架构
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
  # x86_64架构
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

OpenEuler 操作系统
""""""""""""""""""

执行如下命令重操作系统源安装jemalloc

.. code-block:: shell

  yum install jemalloc

如果上述方法无法正常安装，可以通过源码编译安装 前往jemalloc官网下载最新稳定版本，官网地址:https://github.com/jemalloc/jemalloc/releases/

.. code-block:: shell

  tar -xvf jemalloc-{version}.tar.bz2
  cd jemalloc-{version}
  ./configure --prefix=/usr/local
  make
  make install

在启动任务前执行如下命令通过环境变量导入jemalloc：

.. code-block:: shell

  #根据实际安装路径设置环境变量，例如安装路径为:/usr/local/lib/libjemalloc.so.2,可通过以下命令来设置环境变量(可通过 find /usr -name libjemalloc.so.2 确认文件是否存在)
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2

多机任务拉起
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

针对本实践提供的多机任务，可用下面的脚本拉起

.. code-block:: bash

  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray

  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1
  export TASK_QUEUE_ENABLE=1
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600
  export HCCL_ASYNC_ERROR_HANDLING=0
  export CPU_AFFINITY_CONF=1
  export VLLM_USE_V1=1
  export VLLM_ATTENTION_BACKEND=XFORMERS
  export VLLM_ASCEND_ENABLE_FLASHCOMM=1
  export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
  export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
  export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2

  # 修改为当前需要跑的用例路径
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"

  ulimit -n 32768
  mkdir logs

  NNODES=4
  NPUS_PER_NODE=16
  # 修改为对应主节点IP
  MASTER_ADDR="IP FOR MASTER NODE"
  # 修改为当前节点的通信网卡
  SOCKET_IFNAME="Your SOCKET IFNAME"
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  # 获取当前IP
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # 主节点启动
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))

        # 判断device_count 是否与 NNODES 相等
        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
            ray status
            bash $DEFAULT_SH
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done
  else
    # 子节点尝试往主节点注册 ray 直到成功
    while true; do
        # 尝试连接 ray 集群
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

        # 检查连接是否成功
        ray status
        if [ $? -eq 0 ]; then
            echo "Successfully connected to the Ray cluster!"
            break
        else
            echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
            sleep 5
        fi
    done
  fi

  sleep 600

DEFAULT_SH:修改为训练所用配置 sh 文件路径。在此案例中修改为 `Qwen2.5-32B <https://github.com/volcengine/verl/blob/main/examples/gspo_trainer/run_qwen3_32b_gspo_npu.sh>`_ 路径。

NNODES 和 NPUS_PER_NODE:修改为使用节点数量和每个节点 NPU 数量。在此案例中分别为4和16。

MASTER_ADDR:修改为对应主节点 IP。即所有节点的 MASTER_ADDR 应该相同。

SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: 修改为对应通信网卡，通信网卡可以通过以下命令获取：

.. code-block:: bash

  ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1

性能调优
----------------------------------

优化从训练、推理、调度和其他四个方面入手。

训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

动态bsz
""""""""""""""""""""""

.. code-block:: bash

  actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
  infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))

**这个优化点主要调整上面这两个参数，不过需要注意这两个参数调整的太大会导致OOM**

**主要调整** ``actor_ppo_max_token_len`` ,调大了会降低训练的耗时，调整 ``infer_ppo_max_token_len`` 没有明显的收益，可以不动

**这两个参数的作用介绍如下：**

**这两个参数用于控制动态批处理(dynamic batch size)模式下每个GPU处理的最大token数量**

- **``actor_ppo_max_token_len``**: Actor模型在PPO更新(前向+反向传播)时每个GPU能处理的最大token数
- **``infer_ppo_max_token_len``**: 推理阶段(Reference policy和Rollout)计算log概率时每个GPU能处理的最大token数

推理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ACLgraph+FULL_DECODE_ONLY
""""""""""""""""""""""""

推理算子下发方面的优化，平均能有 ``15%~20%`` 左右的性能收益。

先看单开**ACLgraph**，如下：

.. code-block:: bash

  # 开启ACLgraph+FULL_DECODE_ONLY（注意：当设置此参数为False时，TASK_QUEUE_ENABLE必须设置为1，不然会报错）
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[8,16,32,64,128]' \ 
  actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode='FULL_DECODE_ONLY' \

``FULL_DECODE_ONLY`` 开启成功后有如下输出：

.. image:: https://github.com/wucong25/verl-data/blob/main/ascend_acl_graph.png
   :alt: FULL_DECODE_ONLY result

**``cudagraph_capture_sizes``参数设置指南**

cudagraph_capture_sizes设置的值对应的是批大小，这里的批大小不是配置里的DP域对应的那个批次大小，这里是相较于vllm来说的批大小，单位为**token**

默认生成的算法如下，可做参考

.. image:: https://github.com/wucong25/verl-data/blob/main/ascend_set_cudagraph_sizes.png
   :alt: cudagraph_capture_sizes

推理后端切换
""""""""""""""""""""""""

使用方式： ``export VLLM_ATTENTION_BACKEND=XFORMERS``

.. image:: https://github.com/wucong25/verl-data/blob/main/ascend_vllm_attn_backend.png
   :alt: VLLM_ATTENTION_BACKEND

注：需要注意某些后端在一些比较老的vllm-ascend版本内并不支持

使能vllm v1版本
""""""""""""""""""""""""

使用方式： ``export VLLM_USE_V1=1``

可以常开，一般都是正收益。

调度
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AIV
""""""""""""""""""""""""

打开方式：设置 ``export HCCL_OP_EXPANSION_MODE="AIV"``

HCCL_OP_EXPANSION_MODE环境变量用于配置通信算法的编排展开位置，支持如下取值：

- AI_CPU：代表通信算法的编排展开位置在Device侧的AI CPU计算单元。
- AIV：代表通信算法的编排展开位置在Device侧的Vector Core计算单元。
- HOST：代表通信算法的编排展开位置为Host侧CPU，Device侧根据硬件型号自动选择相应的调度器。
- HOST_TS：代表通信算法的编排展开位置为Host侧CPU，Host向Device的Task Scheduler下发任务，Device的Task Scheduler进行任务调度执行。

下面介绍两种展开机制

HOST展开
''''''''''''''''''''''''

.. image:: https://github.com/wucong25/verl-data/blob/main/ascend_task_queue1.png
   :alt: HOST展开
   :scale: 50%

- 软件栈工作在hostcpu，通信算法展开一个个task
- 每个task调用runtime接口，下发到device的rtsqueue
- STARS从rstqueue上顺序拿取task
- 根据task类型分别调用掉SDMA和RDMA引擎。
    **单算子瓶颈**：hostbound 每个task提交是2~5us，一个通信算子有几百个task，单算子场景不会在device上缓存，下发一个执行一个

AICpu机制展开
''''''''''''''''''''''''

.. image:: https://github.com/wucong25/verl-data/blob/main/ascend_task_queue3.png
   :alt: AICpu机制展开
   :scale: 50%

- host侧不下发一个个task，把通信算子作为一个个kernel，放在通信算子kernel的队列上去。
- STARS调度kernel队列流上的kernel，把kernel放到AiCPU上去执行。
- AICPU调用函数（kernel），用一个线程执行kernel 函数，在函数内把通信task展开，把task放到rstqueue上，STARS调用。
- 降低host和aicpu交互，由几百次降低为一次。
- task的提交在AICPU上提交，做了提交的部分合并。

TASK_QUEUE_ENABLE
""""""""""""""""""""""""

**使用方式：** ``export TASK_QUEUE_ENABLE=2``

TASK_QUEUE_ENABLE，下发优化，图模式设置为1（即开启图模式的时候这个要设置为1），非图模式设置为2

示意图：

.. image:: https://github.com/wucong25/verl-data/blob/main/ascend_task_queue2.png
   :alt: ascend task queue

绑核优化
""""""""""""""""""""""""

**使用方式：** ``export CPU_AFFINITY_CONF=1``

详细设置原理可看：https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0059.html

其他
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

以下内容汇总了若干全局环境变量的调优配置。由于这些参数在训练阶段与推理阶段往往都能带来正向收益，且目前尚缺乏足够精细的消融实验来严格区分它们各自对训练或推理的贡献占比，故统一归拢在此，供后续持续监控与进一步拆解分析。

使能jemalloc
""""""""""""""""""""""""

使用方式（注意需要先安装jemalloc库）： ``export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2``

**安装使用教程：** `MindSpeed-RL/docs/install_guide.md · Ascend/MindSpeed-RL - AtomGit | GitCode <https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/install_guide.md#高性能内存库-jemalloc-安装>`_

多流复用
""""""""""""""""""""""""

内存方面有优化

使能方式： ``export MULTI_STREAM_MEMORY_REUSE=1``

原理介绍：https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0040.html

VLLM_ASCEND_ENABLE_FLASHCOMM
""""""""""""""""""""""""

使用方式： ``export VLLM_ASCEND_ENABLE_FLASHCOMM=1``

启用昇腾 NPU 特有的FLASHCOMM高速通信优化技术

地址：https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html

VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE
""""""""""""""""""""""""

使用方式： ``export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1``

启用昇腾 NPU针对大模型推理的稠密计算优化

地址：https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html

VLLM_ASCEND_ENABLE_PREFETCH_MLP
""""""""""""""""""""""""

使用方式： ``export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1``

启用 MLP 层的权重预取机制

.. image:: https://github.com/wucong25/verl-data/blob/main/ascend_prefetch.png
   :alt: ascend prefetch
   :scale: 50%

verl框架参数设置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

主要是内存方面的一些设置开关（注意，这个里面的优化都或多或少会导致吞吐量有一定程度的劣化）

.. code-block:: bash

  # 梯度检查点 (Gradient Checkpointing)
  # 作用: 通过重新计算激活值来节省显存,以计算换内存。在前向传播时不保存中间激活值,反向传播时重新计算,可以显著降低显存占用,允许使用更大的batch size。
  actor_rollout_ref.model.enable_gradient_checkpointing=True

  # 参数卸载 (Parameter Offload)
  # 作用: 将模型参数卸载到CPU内存,训练时再加载回GPU。
  actor_rollout_ref.actor.fsdp_config.param_offload=${offload}  # True  
  actor_rollout_ref.ref.fsdp_config.param_offload=${offload}    # True

  # 优化器状态卸载 (Optimizer Offload)
  # 作用: 将优化器状态(如Adam的动量)卸载到CPU。优化器状态通常占用大量显存(对于Adam,每个参数需要额外8字节),卸载可以节省显存。
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}  # True

  # 释放推理引擎缓存 (Free Cache Engine)
  # 作用: 在训练阶段释放推理引擎的KV cache和权重。这是3D-HybridEngine的核心优化,允许在同一GPU上交替进行推理和训练,显著降低显存需求。
  actor_rollout_ref.rollout.free_cache_engine=True

  #  熵计算优化
  # entropy_checkpointing: 在训练时对熵计算启用重计算,降低显存峰值
  # entropy_from_logits_with_chunking: 分块处理logits张量(如2048 tokens一组),避免一次性加载整个[bsz*seq_len, vocab]张量
  actor_rollout_ref.actor.entropy_checkpointing=True  
  actor_rollout_ref.ref.entropy_checkpointing=True  
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True  
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True

  # 推理引擎显存配置
  # gpu_memory_utilization: 控制vLLM使用的GPU显存比例(0.90 = 90%)
  # enforce_eager=False: 启用CUDA graphs加速推理,但会占用额外显存
  actor_rollout_ref.rollout.gpu_memory_utilization=0.90  
  actor_rollout_ref.rollout.enforce_eager=False

NPU调优参考文章
----------------------------------

环境变量相关：`环境变量列表-Ascend Extension for PyTorch6.0.0-昇腾社区 <https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_001.html>`_

社区性能调优教程：`性能调优流程-Ascend Extension for PyTorch6.0.0-昇腾社区 <https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0001.html>`_


----------------------------------
Ascend SGLang Best Practice
----------------------------------

Last updated: 01/27/2026.

.. _Qwen3-30B: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3moe-30b_sglang_megatron_npu.sh
.. _Qwen2.5-32B: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen2-32b_sglang_fsdp_npu.sh

引言
----------------------------------

SGLang 是当前主流的高性能开源推理引擎, 昇腾已经全面原生支持该推理引擎在verl中使用,
仅需简单的构建流程，开发者即可完成环境构建，本文将提供两个经典用例来帮助开发者了解以下内容：

1. 环境构建
2. 模型训练与评估
3. 性能采集

两个用例模型脚本以及其需要的硬件条件各自如下：

+----------------------+---------------------+----------+------------------------+
| 模型                 | NPU型号             | 节点数量 | 训推后端               |
+======================+=====================+==========+========================+
| `Qwen3-30B`_         | Atlas 800T A3       | 1        | SGLang + Megatron      |
+----------------------+---------------------+----------+------------------------+
| `Qwen2.5-32B`_       | Atlas 900 A2        | 2        | SGLang + FSDP          |
+----------------------+---------------------+----------+------------------------+

环境构建
-----------------------------------

我们在quickstart中提供了两种构建环境的方法, 1.从镜像文件DockerFile进行构建 2.从自定义Conda环境进行构建

在本实践中, 我们额外指定verl 的commit id 以避免引入其他问题

.. code-block:: bash

    cd verl
    git checkout c98cb8cc

模型训练与评估
-----------------------------------

1.模型数据准备
^^^^^^^^^^^

`Qwen3-30B`_
""""""""""""
**下载模型权重**

Qwen3-30B: https://huggingface.co/Qwen/Qwen3-30B-A3B

**下载数据集**

DAPO-Math-17k: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

**HuggingFace To Megatron权重转换(可选)**

.. code-block:: bash

  python scripts/converter_hf_to_mcore.py \
      --hf_model_path Qwen/Qwen3-30B-A3B \
      --output_path Qwen/Qwen3-30B-A3B-mcore \
      --use_cpu_initialization    # Only work for MoE models

*注:verl当前已支持mbridge进行灵活的hf和mcore之间的权重转换,可以修改以下相关参数直接加载hf权重*

.. code-block:: bash

    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.use_mbridge=True

`Qwen2.5-32B`_
""""""""""""""
**下载模型权重**

--local-dir: 模型保存路径

.. code-block:: bash

  export HF_ENDPOINT=https://hf-mirror.com
  hf download --resume-download Qwen/Qwen2.5-32B --local-dir /path/to/local_dir

**下载及处理数据集**

.. code-block:: bash

    wget https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/main/deepscaler.json
    python recipe/r1_ascend/json_to_parquet.py --output_dir ./data/deepscaler --json_path path/to/deepscaler.json --train_data_ratio 0.9

2.训练
^^^^^^^^^^^

根据开发者实际路径配置情况修改模型训练脚本中的以下参数

.. code-block:: bash 

    # Model Weights Paths
    MODEL_PATH=Qwen/Qwen3-30B-A3B
    MCORE_MODEL_PATH=Qwen/Qwen3-30B-A3B-mcore
    RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
    CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

    # File System Paths
    TRAIN_FILE=$RAY_DATA_HOME/dataset/dapo-math-17k.parquet
    TEST_FILE=$RAY_DATA_HOME/dataset/aime-2024.parquet

    #保存频率，-1默认不保存，如需评测请修改此参数
    trainer.save_freq=-1

对于单机任务 `Qwen3-30B`_ , 可以直接bash执行verl仓上示例脚本

.. code-block:: bash 

  bash examples/grpo_trainer/run_qwen3moe-30b_sglang_megatron_npu.sh

对于多节点任务 `Qwen2.5-32B`_ ，我们推荐使用以下脚本进行大规模多节点训练拉起

.. code-block:: bash

  pkill -9 python
  ray stop --force
  rm -rf /tmp/ray
  export RAY_DEDUP_LOGS=0
  export HYDRA_FULL_ERROR=1
  # TASK_QUEUE_ENABLE，下发优化，图模式设置为1，非图模式设置为2
  export TASK_QUEUE_ENABLE=1
  export HCCL_ASYNC_ERROR_HANDLING=0
  export HCCL_EXEC_TIMEOUT=3600
  export HCCL_CONNECT_TIMEOUT=3600
  
  export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
  export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
  # 修改为当前需要跑的用例路径
  DEFAULT_SH="./run_*.sh"
  echo "Use $DEFAULT_SH"
  
  ulimit -n 32768
  mkdir logs
  
  NNODES=2
  NPUS_PER_NODE=8
  # 修改为对应主节点IP
  MASTER_ADDR="IP FOR MASTER NODE"
  # 修改为当前节点的通信网卡
  SOCKET_IFNAME="Your SOCKET IFNAME"
  export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
  # 获取当前IP
  CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
  if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # 主节点启动
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'
  
    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))
  
        # 判断device_count 是否与 NNODES 相等
        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
            ray status
            bash $DEFAULT_SH
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done
  else
    # 子节点尝试往主节点注册 ray 直到成功
    while true; do
        # 尝试连接 ray 集群
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP
  
        # 检查连接是否成功
        ray status
        if [ $? -eq 0 ]; then
            echo "Successfully connected to the Ray cluster!"
            break
        else
            echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
            sleep 5
        fi
    done
  fi
  
  sleep 600

DEFAULT_SH:修改为训练所用配置 sh 文件路径。在此案例中修改为 `Qwen2.5-32B`_ 路径。

NNODES 和 NPUS_PER_NODE:修改为使用节点数量和每个节点 NPU 数量。在此案例中分别为2和8。

MASTER_ADDR:修改为对应主节点 IP。即所有节点的 MASTER_ADDR 应该相同。

SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: 修改为对应通信网卡，通信网卡可以通过以下命令获取：

.. code-block:: bash

  ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1

3.模型评估
^^^^^^^^^^^

不同模型步骤一致,仅以Qwen3-30b为例列举

我们通过 AISBenchmark 评估模型,该工具支持vllm/sglang多种推理后端的评估

**安装方法**

.. code-block:: bash

  git clone https://gitee.com/aisbench/benchmark.git
  cd benchmark
  pip install -e .

**下载评估数据集**

.. code-block:: bash

  cd path/to/benchmark/ais_bench/datasets
  wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
  unzip math.zip
  rm math.zip

**修改AISBench配置代码使能sglang推理评测**

打开 benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py 文件，这是推理配置文件

.. code-block:: bash

    from ais_bench.benchmark.models import VLLMCustomAPIChatStream
    from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content
    from ais_bench.benchmark.clients import OpenAIChatStreamClient, OpenAIChatStreamSglangClient

    models = [
        dict(
            attr="service",
            type=VLLMCustomAPIChatStream,
            abbr='sgl-api-stream-chat',
            path="/path/to/Qwen3-30B", # 修改为 Qwen3-30B 模型路径
            model="qwen3-30b",
            request_rate = 0,
            max_seq_len=2048,
            retry = 2,
            host_ip = "localhost", # 推理服务的IP
            host_port = 8005, # 推理服务的端口
            max_out_len = 8192,  # 最大输出tokens长度
            batch_size=48, # 推理的最大并发数
            trust_remote_code=False,
            custom_client=dict(type=OpenAIChatStreamSglangClient), #使用sglang客户端
            generation_kwargs = dict(
                temperature = 0,
                seed = 1234,
            ),
            pred_postprocessor=dict(type=extract_non_reasoning_content)
        )
    ]

**启动sglang_server服务**

.. code-block:: bash

    python -m sglang.launch_server --model-path "/path/to/Qwen3-30B"  --tp-size 4 --dp-size 1 --port 8005 

**启动sglang_client评测**

.. code-block:: bash

    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt

**评测结果**

经过训练,模型在Math-500上的评分显著上升

+------+----------------------+---------+----------+------+----------------------+
| iter | dataset              | version | metric   | mode | sgl-api-stream-chat  |
+======+======================+=========+==========+======+======================+
|   0  | math_prm800k_500     | c4b6f0  | accuracy | gen  | 	84.4             |
+------+----------------------+---------+----------+------+----------------------+
|  150 | math_prm800k_500     | c4b6f0  | accuracy | gen  |     91.7             |
+------+----------------------+---------+----------+------+----------------------+

性能采集
-----------------------------------

关于NPU profiling的详细文档请参考 `ascend_profiling_zh <https://github.com/volcengine/verl/blob/main/docs/ascend_tutorial/ascend_profiling_zh.rst>`_

在 `Qwen3-30B`_ 的脚本中提供了基本的采集性能选项PROF_CONFIG，默认设置 global_profiler.steps=null 关闭采集， 开发者可根据实际需要进行参数修改

采集完成后，开发者可以使用 `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ 进行数据解析

注: verl框架侧进行采集全量 Profiling 产生海量且重复的算子记录，可以根据文档修改代码仅采集关键阶段


----------------------------------
Ascend Retool Best Practice
----------------------------------

Last updated: 02/10/2026.

引言
----------------------------------

Retool论文参考(`Retool <https://arxiv.org/pdf/2504.11536>`_)
集成代码解释器工具，通过多轮实时代码执行进行策略部署，并教会模型根据结果反馈学习何时以及如何调用工具。

1. 环境构建
2. 模型训练

用例模型脚本以及其需要的硬件条件各自如下：

===============    ============    ============    ===============
模型                NPU型号         节点数量        训推后端
===============    ============    ============    ===============
``Qwen2.5-7B``     Atlas 900 A2         1          ``vllm + FSDP``
===============    ============    ============    ===============

环境构建
-----------------------------------

1.从自定义Conda环境进行构建

============    ============================================================
software        version 
============    ============================================================
Python          ``>= 3.10, <3.12``
CANN            ``== 8.3.RC1``
torch           ``== 2.7.1``
torch_npu       ``== 2.7.1``
verl            ``v0.6.1 commitId=d62da4950573d7a4b7ef2362337952e7ab59e78d``
vllm            ``v0.11.0``
vllm-ascend     ``v0.11.0-dev``
transformers    ``4.57.6``
============    ============================================================

模型训练与评估
-----------------------------------

1.模型数据准备
^^^^^^^^^^^

`Qwen2.5-7B`
""""""""""""
**下载模型权重**

--local-dir: 模型保存路径

.. code-block:: bash

  git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

**下载训练数据集**

.. code-block:: bash

  git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

**下载评估数据集**

.. code-block:: bash

  git clone https://huggingface.co/datasets/Maxwell-Jia/AIME_2024

**下载预训练数据集**

.. code-block:: bash

  python3 recipe/retool/retool_sft_preprocess.py

*注:自动下载ReTool-SFT，最后生成数据默认保存在~/ReTool-SFT/data目录下*

**执行预训练脚本**

.. code-block:: bash

  bash recipe/retool/run_qwen2_7b_sft_npu.sh # 需适配脚本中路径

**合并预训练权重生成checkpoint**

.. code-block:: bash

  python3 -m verl.model_merger merge --backend fsdp \
      --local_dir ${DATASETS}/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/global_step_372 \
      --target_dir ${DATASETS}/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/global_step_372/huggingface

2.代码沙箱准备
^^^^^^^^^^^

开源沙箱代码及部署参考
https://github.com/bytedance/SandboxFusion

**沙箱代码下载**

.. code-block:: bash

  git clone -b main https://github.com/bytedance/SandboxFusion.git

**沙箱安装**

.. code-block:: bash

  conda create -n sandbox -y python=3.11
  conda activate sandbox
  pip install poetry
  poetry lock
  poetry install
  mkdir -p docs/build
  cd runtime/python
  bash install-python-runtime.sh
  make run-online

3.训练
^^^^^^^^^^^

示例配置文件如下，在recipe/retool目录下创建一个run_qwen2.5_7b_dapo_npu.sh
根据开发者实际路径配置情况修改模型训练脚本中的以下参数

.. code-block:: bash 

  set -x

  export VLLM_USE_V1=1
  export TORCHDYNAMO_DISABLE=1
  export VLLM_ASCEND_ENABLE_NZ=0
  export TASK_QUEUE_ENABLE=1
  export VLLM_ENABLE_GRAPH_MODE=1
  export HCCL_OP_EXPANSION_MODE="AIV"
  export VLLM_ASCEND_ENABLE_MLP_OPTIMIZE=1
  export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2
  
  # ================= data/model/tool =================
  HDFS_ROOT=${HDFS_ROOT:-"${PWD}"}
  DATA_ROOT=${DATA_ROOT:-"${PWD}"}
  
  dapo_math_17k=$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k
  aime_2024=$DATA_ROOT/dataset/Maxwell-Jia/AIME_2024
  #aime_2025=$DATA_ROOT/dataset/yentinglin/aime_2025
  model_path=$DATA_ROOT/dataset/checkpoint/multiturn-sft-qwen-2.5-7b-instruct/  global_step_372/huggingface
  
  train_files="['$dapo_math_17k']"
  test_files="['$aime_2024']"
  
  # tool
  tool_config_path=recipe/retool/sandbox_fusion_tool_config.yaml
  
  # wandb
  project_name=retool
  experiment_name=qwen2.5-7b_dapo
  default_local_dir=$DATA_ROOT/checkpoint/$experiment_name
  
  # 创建日志文件
  export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  LOG_DIR="$HDFS_ROOT/verl/logs/$project_name/$experiment_name"
  # 判断路径是否存在
  if [ ! -d "$LOG_DIR" ]; then
    # 路径不存在，创建路径
    mkdir -p "$LOG_DIR"
    echo "Directory $LOG_DIR created."
  else
    echo "Directory $LOG_DIR already exists."
  fi
  
  LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
  touch "$LOG_FILE"
  echo "Log file $LOG_FILE created."

  # ================= algorithm =================
  adv_estimator=grpo
  
  use_kl_in_reward=False
  kl_coef=0.0
  use_kl_loss=False
  kl_loss_coef=0.0
  
  clip_ratio_low=0.2
  clip_ratio_high=0.28
  
  max_turns=16
  max_prompt_length=2048
  max_response_length=20480
  actor_lr=1e-6
  
  train_batch_size=32
  ppo_mini_batch_size=16
  
  n_resp_per_prompt=16
  n_resp_per_prompt_val=30
  
  # ================= performance =================
  infer_tp=2 # vllm
  train_sp=4 # train
  offload=True
  
  actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
  log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

  PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.max_num_batched_tokens=$actor_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=20 \
    trainer.device=npu \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    trainer.total_epochs=1 $@ > $LOG_FILE 2>&1 &

