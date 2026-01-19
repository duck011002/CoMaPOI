# CoMaPOI: 基于多智能体协作的下一个POI预测框架

本仓库包含论文 **CoMaPOI: A Collaborative Multi-Agent Framework for Next POI Prediction** 的官方实现。CoMaPOI 是一个创新框架，通过多智能体协作的方式利用大语言模型（LLM）来增强下一个兴趣点（POI）预测。

虽然大语言模型功能强大，但在直接应用于POI预测任务时面临两个关键挑战：

1. **缺乏时空理解能力**：LLM 本身难以理解坐标、时间和距离等原始数值数据，这阻碍了对用户移动模式的准确建模。
2. **候选空间巨大**：城市中潜在POI数量庞大且不受约束，往往导致不相关或随机的预测。

![CoMaPOI 框架架构](./MODEL.png)

CoMaPOI 通过将预测任务分解给三个专门的协作智能体来解决这些挑战：

* 🤖 **画像智能体 (Profiler Agent)**：将原始数值轨迹数据转换为丰富的语义语言描述，使LLM能够理解用户的画像和移动模式。
* 🎯 **预测智能体 (Forecaster Agent)**：动态约束和精炼庞大的候选POI空间，提供更小、更高质量的可能选项集。
* 🧠 **决策智能体 (Predictor Agent)**：整合来自画像智能体和预测智能体的结构化信息，生成最终的高精度预测。

该框架不仅开创性地将多智能体系统用于这一复杂的时空任务，还提供了从数据生成（使用我们提出的**逆向推理微调（RRF）**策略）到模型微调和推理的完整流程。我们的工作展示了最先进的性能，在三个基准数据集上将关键指标提高了 **5% 到 10%**。

## 项目结构

```
CoMaPOI/
├── README.md                  # 项目文档
├── requirements.txt           # 环境依赖
├── agents.py                  # 智能体定义
├── evaluate.py                # 评估函数
├── finetune_sft_new.py        # 微调脚本
├── inference_forward_new.py   # 多智能体推理脚本
├── inference_inverse_new.py   # 逆向推理脚本
├── inference_ori_new.py       # 单智能体推理脚本
├── parser_tool.py             # 解析工具
├── prompt_provider.py         # 提示词生成
├── utils.py                   # 工具函数
├── docs/                      # 文档目录
│   ├── Forward_Inference.md   # 多智能体推理文档
│   ├── RRF.md                 # 逆向推理文档
│   ├── SFT_Finetune.md        # 微调文档
│   ├── Single_Agent_Inference.md # 单智能体推理文档
├── rag/                       # 检索增强生成
│   └── RAG.py                 # RAG实现
├── tool/                      # 工具目录
│   └── base_tools.py          # 基础工具
├── dataset_all/               # 数据集目录
└── results/                   # 结果目录
```

## 快速开始

### 1. 环境配置

使用 conda 创建虚拟环境：

```bash
# 创建并激活环境
conda create -n comapoi python=3.8
conda activate comapoi

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据集下载

CoMaPOI 支持三个数据集：
- NYC（纽约）
- TKY（东京）
- CA（加利福尼亚）

上述三个处理好的数据集可以从以下链接下载：[https://huggingface.co/datasets/Chips95/Data_CoMaPOI_SIGIR_2025]

下载后将数据集放入 `dataset_all/` 目录。

## 核心组件

### 1. 逆向推理（数据生成）

逆向推理模块（`inference_inverse_new.py`）基于目标POI生成合成训练数据。它使用语言模型创建真实的用户画像、移动模式和POI偏好。

```bash
python inference_inverse_new.py --dataset nyc --api_type qwen2.5-7b-instruct --batch_size 32
```

### 2. 模型微调

微调模块（`finetune_sft_new.py`）使用 LoRA 技术对大语言模型进行参数高效微调，使其适应特定的智能体角色。

```bash
python finetune_sft_new.py --dataset nyc --model llama3.1-8b-instruct --type merged --batch_size 16 --max_steps 200
```

### 3. 多智能体推理

前向推理模块（`inference_forward_new.py`）实现了预测用户下一个访问POI的多智能体方法。它协调三个专门的智能体（画像智能体、预测智能体和决策智能体）。

```bash
python inference_forward_new.py --dataset nyc --model llama3.1-8b-instruct --agent1_api agent1 --agent2_api agent2 --agent3_api agent3
```

### 4. 单智能体推理

单智能体推理模块（`inference_ori_new.py`）使用单个智能体实现更简单的POI预测方法。

```bash
python inference_ori_new.py --dataset nyc --prompt_format json --model llama3.1-8b-instruct --batch_size 16
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- AgentScope
- PEFT（参数高效微调）
- TRL（Transformer强化学习）
- tqdm
- concurrent.futures

## 跨平台执行

如需在 Mac 上运行 CoMaPOI 并使用部署在 Linux 服务器上的 VLLM API：

1. 确保 VLLM API 可从 Mac 访问（检查网络连接）
2. 更新命令行参数中的端口以匹配服务器配置
3. 在 Mac 上安装所需的 Python 包
4. 按照使用示例中的说明运行脚本

## 详细文档

有关每个组件的更多详细信息，请参阅 `docs/` 目录中的文档：

- [多智能体推理](docs/Forward_Inference.md)
- [逆向推理](docs/RRF.md)
- [模型微调](docs/SFT_Finetune.md)
- [单智能体推理](docs/Single_Agent_Inference.md)
