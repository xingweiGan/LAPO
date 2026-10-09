# How to Run

Use a machine with an NVIDIA GPU and a working CUDA environment.

## 1. First-time setup

Replace the path below with the actual location of this folder:

```bash
cd /path/to/LAPO
pip install uv
uv sync --no-install-package flash-attn
uv sync
#uv run pytest
source .venv/bin/activate
export HF_TOKEN="YOUR_HUGGING_FACE_TOKEN"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export HF_USERNAME="YOUR_HF_USERNAME_OR_ORGANIZATION"
pip install vllm
```

On Runpod, provide these same three names through the template environment variables/secrets. `HF_USERNAME` selects the account or organization for model uploads. Replace the placeholders only when running locally; do not put actual credentials in this README.

Your Hugging Face token must allow access to the required models and uploading training results. The first run requires internet access to download models and the dataset.

## 2. Run one training script

All five scripts use the Qwen2.5-3B model family and the `Maxwell-Jia/MATH` dataset. The four post-SFT scripts use the fixed SFT checkpoint `xw1234gan/SFT_Qwen2.5-3B-Instruct_MATH`; no additional SFT checkpoint environment variable is required. Each script runs independently.

SFT (`Train_MATH_3B_SFT.py`): fine-tunes `Qwen/Qwen2.5-3B-Instruct` on the first 768 MATH training examples using the provided solutions as supervised targets. It evaluates with vLLM and uploads the trained model to `${HF_USERNAME}/SFT_Qwen2.5-3B-Instruct_MATH_MATH3B_copy`.

```bash
python Train_MATH_3B_SFT.py
```

This SFT script uses two visible GPUs: `cuda:0` for training and `cuda:1` for evaluation. Its uploaded checkpoint is a separate experiment output and does not automatically replace the fixed SFT checkpoint used by the four post-SFT scripts.

GRPO + KL (`Train_MATH_3B_GRPO_KL.py`): fine-tunes the MATH SFT model using clipped GRPO with a KL penalty (`beta=0.01`) against a frozen copy of the same SFT model.

```bash
python Train_MATH_3B_GRPO_KL.py
```

Fixed (`Train_MATH_3B_Fixed.py`): trains a Qwen2.5-3B-Instruct policy using GRPO with its logits mixed with a frozen MATH SFT model. The mixing weight stays fixed at `alpha=0.5`.

```bash
python Train_MATH_3B_Fixed.py
```

Adaptive (`Train_MATH_3B_Adaptive.py`): uses the same policy initialization and logit mixing as Fixed, but starts at `alpha=0.5` and adjusts the mixing weight based on evaluation results.

```bash
python Train_MATH_3B_Adaptive.py
```

Adaptive validation-weighted (`Train_MATH_3B_Adaptive_ValWeighted.py`): updates the mixing weight from the validation accuracies of the trainable policy, mixed policy, and frozen SFT policy on the same validation questions.

```bash
python Train_MATH_3B_Adaptive_ValWeighted.py
```

Choose one command to start the corresponding training run.

## “简略图”的制作流程与指导

以后用户要求“简略图”时，按照下面的方法从当前代码整理，不直接复用可能已经过期的旧图。

### 制作流程

1. **先确认图的范围。** 默认画一次完整运行的时间顺序，包括启动阶段、每轮循环和训练结束后的操作，而不是画类、函数或模块关系。
2. **沿主执行路径读代码。** 依次定位模型加载、数据选择、生成、reward/advantage、反向传播、validation、状态更新、W&B 记录和 Hugging Face 上传；以实际执行代码为准，不只看注释或变量名。
3. **先分成三个时间段。** 把一次性准备放在“启动”部分，把重复操作放进“每一轮”边界，把只执行一次的保存或上传放在最后。
4. **追踪关键对象和状态。** 标明哪个 policy 可训练、哪个冻结、哪个由两者混合；同时追踪 `alpha/old`、`alpha/target`、`alpha/new` 分别在哪一步计算、在哪一轮使用。
5. **确认数据对应关系。** 写清训练和 validation 各取多少题、是否按顺序、每题生成多少回答，以及多个 policy 的 accuracy 是否在同一批问题上计算。提前缓存的结果也要标明它对应的是哪一轮、哪一批问题。
6. **确认更新边界。** 写清哪一步发生梯度更新、实际更新哪个模型、哪些模型始终冻结，以及一个新参数是立即生效还是下一轮才生效。
7. **单独核对外部交互。** 从实际 `wandb.log` 调用提取准确字段和记录时机；从实际 Hub 上传代码确认上传发生的时间与包含的内容。
8. **去重并压缩。** 删除同义指标、框架自动生成的重复 step，以及前后重复说明；保留会影响理解或复现实验的数量、超参数、公式和先后关系。
9. **转成纵向纯文本图。** 每个节点只表达一个主要动作，细节缩进放在节点下方，用 `↓` 连接严格的执行顺序，用明显的横线标出循环范围。
10. **最后反向核验。** 从图的第一行沿箭头走到最后一行，与代码逐步对照，检查是否遗漏、调换顺序或使用了旧参数。

### 输出规则

- 使用 `text` 代码块，使内容可以直接复制到 Markdown。
- 默认不使用 Mermaid、图片或表格。
- 使用 `↓` 表示顺序，使用 `================ 每一轮 ================` 标出循环。
- 关键数量和超参数直接写在对应步骤中，不集中堆在图外。
- W&B 必须列出实际记录的字段；Hugging Face 必须说明真正的上传时机。
- 区分“当前值”“本轮更新后的值”和“下一轮使用的值”。
- 简短但不能用“做训练”“记录参数”等模糊表述代替真实步骤。
- 代码变化后先重新核对，再更新图中的数值、字段和顺序。

### 当前 LAPO 示例

```text
启动
  ↓
加载：
  可训练策略 θ
  冻结 SFT 策略
  α = 0.5
  ↓
提前计算冻结 SFT 在 40 轮 validation 问题上的结果
（每轮对应的 100 题仍与 θ、mixed 使用完全相同的问题）
  ↓
================ 每一轮 ================
  ↓
按顺序取 256 个训练问题
  ↓
用当前 mixed policy 生成答案
每题 8 个回答，共 2048 个 rollouts
mixed logits = (1-α)·θ logits + α·SFT logits
  ↓
计算 reward 和 GRPO advantage
  ↓
分 batch 反向传播，只更新 θ
本轮训练过程中 α 保持不变
  ↓
每个 effective batch 记录到 W&B：
  grpo_step
  effective_batch
  train/loss
  train/avg_entropy
  train/grad_norm
  train/alpha
  ↓
按顺序取接下来的 100 个 validation 问题
  ↓
在完全相同的 100 题上计算：
  q_theta = 当前 θ 的 accuracy
  q_mix   = 当前 mixed policy 的 accuracy
  q_ref   = 冻结 SFT 的 accuracy（读取提前缓存的结果）
  ↓
用 τ = 0.25 计算 controller 权重：
  w_theta, w_mix, w_ref
  ↓
计算：
  alpha/target
  alpha/new
使用 alpha_update_rate = 0.5
并限制在 [0.05, 0.95]
  ↓
在本轮最后一个 W&B step 额外记录：
  eval/accuracy
  eval_pi_theta
  alpha/target
  alpha/new
  alpha/q_theta
  alpha/q_mix
  alpha/q_ref
  alpha/w_theta
  alpha/w_mix
  alpha/w_ref
  ↓
下一轮使用 alpha/new
========================================
  ↓
完成全部 40 轮
  ↓
最后一次性上传到 Hugging Face：
  最终 θ 权重
  tokenizer
  最终 lapo_alpha
  frozen SFT reference 信息
  controller 类型
```
