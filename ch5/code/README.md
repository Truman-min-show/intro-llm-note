# LoRA-test：

trainable params: 2,359,296 || all params: 1,231,940,608 || trainable%: 0.1915

可以直观地看到使用LoRA微调训练的mt0-large模型的参数量仅为全量的0.2%，极大地减少了计算量

# Qwen3-0.6B 指令微调 — 实验 README（简明版）

## 概述

在单卡 1/2 NVIDIA H20（约 40GB 显存）上，使用 DeepSpeed 对 **Qwen3-0.6B** 进行指令微调，数据为 `data/MyDataset/CVPR2024Summary.json`（≈300 条科研论文摘要 → 总结对）。本次实验目标是：在受限硬件与小样本条件下验证微调流程并得到可解释的训练曲线（loss / perplexity）。

---

## 环境（关键项）

* GPU：1/2 NVIDIA H20（半卡，≈40GB）
* DeepSpeed：0.17.5
* PyTorch / Transformers：支持 qwen3 的版本（源码或新版 Transformers）
* 模型：`./models/Qwen3-0.6B`
* 数据：`./data/MyDataset/CVPR2024Summary.json`（字段 `prompt` / `chosen` / `rejected`）是我在大创项目中根据同学对摘要的评分收集的文献摘要数据集，地址是我的 [res](https://github.com/Truman-min-show/CVPR2024-DPO-Summary)
* 训练脚本调用（超参数设置如下）：

  ```bash
  deepspeed --master_port=${MASTER_PORT} main.py \
    --model_name_or_path ./models/Qwen3-0.6B \
    --data_path ./data/MyDataset/CVPR2024Summary.json \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 5e-6 \
    --max_seq_len 512 \
    --output_dir ./outputs/qwen0.6b_step1 \
    --seed 42 \
    --dtype bf16 \
    --gradient_checkpointing \
    --deepspeed \
    --data_output_path ./data/cache
  ```

---

## 我做的主要改动（相对于原书）

1. **模型替换**

   * 用 Qwen3-0.6B 替换原教程的 Baichuan-7B，并修改 `main.py` 中对应的模型/分词器导入逻辑。
2. **数据加载 / `my_load`**（核心修复）

   * `tokenizer` 对 `Human: <prompt> \n Assistant: <chosen>` 做分词、`padding` 与 `truncation`。
   * **将所有 pad token 在 labels 中替换为 `-100`**（避免 pad 被计进 loss）。
   * 用 `datasets.Dataset.map` 生成 `input_ids/attention_mask/labels` 并做 90/10 划分。
3. **训练超参与 DeepSpeed 调整**

   * per-device batch = 8，grad\_accum = 4（有效 micro/全局等设置见上）。
   * 学习率降为 `5e-6`（全参微调时更稳）。
   * 使用 `bf16` + gradient checkpointing 以节省显存与提高速度。
4. **批处理准备**

   * 修改 `_prepare_batch_for_model`，能兼容 `Tensor` / `np.ndarray` / `list` 等输入，避免类型转换错误。

---

## 训练结果（从 `train.log` 提取，节选）

训练过程中 eval 的 loss / perplexity 随 epoch 的变化如下（节选）：

* Epoch 0 — loss ≈ 3.0460, ppl ≈ 21.03
* Epoch 1 — loss ≈ 2.7887, ppl ≈ 16.26
* Epoch 2 — loss ≈ 2.6612, ppl ≈ 14.31
* Epoch 3 — loss ≈ 2.5731, ppl ≈ 13.11
* Epoch 4 — loss ≈ 2.5323, ppl ≈ 12.58
* Epoch 5 — loss ≈ 2.5177, ppl ≈ 12.40
* Epoch 6 — loss ≈ 2.5074, ppl ≈ 12.27
* Epoch 7 — loss ≈ 2.5025, ppl ≈ 12.21
* Epoch 9 — loss ≈ 2.5031, ppl ≈ 12.22

（训练日志可查阅：train.log。）

**结论**：经过上述修改后，loss 与 perplexity 在多数 epoch 中呈**稳定下降并收敛到 ≈2.50 / ppl ≈12**，训练稳定性大幅提升（相比先前 pad 未 mask 时的发散/回弹问题）。

---

## 改动的简要说明

* **labels 中把 pad → -100**：避免将大量填充 token 计入交叉熵，从而消除无效/噪声梯度，是训练质量提升的最重要修复项。
* **更小学习率（5e-6）**：在仅 300 条样本上全参训练，较小 lr 更稳定、减少过拟合与发散风险。
* **bf16 + grad checkpointing**：在 H20 上既利用数值/算力优势又降低显存压力，使较大 seq len（512）可行。
* **数据划分 + 动态 map**：合理的划分与 `datasets.map` 管道让数据预处理统一、无类型冲突（减少运行时错误）。

---

## 经验与下一步

1. **改用 LoRA / Adapter**：如果想在更短时间或更低显存下得到更稳健效果，建议只微调 LoRA（`--lora_dim 8 --only_optimize_lora`），并把 LoRA lr 调到 `1e-4`。
2. **减小 max\_seq\_len 或使用动态 padding + collator**：你的文本偏长（论文摘要），若大部分样本远小于 512，可把 `max_seq_len` 降到 384，或在 `tokenizer(..., padding='longest')` 并使用 `DataCollator`，减少无效 tokens。
3. **混合更多 SFT 数据**：300 条数据对通用性不足，建议把小数据和公开 SFT（Alpaca/ShareGPT 等）混合训练以提高泛化并减少过拟合风险。
4. **评估多维指标**：除了 ppl/loss，做若干抽样测试（人工检查摘要质量）或自动指标（ROUGE/BLEU/bert-score）以更准确衡量“总结”质量。
5. **早停 / learning-rate schedule**：加入早停或更长的学习率衰减策略可以进一步稳定训练。








