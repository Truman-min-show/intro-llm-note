# DeepSpeed-Chat 指令微调实践

## 简介
本项目提供了使用 **DeepSpeed** 和 **PyTorch** 进行 **Qwen3-0.6B** 模型指令微调训练的示例代码。它支持分布式训练、LoRA（低秩适应）、训练数据的动态加载、内存优化等功能，旨在帮助用户高效地进行大规模语言模型的训练。

## 使用说明
- 下载 **Qwen3-0.6B** 预训练模型至 `models` 目录下。
- 运行第一阶段训练脚本 `run_qwen_0.6b.sh` ，训练指令微调用模型。
```
bash training\step1_supervised_finetuning\training_scripts\qwen3-0.6b\run_qwen_0.6b.sh
```