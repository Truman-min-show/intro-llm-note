import re
import matplotlib.pyplot as plt

# 读取日志文件
log_file = "./train.log"
losses = []
ppls = []
epochs = []

pattern = re.compile(r"ppl:\s*([\d.]+),\s*loss:\s*([\d.]+)")

with open(log_file, "r", encoding="utf-8") as f:
    epoch = 0
    for line in f:
        match = pattern.search(line)
        if match:
            ppl = float(match.group(1))
            loss = float(match.group(2))
            ppls.append(ppl)
            losses.append(loss)
            epochs.append(epoch)
            epoch += 1

# 绘制双y轴曲线
fig, ax1 = plt.subplots(figsize=(8,5))

color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs, losses, marker='o', color=color, label="Loss")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 共享 x 轴
color = 'tab:orange'
ax2.set_ylabel('Perplexity', color=color)
ax2.plot(epochs, ppls, marker='s', color=color, label="Perplexity")
ax2.tick_params(axis='y', labelcolor=color)

# 标题 & 网格
fig.suptitle("Loss and Perplexity over Epochs")
ax1.grid(True, linestyle="--", alpha=0.6)

plt.show()
