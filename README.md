# lfw_cnn

LFW 图片加载实验，包含 PyTorch Dataset 和保留旧方法名的批量加载器。仓库不包含 CNN 模型、训练流程、LFW 图片或标签文件。

[在线维护文档](https://xiaoshecode.github.io/xiaoshecode/repos/lfw_cnn/)

## 安装

使用独立 Python 3.10+ 环境，按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择与平台匹配的 torch/torchvision，再安装其余依赖：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Bash: source .venv/bin/activate
python -m pip install -r requirements.txt
```

依赖文件声明兼容下限，未锁定所有平台的 wheel；本轮 CPU 测试使用的确切版本见在线文档。

## 数据格式

```text
dataset/
├── easy_trainingdata/       图片，可含子目录
├── male_names.txt           UTF-8，每行一个相对图片路径
└── female_names.txt         同上；空行忽略
```

标签 0/1 仅沿用输入文件的分组，代码不推断或核实人的属性。两份列表不允许包含同一图片；路径不能越出图片目录。默认从每组不重复抽取 250 张，可通过正偶数 `data_count` 修改；数量不足会报错。数据及使用许可需自行准备。

## 使用 PyTorch Dataset

```python
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from lfw_loader import LFW

dataset = LFW(
    "dataset/easy_trainingdata",
    data_count=500,
    seed=42,
    transform=Compose([Resize((128, 128)), ToTensor()]),
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(loader))
```

没有指定 transform 时返回 RGB、`uint8`、CHW 张量，值域 0–255；传入 transform 时接收 PIL RGB 图像。不同尺寸图片组 batch 前需要 Resize。可用 `male_file`、`female_file` 显式指定标签文件。

## 兼容加载器

```python
from LFW_pytorch_loader import DataLoader

data = DataLoader("dataset/easy_trainingdata", data_count=500, seed=42)
data.loadLFW_label()
data.loadLFW_facedata()
train_images, train_labels, train_onehot = data.getTrainData()
```

兼容加载器按实际成功加载数量划分 65%/20%/15%，每个实例独立存储，重复加载不会累积。该划分是图片级随机划分，不保证类别分层或人物身份隔离；研究评测需另行使用正确的身份划分协议。

## 目录与检查

| 路径 | 职责 |
| --- | --- |
| `lfw_data.py` | 共用标签读取、去重、采样和路径检查 |
| `lfw_loader.py` | 惰性 PyTorch Dataset，支持 len/index/transform |
| `LFW_pytorch_loader.py` | 旧接口兼容及一次性加载、划分 |
| `tests/test_loaders.py` | 合成图片回归，验证张量、标签、批次、隔离和非法输入 |

```bash
python -m unittest discover -s tests -v
```

导入模块不读取数据、不启动训练。检查使用合成小图片，不代表真实 LFW 模型准确率、GPU 训练或数据授权已验证。仓库尚未声明代码许可证。
