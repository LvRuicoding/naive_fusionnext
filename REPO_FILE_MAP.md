# FusionNeXt 文件功能对照表

这份文档只解释当前仓库里训练主路径相关的目录和文件。
如果你只是想看清模型和数据流，优先看“建议先看”这一节，不要一开始把所有文件都展开。

## 建议先看

按这个顺序看，最容易建立整体理解：

1. [configs/fusionnext_nuscenes_mini_3d.py](/home/dataset-local/lr/code/fusionnext/configs/fusionnext_nuscenes_mini_3d.py)
2. [tools/train.py](/home/dataset-local/lr/code/fusionnext/tools/train.py)
3. [fusionnext/models/detectors/fusionnext.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/detectors/fusionnext.py)
4. [fusionnext/models/fusion_models/fusionnext_core.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/fusion_models/fusionnext_core.py)
5. [fusionnext/models/backbones/fusionnext_backbone.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/backbones/fusionnext_backbone.py)
6. [fusionnext/models/serialization/geometry_serializer.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/serialization/geometry_serializer.py)
7. [fusionnext/models/layers/flash_window_block.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/layers/flash_window_block.py)
8. [fusionnext/models/dense_heads/fusionnext_simple_head.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/dense_heads/fusionnext_simple_head.py)

## 顶层目录

| 路径 | 作用 |
| --- | --- |
| `configs/` | 训练配置。决定模型、数据集、优化器、hook、工作目录等。 |
| `fusionnext/` | 主 Python 包。数据集、数据管线、模型都在这里。 |
| `tools/` | 训练启动脚本。 |
| `scripts/` | 留作辅助脚本目录，目前基本没有主训练逻辑。 |
| `work_dirs/` | 训练日志、checkpoint、运行时备份配置输出目录。不是源码。 |
| `README.md` | 仓库说明文档。 |
| `pyproject.toml` | 包安装配置，定义 `fusionnext` 作为独立仓库如何被安装。 |

## 训练主路径

训练时最核心的数据流是：

`config -> tools/train.py -> dataset/pipeline -> detector -> tokenizer -> fusion backbone -> bbox head`

可以对应到这些文件：

| 路径 | 作用 |
| --- | --- |
| [configs/fusionnext_nuscenes_mini_3d.py](/home/dataset-local/lr/code/fusionnext/configs/fusionnext_nuscenes_mini_3d.py) | 主配置文件。把数据集、模型、训练策略拼起来。 |
| [tools/train.py](/home/dataset-local/lr/code/fusionnext/tools/train.py) | 训练入口。读取配置并启动 `mmengine.Runner`。 |
| [fusionnext/datasets/fusionnext_nuscenes_dataset.py](/home/dataset-local/lr/code/fusionnext/fusionnext/datasets/fusionnext_nuscenes_dataset.py) | nuScenes 数据集定义。负责从 `infos` 中组织样本。 |
| [fusionnext/datasets/pipelines/fusionnext_nuscenes.py](/home/dataset-local/lr/code/fusionnext/fusionnext/datasets/pipelines/fusionnext_nuscenes.py) | 数据管线。负责加载图像、点云、标注并打包成训练输入。 |
| [fusionnext/models/data_preprocessors/fusion_det3d_data_preprocessor.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/data_preprocessors/fusion_det3d_data_preprocessor.py) | 模型前的数据预处理，负责 batch 级整理。 |
| [fusionnext/models/detectors/fusionnext.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/detectors/fusionnext.py) | 最外层 detector。串起特征提取、loss、predict。 |

## `fusionnext/datasets/`

这一层负责“把磁盘上的样本变成模型能吃的输入”。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/datasets/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/datasets/__init__.py) | 注册并导出数据集相关模块。 |
| [fusionnext/datasets/fusionnext_nuscenes_dataset.py](/home/dataset-local/lr/code/fusionnext/fusionnext/datasets/fusionnext_nuscenes_dataset.py) | 自定义 nuScenes 数据集，兼容当前仓库需要的样本字段组织方式。 |

### `fusionnext/datasets/pipelines/`

这一层负责“单样本读取和打包”。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/datasets/pipelines/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/datasets/pipelines/__init__.py) | 导出 pipeline 组件。 |
| [fusionnext/datasets/pipelines/fusionnext_nuscenes.py](/home/dataset-local/lr/code/fusionnext/fusionnext/datasets/pipelines/fusionnext_nuscenes.py) | 读取相机图像、点云、3D 标注、相机参数，并整理成 `img_inputs`、`points`、`data_samples`。这是数据管线 debug 的首要文件。 |

## `fusionnext/models/`

这一层负责“模型本体”。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/__init__.py) | 注册并导出模型相关模块。 |

### `fusionnext/models/detectors/`

最外层模型壳子。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/detectors/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/detectors/__init__.py) | 导出 detector。 |
| [fusionnext/models/detectors/fusionnext.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/detectors/fusionnext.py) | 模型总入口。`extract_feat()` 调用融合 core，`loss()` 和 `predict()` 调用检测头。 |

### `fusionnext/models/fusion_models/`

把各个组件拼成融合主干。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/fusion_models/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/fusion_models/__init__.py) | 导出 fusion model。 |
| [fusionnext/models/fusion_models/fusionnext_core.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/fusion_models/fusionnext_core.py) | 融合核心。内部包含 `ImageTokenizer`、`RealLidarTokenizer`、`FusionNeXtBackbone`。 |

### `fusionnext/models/tokenizers/`

负责把原始模态变成 token。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/tokenizers/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/tokenizers/__init__.py) | 导出 tokenizer。 |
| [fusionnext/models/tokenizers/image_tokenizer.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/tokenizers/image_tokenizer.py) | 把多视角图像送进 ResNet50，展开成 image tokens，并生成每个 token 的 `(camera_id, u, v, depth)`。 |
| [fusionnext/models/tokenizers/lidar_tokenizer.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/tokenizers/lidar_tokenizer.py) | 把点云做 voxel 化并聚合成 lidar tokens，同时给出每个 voxel token 的 3D 中心坐标。 |

### `fusionnext/models/backbones/`

负责真正的多层融合。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/backbones/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/backbones/__init__.py) | 导出 backbone。 |
| [fusionnext/models/backbones/fusionnext_backbone.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/backbones/fusionnext_backbone.py) | 交替执行 `2d -> 3d -> 2d` 三层融合。每层都先排序，再做 attention，再按逆索引恢复 token 顺序。 |

### `fusionnext/models/serialization/`

这里是你最关心的“1D token 排序”位置。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/serialization/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/serialization/__init__.py) | 导出 serializer。 |
| [fusionnext/models/serialization/geometry_serializer.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/serialization/geometry_serializer.py) | 核心排序逻辑。`2d` 模式把 lidar token 投影到图像平面排序，`3d` 模式把 image token 反投影到 3D 排序。这里也是你检查“1D token 排序有没有生效”的第一现场。 |

### `fusionnext/models/layers/`

排序后的 token 在这里发生交互。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/layers/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/layers/__init__.py) | 导出层模块。 |
| [fusionnext/models/layers/flash_window_block.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/layers/flash_window_block.py) | 排序后的 1D 序列计算块。当前 attention 分支已经是 `X <- X + Attn(RMSNorm(X))`，后面再接一层 MLP 残差。 |

### `fusionnext/models/dense_heads/`

负责把融合后的 token 变成检测结果。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/dense_heads/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/dense_heads/__init__.py) | 导出检测头。 |
| [fusionnext/models/dense_heads/fusionnext_simple_head.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/dense_heads/fusionnext_simple_head.py) | 检测头。只取前面的 lidar tokens 做 3D 检测，同时拼接整条融合序列的全局上下文。训练时用 `points-in-boxes` 给 lidar token 分配目标。 |

### `fusionnext/models/data_preprocessors/`

负责模型输入进入网络前的 batch 整理。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/data_preprocessors/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/data_preprocessors/__init__.py) | 导出 data preprocessor。 |
| [fusionnext/models/data_preprocessors/fusion_det3d_data_preprocessor.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/data_preprocessors/fusion_det3d_data_preprocessor.py) | 训练前的 batch 级输入整理模块。 |

### `fusionnext/models/utils/`

一些辅助函数，主要用来整理输入和几何计算。

| 路径 | 作用 |
| --- | --- |
| [fusionnext/models/utils/__init__.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/utils/__init__.py) | 导出工具函数。 |
| [fusionnext/models/utils/fusion_inputs.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/utils/fusion_inputs.py) | 把 `points`、`img_inputs`、`img_metas` 组装成模型 core 需要的统一输入格式。 |
| [fusionnext/models/utils/geometry.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/utils/geometry.py) | 几何辅助函数。 |

## 你做 debug 时最值得盯的文件

如果你想查“数据管线怎么读数据”和“1D token 排序是否生效”，优先盯这几个：

| 路径 | 为什么要看 |
| --- | --- |
| [fusionnext/datasets/pipelines/fusionnext_nuscenes.py](/home/dataset-local/lr/code/fusionnext/fusionnext/datasets/pipelines/fusionnext_nuscenes.py) | 看原始样本是怎么被读进来、怎么变成 `results` 的。 |
| [fusionnext/models/utils/fusion_inputs.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/utils/fusion_inputs.py) | 看数据是怎么从 dataset 输出被整理成模型输入的。 |
| [fusionnext/models/tokenizers/image_tokenizer.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/tokenizers/image_tokenizer.py) | 看 image token 是怎么生成的。 |
| [fusionnext/models/tokenizers/lidar_tokenizer.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/tokenizers/lidar_tokenizer.py) | 看 lidar token 和 voxel 坐标是怎么生成的。 |
| [fusionnext/models/serialization/geometry_serializer.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/serialization/geometry_serializer.py) | 看投影、反投影、量化和排序键。这里最能直接验证你的 1D token 排序。 |
| [fusionnext/models/backbones/fusionnext_backbone.py](/home/dataset-local/lr/code/fusionnext/fusionnext/models/backbones/fusionnext_backbone.py) | 看排序后有没有被正确送进 block，以及有没有按逆索引恢复。 |

## 一句话记忆版

如果你只想先抓住主线，可以这样记：

- `configs/`：训练菜单
- `tools/train.py`：启动按钮
- `datasets/`：把磁盘数据读出来
- `tokenizers/`：把图像和点云变成 token
- `serialization/geometry_serializer.py`：做跨模态 1D 排序
- `backbones/`：多层重复融合
- `layers/flash_window_block.py`：排序后做 attention
- `dense_heads/`：输出 3D 检测结果
