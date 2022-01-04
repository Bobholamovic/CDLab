# 设计思想

本目录中包含一些关于项目设计思想的文档。

## 代码布局

+ `configs`: 配置文件存放位置
+ `docs`: 项目文档
+ `exp`: 默认实验目录
+ `scripts`: 包含一些有用的脚本（如数据集预处理脚本）
+ `src`:
    - `core`: 项目核心功能实现，与其它部分相隔离，可单独维护
        - `builders.py`: 包含一些基础的建造器
        - `config.py`: 包含配置系统的实现
        - `data.py`: 定义数据集抽象基类
        - `factories.py`: 包含工厂的实现
        - `misc.py`: 包含 `Logger` 以及 `OutPathGetter` 的实现
        - `trainer.py`: 包含训练器基类和`TrainerSwitcher`的定义
    - `data`: 数据集
        - `__init__.py`: 定义变化检测数据集的抽象基类
    - `impl`: 包含对 `core` 中规范的建造器和训练器的具体实现
        - `builders`: 各种建造器实现
        - `trainers`: 各种训练器实现
            - `__init__.py`: 包含一个`TrainerSwitcher`对象，在其中添加选择训练器的规则
    - `models`:
        - `_blocks.py`: 基础模块实现，对 PyTorch 提供的底层模块的再封装
        - `_common.py`: 一些通用的高层模块
    - `utils`:
        - `data_utils`: 与数据读写、预处理、后处理等相关的辅助函数和辅助类
            - `augmentations.py`: 包含数据增强类
            - `preprocessors.py`: 包含数据预处理类
        - `losses.py`: 定义一些损失
        - `metrics.py`: 包含常用的评价指标
        - `utils.py`: 其它辅助函数和辅助类
    - `constants.py`：定义一些全局常量
    - `infer.py`: 前向推理脚本（尚未实现）
    - `sw_test.py`: 滑窗测试脚本
    - `train.py`: 训练/评估模型的主入口
+ `tests`: 测试相关

### 项目特色

- 提倡自由但也施加约束，减少 hook 的使用，压缩嵌套层级，权衡二次开发难度和调试难度；
- 三级配置系统，不同层级之间可复写，适应不同场合、不同需要；
- 利用面向对象特性和 Python 语言特性，提供一些具有特色的抽象类、接口和工具。

## 目录

- [总体流程](./总体流程.md)
- [工厂与建造器](./工厂与建造器.md)
- [训练器](./训练器.md)
- [配置系统](./配置系统.md)
- [数据集接口](./数据集接口.md)
- [数据预处理](./数据预处理.md)
- [其它](./其它.md)