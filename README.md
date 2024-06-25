# 代码仓库链接：
可从[百度云](https://pan.baidu.com/s/1_uxK0yoR0nU8bvBJAMcnhA)下载仓库代码, 提取码：nust
## 数据集
请先下载NUS-WIDE数据集，解压后放在ImageData目录下。NUS-WIDE下载链接：[NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)
经过处理后的数据应该是如下形式:
```
    ├── NJUST知识增强跨模态语义分析代码库
    │   ├── ImageData
    │   │   ├── NUS-WIDE
```
## 训练
### 训练无知识增强模型
```sh
python train.py
```
### 训练基于知识增强预训练模型的跨模态语义分析模型
```sh
python train_with_know.py
```
## 测试
可从[百度云](https://pan.baidu.com/s/1hz9tSUtVJy6cHXED5_Tnig),提取码：nust, 下载训练好的模型。
经过处理后的数据应该是如下形式:
```
    ├── NJUST知识增强跨模态语义分析代码库
    │   ├── model.t7
    |   ├── model_with_knowledge.t7
```
运行下述代码可得到基础模型和使用知识增强预训练模型的测试结果
```sh
python inference.py
```
