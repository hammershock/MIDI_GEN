## 数据集下载：
https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
下载之后解压缩到./data目录下

源数据集地址：[MAESTRO 数据集地址](https://magenta.tensorflow.org/datasets/maestro)

## 设计思路：
一种符号音乐生成模型（可以采用GPT架构）

### 引言：
符号音乐，由若干个音符组合而成，每个音符包含：起始和终止时间start和end，音高pitch，强弱velocity四个属性，可以将其看作符号token，用于生成任务

对于符号音乐的token表示方式，有以下三种方案：
1. 基于流程控制（对于每一个音符，都使用起始和终止符号）
2. 使用duration
3. 使用钢琴卷帘

### 与文本生成的不同点：
但是符号音乐生成，不同于文本生成，存在以下挑战：
1. 多音轨（可能很多种类乐器，不同音符可能同时响起）
2. 多维度：一个音符有多种属性

我们只考虑钢琴这一种乐器，但是依然存在同一段时间很多音符响起的问题
如何解决？
1. 对音符进行组合编码。使用基于BPE的子词聚合算法，但是不能很好地处理每个音符的独立属性
2. 

如何表示音符？
1. 嵌入时间信息，正余弦位置编码
2. 嵌入音高，考虑到音高的周期性，以12为周期（12平均律），正余弦位置编码
3. 强弱信息，0-1连续实值嵌入

如何解码音符？
使用多头分类器，分别输出音高pitch，强弱velocity，持续时间duration，距离上一个音符的偏移量offset
- 音高使用离散分类器
- 强弱输出层不采用激活函数，clip到-1到1
- duration离散化，使用离散分类器
- offset离散化，使用分类器

### 实现流程
1. 预处理：
    切割序列，得到输入表示
2. 模型结构
- GPT架构
- 词嵌入维度：64
- 层数：6
- pad长度128
- token个数：（多维度，pitch128,offset和duration暂定）
- 对于duration, offset和pitch输出层，添加HMM模型

## 已完成:
1. 统计其中第一个音乐的音频统计特征，绘制图表


## TODO&成员分工:
1. 需求分析
2. 概要设计（5月28日前）
3. 使用GPT架构的一个demo
4. 实现一个前端，以供展示音乐生成
