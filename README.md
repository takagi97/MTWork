 # 在transformer编码器解码器连接方式方面的探索
本项目主要设计以下三个模型，并对前两个进行了实验：
- Direct Connect Transformer : 编码器于解码器直连
- Full Connect Transformer : 将编码器各层加权平均送给解码器
- 未完成的模型 : 在前一模型基础上根据解码器层数分配不同权重
# Dependencies
- fairseq version = 0.6.2
- PyTorch version >= 1.5.0
- Python version >= 3.6
- For training new models, you'll also need an NVIDIA GPU
# Dataset
使用iwslt14-de-en数据集

可通过以下命令获取：  
```py 
# 项目根路径下
cd examples/translation
bash prepare-iwslt14.sh
```
# preprocess
数据集预处理可通过以下命令实现：
```py 
# 项目根路径下
bash preprocess.sh
```
# training
```sh
bash train.sh
```
注：
- Direct Connect Transformer：arch = stransformer_t2t_iwslt_de_en
- Full Connect Transformer: arch = dense_all_transformer_t2t_iwslt_de_en


# 模型结构
## 1、Direct Connect Transformer
将编码器各层和解码器各层直接相连。

## 2、Full Connect Transformer
将编码器各层经过加权平均送给解码器各层，加权平均会为编码器高层赋予更高权重。

## 3、未完成的模型
在前一模型的基础上根据解码器的层数分配不同的权重，送给解码器每层的信息不同。
让解码器头和尾得到更多编码器低层信息，解码器中间层获得更多编码器高层信息。

# 实验结果
## Direct Connect Transformer & Full Connect Transformer
#| Model |  De -> En |
--|--|--
1|Baseline|35.82
2|Gate_encoder_decoder|32.74
3|layer attention|32.87

注：baseline为transformer_t2t_iwslt_de_en


# 总结
在本文中，我凭借直觉对 Vanilla Transformer 模型进行了一些简单的修改，并且得到了一些推论。算是浅尝“炼金术”，不过在这个过程中熟悉了FairSeq代码以及整个训练流程，相信对未来研究有所帮助。另外，我会在之后时间尝试第三个模型。
详情请见[2001788-穆永誉-机器翻译报告](https://github.com/takagi97/MTWork/blob/master/2001788-%E7%A9%86%E6%B0%B8%E8%AA%89-%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91%E6%8A%A5%E5%91%8A.pdf)
