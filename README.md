# 鲁迅说没说

## 前言

项目的灵感来自：
https://www.bilibili.com/video/BV1ZmBQYjEea

然后下载了对应的项目
https://github.com/BushJiang/LuXunWorks

项目基于 Milvus 数据库和基于网络的LLM Api，项目有点重，不适合本地部署。

所以基于 c# 和 Ollama 做了一下重写。
在这里感谢源项目已经整理好的data数据，以及提示词。

## 项目说明

1. Ollama 使用了2个模型，embedding 模型用 bge-m3，LLM模型用 chatglm4，没有使用向量数据库，直接对Embedding后的向量进行距离计算。
2. 项目开始前需要先将所有数据做一次embedding，因为不使用向量数据库，因此数据都保存到本地文件，并且文件采用msgpack格式保存。
3. 整个 embedding 过程大概需要 6 小时（基于 2080ti），最终的文件有 2G 大小。
    3.1 embedding的代码有2个，一个基于python，一个是c#项目里的。python 出来的文件会更大一些，因为embedding之后的向量是 double 保存的，而c#是 float 。

