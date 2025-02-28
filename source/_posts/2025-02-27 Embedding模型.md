---
title: Embedding模型笔记
date: 2025-02-27 10:00:00
tags: LLM, 对比学习
layout: post
mathjax: true
---

#### 问题
1.来到2024，文本向量化的SOTA模型都有哪些，各自的特点为何，leading board ?
2.对于汉语增强的，国内提供的汉语为主及多语言embedding模型都有哪些?
3.Embedding 模型的训练数据都有哪些?OpenAI的方法，以及这篇论文的方法的区别和联系在哪里?https://instructor-embedding.github.io/， 显然你需要找到提到的两篇论文原文进行对比阅读再分析相关评述和引文


1.
MTEB（Massive Text Embedding Benchmark）是一堆衡量文本嵌入模型的评估指标合集，与之对应的中文评估指标是 C-MTEB （flag-embedding开源项目）。

NV-Embed-v2 (Rank 1)提出了几项新设计，包括让 LLM 关注潜在向量以获得更好的池化嵌入输出，并展示了一种两阶段指令调整方法，以提高检索和非检索任务的准确性。 此外，NV-Embed-v2 还采用了一种新颖的硬负例挖掘方法，这种方法考虑了正相关性得分，能更好地去除假阴性。

gte-Qwen2-7B-instruct (Rank 5)该模型融合了几项关键的进步：
集成了双向注意力机制，丰富了其上下文理解能力。
指令微调，仅应用于查询侧，以提高效率。
在一个庞大的多语言文本语料库上进行全面训练，涵盖了不同的领域和场景。这种训练利用了弱监督和监督数据，确保了模型在多种语言和广泛的下游任务中的适用性。

Linq-Embed-Mistral (Rank 12)专注于使用先进的数据精炼方法来改进文本检索，包括复杂的数据制作、数据过滤以及由教师模型指导的负例挖掘，这些都是为每个任务高度定制的，以提高由大型语言模型（LLM）生成的合成数据的质量。

其他的榜单前10要么是没有说明创新性（根据其他模型的基座微调），要么是同一个主要模型的其他版本，所以没有列举。
榜单来源为：https://huggingface.co/spaces/mteb/leaderboard

2.
Conan-embedding, 由腾讯发布，来自论文 Conan-embedding: General Text Embedding with More and
Better Negative Samples，在C-MTEB上排名第一。该模型旨在通过最大化负例的质量和数量来增强嵌入模型的性能，有两个主要创新：动态硬负例挖掘和跨GPU平衡损失。
xiaobu-embedding-v2，来自论文Circle loss: A unified perspective of pair similarity optimization。
gte-Qwen2-7B-instruct，来自阿里，如问题1所陈述。

3.
常用的Embedding模型的训练数据是多样化的，主要是文本对，包括标题-内容对、输入-输出对以及问题-答案对(Shiyu Li et.al, Conan-embedding)。而在 https://instructor-embedding.github.io/ 里所指的方法中，构建了MEDI数据集，包含了300余个数据集的数据，包含了Super-NI，它是是一个包含1616个NLP任务及其自然语言指令的大型基准，76种广泛的任务类型，涵盖55种不同的语言。

2024年1月，OpenAI发表的关于产生Embedding的方法：https://openai.com/index/new-embedding-models-and-api-updates/。
Paper方法：

OpenAI的训练方法体现在2022年的论文《Text and Code Embeddings by Contrastive Pre-Training》，第二篇论文则是《One Embedder, Any Task: Instruction-Finetuned Text Embeddings》。它们主要探讨了如何生成用于不同下游任务的文本嵌入，但是有一些区别。
区别：
首先，嵌入方法的目的和方法上，OpenAI的方法主要关注于通过对比预训练（contrastive pre-training）来生成文本和代码的嵌入。这种方法不依赖于监督数据，而是使用大规模的未标记数据来训练模型。
而第二篇论文介绍了INSTRUCTOR，这是一个单一的嵌入模型，它可以根据任务指令生成针对不同下游任务和领域的文本嵌入，而无需进一步的训练。INSTRUCTOR通过结合任务指令和文本输入来生成特定的嵌入，这使得它能够适应多种任务。
其次，在训练策略上，第一篇论文中的方法在多种数据规模上训练，依赖于对比学习目标，使用批次内负样本（in-batch negatives）来训练模型，并且强调了大批量大小对于实现最佳性能的重要性。而第二篇论文中的INSTRUCTOR模型则是在多任务数据集上进行训练，这些数据集包含了人类编写的任务指令（MEDI数据集）。INSTRUCTOR使用对比损失函数来训练，该函数最大化语义相关文本对之间的相似性，同时最小化不相关对之间的相似性。
在最终的效果上，第二篇论文的INSTRUCTOR模型则被设计为更通用，能够适应多种不同的任务和领域，这主要得益于其在训练期间特意针对任务指令训练数据的构造。

联系：
两篇论文都使用了对比学习这种自监督的方法来训练他们的模型，也就是说，它们均通过区分正负样本对来学习数据的有效表示。
两篇文章均对不同数据集有一定的适应性，但是前者主要是针对不同数据规模，且适应的任务主要在检索和线性探测分类上。而第二篇论文则通过在训练期间引入任务指令来实现，训练数据更加多样，能够适应更多的任务类型。

#### References

[1]MTEB指标包含些啥、啥和那啥 https://zhuanlan.zhihu.com/p/676495018
[2]Xiao, Shitao, et al. C-pack: Packaged resources to advance general chinese embedding. arXiv preprint arXiv:2309.07597 (2023).
[3]MTEB Leaderboard https://huggingface.co/spaces/mteb/leaderboard
[4]Sun, Yifan, et al. Circle loss: A unified perspective of pair similarity optimization. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
[5]https://www.datacamp.com/tutorial/exploring-text-embedding-3-large-new-openai-embeddings
[6]Text and Code Embeddings by Contrastive Pre-Training, https://arxiv.org/pdf/2201.10005