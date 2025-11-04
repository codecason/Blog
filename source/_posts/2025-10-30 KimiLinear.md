---
title: Kimi-Linear论文解读--线性注意力混合架构
date: 2025-11-04 10:00:00
tags: LLM, 对比学习
layout: post
mathjax: true
---



#### 1.背景知识

​	最近Kimi发布了Kimi-Linear的技术报告，又重新把开源社区的注意力拉到了线性注意力来。事实上之前比较大的线性注意力模型是MiniMax发布的MiniMax-01，用了7：1的Lightning Attention和Softmax Attention。

​	但也正是MiniMax后来不再死磕线性注意力，转而继续使用传统的Softmax Attention了。而这次Kimi-Linear其实采用的也不是完全的线性注意力，而是一个混合的架构，其中线性注意力的部分是一个叫做KDA的结构，正是此次发布的核心。显然从这里，我们可以一窥其门径，目前的架构中都没有完全放弃全注意力，这是因为所谓的线性注意力有天然的劣势。因为这种劣势，现在的线性注意力架构都在加各种补丁。

#### 2.前置知识

##### 1.线性注意力机制

首先是线性注意力机制的来源和动机。大家都知道，因为最经典的Attention机制，其计算复杂度是 O(n^2)，其中 *n* 是序列长度，当序列很长时，计算量和内存需求会急剧增加，难以处理超长序列和大规模数据。当然，其实有人从另外一个并行算法的角度进行了分析，说它其实本质上可以化成O(nlog n)，但这是建立在抽象的无穷多GPU的理论前提下的，而目前也不太可能出现对应的GPU架构，所以意义不大。详情可见 https://supaiku.com/attention-is-logarithmic

Kimi-Linear正以此引入，注意到线性注意力机制的计算复杂度低，但是性能比softmax注意力机制要弱，即使在短序列也是如此。而近期（2025年）的进展通过两个方式解决这个问题，一个是门控机制+长程衰减，一个是Delta Rule。经过修正，使用这些模块的LLM在中等长度上显现出了较好的效果。

但即使如此，在超长的序列和上下文检索上都在理论上都显得很有挑战性。

对agentic任务（即需要大量耗费资源的环境交互操作）和重解码的LLM也是如此。

##### 2.从DeltaNet 到 Gated DeltaNet (GDN)

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t^\top \mathbf{q}_t.
$$

从fast-weight 的角度看（什么是fast-weight?)，$$S_t$$作为一个关联矩阵记录了从K到V的映射。


$$
\mathcal{L}_t(\mathbf{S}) = -\langle \mathbf{S}^\top \mathbf{k}_t, \mathbf{v}_t \rangle,
$$


根据经典的梯度下降方法，以Reconstruction损失为目标，递推公式为

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \nabla_{\mathbf{S}} \mathcal{L}_t(\mathbf{S}_{t-1}) = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top.
$$



此时对上一个状态加上一个遗忘率$$\alpha_t$$，得到
$$
\mathbf{S}_t = \alpha_t (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top.
$$

这就是**Gated DeltaNet(GDN)**。

这里的思想是，虽然DeltaNet能够保留长期记忆，但是我们现在又嫌它记得太多了，说当文本太长的时候我们不希望它把记忆无节制地保留下来。


Gated DeltaNet(GDN)的提出，旨在从**计算理论**和**在线学习**的视角，为线性注意力提供一个更具原则性的记忆更新范式。它融合了两个关键思想：

1. **Delta规则**：将记忆更新视为一个在线最小二乘优化过程。
2. **门控机制**：引入数据依赖的遗忘门，以调控记忆的生命周期。



加上了$$\alpha_t$$之后则代表了对以往的记忆有衰减率。



#### 3.论文的架构

##### 3.1 符号说明：

$$
\square_t \in \mathbb{R}^{d_k} \text{ or } \mathbb{R}^{d_v}, \text{ s.t., } \square \in {q, k, v, o, u, w}
$$

下三角矩阵：$$M,M^-$$

把每个输入序列分chunk，每个token对应一个embedding向量，所以一个chunk里就是多个embedding。

同样地，状态矩阵也分chunk。（这里的状态矩阵是RNN里常用的概念，简单的理解就是一个内部隐状态的矩阵。）

我们定义累积衰减 $$\gamma_{[t]}^{i \rightarrow j} := \prod_{k=i}^{j} \alpha_{[t]}^{k}$$，并且将 $$\gamma_{[t]}^{1 \rightarrow r}$$简写为 $$\gamma_{[t]}^{r}$$。

##### 3.2 KDA

KDA（Kimi Delta Attention）在GDN的公式上又修改了遗忘率的计算。

![image-20251104151712076](../../../../images/kimi-linear\fig1.png)
$$
\mathbf{S}_t = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \text{Diag}(\alpha_t) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top
$$

看到这里我们应该明白了，跟以往的模型不同的是KDA遗忘门的粒度精细到了token的维度。你说没有人这么想过吗？那肯定不是，但是为什么其他人不去做这个呢？首先是这样会有额外的开销，其次现在大模型的训练是算力和人力密集的，要验证一个架构的有效性需要消耗大量的算力资源。

##### 3.3 Kimi-Linear

​	Kimi-Linear是在Gated DeltaNet上扩展得到的线性注意力机制，一个混合的Linear注意力架构。

​	Kimi Linear 以 3:1 的固定比例将 KDA 与周期性全局注意力层交错排列。

​	（看到这里很难不对这两个词"周期性"和"交错排列"感到疑惑，没事且听本节的分解）

**分块矩阵的计算**

![image-20251104152514401](../../../../images/kimi-linear\fig2.png)

![image-20251104152240397](../../../../images/kimi-linear\fig3.png)

对于每个序列，Kimi-Linear进行了分块。$$_{[t]}^r$$ 代表的是第t个chunk的第r个元素，事实上跟一个token（的embedding)对应；而上面的式子是对递推公式进行展开之后得到的，最终得到一个跟初始状态值和对角线矩阵相关的系数；上式的第一项和第二项分别定义了 $$P_{[t]}^r$$ 和 $$H_{[t]}^r$$ ，揭示了每个元素的状态可以直接从块内的原始状态直接计算而无需递归。

为了高效地计算$$S_{[t]}^r$$，就是将这些序列依赖的递归计算，转化为一个**可以并行求解的线性系统**，要对以上的形式进行化简。而经过附录里的一大坨推导之后，这两个系数可以转成另外一个形式，这个形式也叫做WY表示形式。



经过UT Transform的代换，$$S_{[t+1]}$$可以写成跟$$S_{[t]}$$的递推形式

![image](../../../../images/kimi-linear\fig5.png)

最终启用了块内并行，块间递归，这是最直接的收益。没有这个变换，KDA就无法在训练时以分块的方式进行并行计算，其训练速度将与传统RNN一样缓慢，训练效率大打折扣。

​	Kimi-Linear的架构图如下所示。可以看到除了KDA之外还有一些MLA和MoE的层。作者通过实验确定了三层KDA搭配一层MLA的长文本在效率和质量上的平衡是最好的。以固定的、周期性的间隔，插入标准的、具有全局感受野的 Softmax 注意力层，这就是**周期性**和**交错排列**的含义。



![image](../../../../images/kimi-linear\fig6.png)



##### 3.4 消融实验与NoPE

Kimi Linear做了一个非常反直觉的设计：在所有的MLA层中，彻底移除了位置编码（NoPE）。

- KDA内部的精细遗忘机制和Delta规则，本身就形成了一种**数据驱动的、动态的位置感知系统**。它通过衰减来控制信息的新旧，这本身就是一种位置信号。如果全注意力层也保留RoPE，它会携带一个非常强烈且固定的位置偏置。这会导致模型在短文本上表现过好，但在需要外推到长文本时，两种不同的位置机制（RoPE和KDA）会产生冲突，降低模型的适应性和鲁棒性。

KDA层承担了大部分繁重的序列建模工作，以其线性复杂度保证了效率。而MLA层，则像全局信息的中继站，确保了模型不会丢失关键的远程依赖关系。当然，作者在文中只是说了这个冲突的结果是RoPE会对短文本有利，而对长文本不利，因此移除了位置编码。实验结果也支持这个分析。

##### 3.5 Kimi-Linear的强大的解码优势

预填充阶段，在序列长度超过128k后，Kimi Linear的预填充速度开始显著优于全注意力MLA，并且在512k和1M长度时，分别达到**2.3倍和2.9倍**的加速。而在解码（生成）阶段，由于KV缓存的大幅减少，Kimi Linear在生成每个token时的耗时（TPOT）几乎不随上下文长度增长而显著增加。在1M的长度下，解码吞吐量达到了全注意力的6倍以上。

#### 4. 其他细节

##### 4.1.Scaling Law

测试了MLA和KDA的配比；对比了纯MLA架构；

##### 4.2 长上下文的评测

在128k上下文长度的综合评测RULER上，Kimi Linear取得了**84.3分**的优异成绩，显著高于MLA的81.3分和GDN-H的80.5分。在代码库理解基准RepoQA上，也以68.5分领先。

##### 4.3 RL性能评测

文中提到了PTX loss，一个RL微调的正则化方法，在RL训练的时候混入预训练数据。

##### 4.4 评估框架

文中的评估框架其实有点意思，叫作LM-Harness-Evaluation，据说是统一了评估标准的测评框架，确保相同的模型在不同的环境下，按照相同的规则（如few-shot示例、格式、度量标准）进行评估。

#### 5. RWKV的争议

​	因为RWKV在线性注意力机制里有很多的工作，所以只要是线性注意力（或者类Mamba的架构）总是会引来相关人员对架构原创性的争论。既然如此，我们不妨来看看Kimi-Linear的KDA跟RWKV-7有什么区别。

​	KDA采用了一种特殊的**对角加低秩（DPLR）** 矩阵来参数化其状态转移，并为此量身定制了分块并行算法。相比通用的DPLR实现，KDA通过巧妙的参数绑定（把a和b都绑定到k上），**计算量大幅减少，内核速度提升了近100%**。

​	在下面这个表里第一列是递归形式，第二列是并行形式，可以看到RWKV和KDA的第二列几乎是一摸一样的。显然为了避免口水之争，作者先提前做了准备，而解释的内容正是文中的6.2节。

​	

![image](../../../../images/kimi-linear\fig7.png)

​	St的递归公式可以写作

![image](../../../../images/kimi-linear\fig8.png)

​	也即（下面这个$$k_t v_t^T$$少了个β_t，是typo？)：

![kda-equation](../../../../images/kimi-linear\kda-equation.png)

​	而RWKV-7的公式里$$b_t$$是可变的，这就是一切区别的根源。这两种方式都是DPLR，即一个矩阵可以被分解为**一个对角矩阵**加上**一个（或多个）低秩矩阵**的和。在经由固定了$$a_t$$和$$b_t$$之后，KDA消除了大约三次矩阵乘法，使得计算效率明显更高。

​	怎么说呢，这个说法虽然证明了确实有区别，但是实际上两者确实非常相像，而且新的架构也像工程上的优化。而RWKV-7是在前面发表的，说是受到它的启发也完全没问题。只能说在当前的LLM中趋同的研究其实非常多，打口水仗真的很没有意义。本人觉得可以给类似的辩论都下一个武断的结论，大部分改进都没有彻底的创新点，但是也不能算抄袭，本质上只是在抢占话语权，对于一般开发者而言没有必要去关心。



#### 开源代码实现

https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda

#### 参考文献

[1]Kimi Linear: An Expressive, Efficient Attention Architecture  https://arxiv.org/abs/2510.26692

[2]Gated Delta Networks: Improving Mamba2 with Delta Rule https://arxiv.org/abs/2412.06464

[3]State Space Duality (Mamba-2) Part I - The Model https://tridao.me/blog/2024/mamba2-part1-model/
