---
title: OPD里的token-level的无偏估计
date: 2026-05-09 10:00:00
tags: LLM, OPD
layout: post
mathjax: true
---

2026年以来，On-Policy Distillation又进入了人们的视野中。在此之前，后训练大概只有两种手段，一个是SFT，一个是RL。笔者最近看了一篇论文Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe，论文中提到了On-Policy Distillation。于是本文也分析一下OPD中的一个概念，即token-level reverse KL。 

首先，OPD是一种知识蒸馏，就是让一个模型学习另一个模型的输出行为。接受训练的模型叫作学生模型（student），提供指导的模型叫作教师模型（teacher）。训练时，可以比较两个模型给候选 token 分配的概率，再调整学生模型，使它的分布接近教师模型。跟常见的SFT不同，OPD先让 student 按照自己当前的概率分布逐步生成token，再让 teacher 根据 student 输出的上下文输出token概率并进行引导。

一般情况下，我们都认为教师模型是冻结的，另外一个模型（学术模型）最后都会倾向于拟合冻结模型。

但是**前向KL会尽可能同时拟合多个峰，反向KL倾向于拟合单个峰**。

设教师分布为 $q$，学生分布为 $p$。前向 KL 为 $D_{\mathrm{KL}}(q\|p)=\sum_x q(x)\log\frac{q(x)}{p(x)}$。

若某处 $q(x)>0$ 而 $p(x)\to 0$，则 $\log\frac{q(x)}{p(x)}\to+\infty$，该项被 $q(x)>0$ 加权，KL 发散。所以学生必须在该处给正概率，即教师有概率的地方学生不敢给零。反向 KL $D_{\mathrm{KL}}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}$ 的权重是 $p(x)$，学生不给概率的地方不惩罚，只惩罚学生有概率、教师没概率的区域。




直观来说，是因为在工业界OPD通常用于合版，也就是将多个teacher的能力合并起来，而训练的流程正是多阶段的OPD，此时让学生学习多峰就没必要，因为它们是学习多个拥有专长的老师而不是学习一个全才的老师。这样最小化 $D_{\mathrm{KL}}(p_{\text{student}}\|q_{\text{teacher}})$ 时，**教师固定，更新学生参数，让学生分布依次向多个教师分布靠近**而不会遗忘原有的知识。

同时，因为 OPD 使用**学生自己生成的回答**训练，Reverse KL 正好按学生分布计算期望：

$$
D_{\mathrm{KL}}(p\|q)
=\mathbb E_{y\sim p}\left[\log p(y)-\log q(y)\right].
$$


回到这篇论文，它区分了OPD的三种监督粒度（sampled-token，全词表，top-k）。其中**Sampled-token OPD**，是只看 student 实际采样出来的那个 token。它的单步损失是 student 对该 token 的 log 概率减去 teacher 输出的 log 概率。论文指出，这是一种 token-level reverse KL 的无偏单样本估计，也是较轻量、较常见的实现。

无偏是指随机估计量在重复采样很多次之后，平均值等于它想估计的真实量。在多次采样求和的情况下，Sampled-token OPD正是全词表的无偏估计。

Sampled-token OPD和全词表两者针对的是**同一个位置、同一个前缀下，学生与教师的下一词分布差异**，区别在计算方式。

- **全词表监督**：遍历所有候选词，精确计算该位置的反向 KL：$ K=\sum_v p(v)\log\frac{p(v)}{q(v)} $ 一次就利用所有候选词的概率信息，监督更密，开销更大。
- **Sampled-token**：从学生分布抽一个词 $y\sim p$，只用它的 $\ell=\log\frac{p(y)}{q(y)}$作为估计。因为 $\mathbb E[\ell]=K$，长期平均对应同一个 KL，但单次噪声更大，所需教师概率信息更少。

另外，全词表也只监督学生走到的前缀，并不遍历所有可能序列。

## 1. 有偏估计/无偏估计的定义

设 $A$ 为待计算的目标量。由于直接计算的成本较高，采用随机估计量 $\hat{A}$ 对其进行估计。

如果满足：
$$
\mathbb{E}[\hat{A}] = A
$$
就叫 **无偏估计**，否则称为 **有偏估计**。



## 2. 本文的OPD指什么

在 OPD 中，给定输入 $x$ 和学生已经生成的前缀 $\hat y_{<t}$，学生和教师分别预测第 $t$ 个 token，其概率分布记为：

$$
p_t(v)=\pi_\theta(v\mid x,\hat y_{<t}),
\qquad
q_t(v)=\pi_T(v\mid x,\hat y_{<t}).
$$

其中，$\pi_\theta$ 表示学生模型，$\pi_T$ 表示教师模型；$V$ 是完整词表，$v\in V$ 是其中任意一个候选 token。

“完整 KL”即全词表 KL：遍历所有候选 token，将学生概率乘以对数概率比，再求和：

$$
D_{\mathrm{KL}}(p_t\|q_t)
=
\sum_{v\in V}
p_t(v)
\left[\log p_t(v)-\log q_t(v)\right].
$$

这个公式按学生的概率 $p_t(v)$，对整个词表上的对数概率差加权求和。但是整个词表很大，完整计算太昂贵了。所以 sampled-token OPD 不看整个词表，只从 student 分布里采样一个 token：

$$
\hat{y}_t \sim p_t
$$

然后只算这个 token 上的 loss：

$$
\ell_t^{\mathrm{sample}}
= \log p_t(\hat{y}_t) - \log q_t(\hat{y}_t)
$$
为什么说它是无偏的？因为这个 token 本来就是按 $p_t$ 采样的，所以它的期望正好等于完整 KL：

$$
\mathbb{E}_{\hat{y}_t \sim p_t} [ \ell_t^{\mathrm{sample}} ]
= \sum_{v \in V} p_t(v) \left[ \log p_t(v) - \log q_t(v) \right]
= D_{\mathrm{KL}}(p_t \| q_t)
$$
论文说 sampled-token OPD 是 token-level reverse KL 的 unbiased single-sample estimator，指的就是这件事：**单次只看一个 token，但如果无限次重复按 student 分布采样并取平均，平均值会等于完整词表上的 reverse KL**。

## 3. 具体例子

假设词表只有两个 token：A 和 B。

student 分布是：
$$
p(A)=0.8,\quad p(B)=0.2
$$
teacher 分布是：
$$
q(A)=0.4,\quad q(B)=0.6
$$

完整 KL 是：


$$
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
&= 0.8[\log 0.8-\log 0.4] \\
&\quad + 0.2[\log 0.2-\log 0.6]
\end{aligned}
$$

也就是：

$$
D_{\mathrm{KL}}(p\|q) =
0.8\log 2
+
0.2\log \frac{1}{3}
$$
sampled-token OPD 怎么做？

它按 student 分布采样：

A 有 80% 概率被采到；

B 有 20% 概率被采到。

如果采到 A，loss 是：

$$
\log 0.8-\log 0.4 = \log 2
$$
如果采到 B，loss 是：

$$ \log 0.2-\log 0.6 = \log \frac{1}{3}$$

所以这个随机 loss 的期望是：
$$
0.8\log 2
+
0.2\log \frac{1}{3}
$$
刚好就是完整 KL。所以它是无偏的。

## 4. 混淆点

无偏不代表低方差。

sampled-token OPD 是无偏的，但因为每次只采一个 token，所以单次估计可能偏差很大。Full-vocabulary OPD 不采样，直接算完整 KL，所以更稳定，但代价更高。论文也正是这样区分 sampled-token OPD、full-vocabulary OPD 和 top-k OPD 的：sampled-token 轻量但采样噪声更大；full-vocabulary 梯度更密集但内存更贵；top-k 介于两者之间，是一种近似。

#### References:

[1]Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe https://arxiv.org/abs/2604.13016
[2]Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
