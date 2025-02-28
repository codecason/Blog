远程衰减；

Attention Mask；

窗口外注意力截断；

Sliding Window?

【大语言模型】LongLoRA:大语言模型长文本的高效微调方法 https://zhuanlan.zhihu.com/p/658067243

$$LoRA=W_0x+\triangle Wx$$

LoRA=W0+deltaW=W0+BA

A=(rxd+(r<d)+(B))

B=(dxr)

Attention的计算复杂度=O(n^2\*d)=如果把向量维度d 看作常数，则可以说self-*attention* 的*时间复杂度*是序列长度的平方

~~~
LongLoRA=swin-transfomer+shift short attention+mask


~~~

self-attention

