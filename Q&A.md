# Q&A

- #### 概念

  - Autoregressive(AT) 和 Non-autoregressive(NAT) 的区别？

    1，AT 是自回归每次将上一次输出加到下一次输入中再输出序列；

    2，NAT 是平行化计算，可以一次性输出所有的序列。

    

  - Non-autoregressive(NAT)  如何一次性输入所有的序列？

    1，另外训练一个可以输出序列长度的模型，然后输入N个和长度一样的起始符到NAT模型中；

    2，认为确定一个超参数的序列长度，然后输入N个和长度一样的起始符到NAT模型中，取输出序列中截至符左边的输入作为最终的结果。

    

  - Sequence to sequence(seq2seq) 模型如何训练？

    使用 Encoder 和 Decoder 结构进行训练，Encoder 接收训练数据，Decoder 输入和输出接收错位的标注数据。

    

- #### 激活函数（Activation）

  - Soft-max 的作用？

    将上一层的输出做 Normalize 归一化到0~1之间，方便计算相似度。

    

- #### 损失函数（Loss）

  - 为什么 Cross-entropy 比 Mean Square Error(MSE)  更适合做分类问题？

    使用MSE可能会遇到梯度消失很难收敛的情况。
    
    

- #### 模型层（Layer）

  - 在深度学习中 Normailzation 的用作？

    用于将模型输入特征做标准化，使得特征处于相同的计量范围，有利于梯度的计算。

    

  - 当使用 Bacth Normailzation 在 Testing/Inference 时，如何处理使用不用于 Training 时的 Btach 大小而遇到 mu(μ) 与 sigma(σ) 计算问题？

    使用 Traing 时所计算的 μ 和 σ 做移动平均计算来代替 Testing/Inferenc 时 Batch Normailzationz 中的 μ 和 σ 。

    

  - Layer Normailzation 和 Bacth Normailzation 的区别？

    1，Layer Normailzation 是对不同维度的同一组 Feature 做 Normailzation；

    2，Bacth Normailzation 是对同一维度的不同 Feature 做 Normailzation。

    

- #### Transformer

  - Transformer 中的 Encoder 和 Decoder 是如何传递模型输出的？

    原论文中使用 Corss attention。

    

  - Transformer 中 Corss attention 的作用？

    Corss attention 将 Decoder 输入和 Encoder 输出做 attention 计算，得到 Decoder 的输出单元。

    

- #### 自注意力机制（Self-attention）

  - 为什么需要 Multi-head Self-attention？

    因为在模型输入的句子序列中，相同的词可能存在多种不同的相关性。

    

  - 为什么需要 Self-attention？

    解决序列元素之间相关性已经并行计算量的问题，可以进行全部捕捉和并行计算。

    

  - Self-attention 如何解决输入序列顺序的问题？

    在 self-Self-attention 的输入数据上，给序列中每一个元素加上 Postional Encoding。

    

  - Self-attention 和 CNN 之间的区别？

    Self-attention 可以看作复杂版的 CNN。

    

  - Self-attention 和 RNN 之间的区别？

    1，Self-attention 考虑了整个了序列之后再输出，而 CNN 输出时只考虑左边的序列输入；

    2，Self-attention 可以并行计算，而 RNN 只能依次计算。

    

  - Self-attention 和 Masked Self-attention 的区别？

    Self-attention 输出时考虑了所有的输入信息，而 Masked Self-attention 输出时只能考虑序列前面出现的输入信息。
