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

    
    
  - 什么是梯度消失和梯度爆炸？如何解决？

    梯度消失/爆炸都是模型在梯度下降中不稳定的表现

    解决思路：尝试不用的激活函数，或者使用梯度裁剪，正则化，LayerNormaliation，残差等方法在不同的场景下解决

    

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

    1，Layer Normailzation 是对单个批次的不同特征做  Normailzation；

    2，Bacth Normailzation 是对一组批次里面相同的特征做 Normailzation。

    
    
  - 如何解决RNN梯度爆炸和梯度消失的问题？

    传统RNN在反向传播时，因为模型架构问题，会造成误差累计，从而导致梯度消失/爆炸，因此出现LSTM，通过其内部"门"的设计能缓解该问题

    

- #### Transformer

  - Transformer 中的 Encoder 和 Decoder 是如何传递模型输出的？

    原论文中使用 Corss attention。

    

  - Transformer 中 Corss attention 的作用？

    Corss attention 将 Decoder 输入和 Encoder 输出做 attention 计算，得到 Decoder 的输出单元。

    

- #### 自注意力机制（Self-attention）

  - 为什么需要 Multi-head Self-attention？

    因为在模型输入的句子序列中，相同的词可能存在多种不同的相关性。

    

  - 为什么需要 Self-attention？

    解决序列元素之间相关性已经并行计算量的问题，可以进行	全部捕捉和并行计算。

    

  - Self-attention 如何解决输入序列顺序的问题？

    在 self-Self-attention 的输入数据上，给序列中每一个元素加上 Postional Encoding。

    

  - Self-attention 和 CNN 之间的区别？

    Self-attention 可以看作复杂版的 CNN。

    

  - Self-attention 和 RNN 之间的区别？

    1，Self-attention 考虑了整个了序列之后再输出，而 CNN 输出时只考虑左边的序列输入；

    2，Self-attention 可以并行计算，而 RNN 只能依次计算。

    

  - Self-attention 和 Masked Self-attention 的区别？

    Self-attention 输出时考虑了所有的输入信息，而 Masked Self-attention 输出时只能考虑序列前面出现的输入信息。

    

- #### BERT(Bidirectional Encoder Representation form Transformers)

  - BERT 的模型结构？

    BERT 采用 Transformers Encoder 的模型架构。

    

  - BERT 预训练方法？

    1，Masked Language Model(MLM) 采用15%概率用 mask token 随机的对输入的token进行替换，再在输出中预测被替换的token；

    2，Next Sentence Prediction(NSP) 输入两个句子，BERT 预测两个句子是否属于上下文关系。

    

  - 为什么BERT 会有用？

    因为 BERT 在预训练中得到了词关于上下文的向量表示，称为 Embedding。

    

- #### GPT(Generative Pre-Trained Transformer)

  - GPT 的模型结构？

    GPT 采用 Transformers Decoder 的模型架构。

    

  - GPT 预训练方法？

    Next Token Prediction(NTP) 词级自回归预测完整的句子。



面试可能被问到的问题

1，说一下Transformer把？

transformer 是来源一篇Google的论文，叫做Attention is all your need，

它论文里面描述的是用来做翻译任务的，它的模型结构是一个AutoEncoder的架构，它这么火主要是它提出了Attention的概念，通过词与词之间的注意力计算来获得上下文的语义表示，并且可以并行计算，而且效果比ELMO的模型更好，以及后续的BERT模型也是采用Transformer编码层的架构。

 

2，说一说CNN？

CNN 主要是应用在图片的领域上，它的模型架构在图片上做了一系列的前提设想，其中最主要优势就是它的卷积核的设计非常的切合图片中局部的像素特征，以多过多层的卷积操作之后就可以考虑到整个图片的信息，对于全连接层来说它的参数量更少，效果更好。

 

3，说一说LSTM？

LSTM是属于RNN的一种，它主要解决的是信息持久化的问题，通过模型内部的门记忆单元来实现，解决那些信息可以被传递到下个序列的输入，它的缺点就是无法处理非常长的序列，存在丢失信息的问题。

 

4，说一说Attention？

Attention的出现解决了LSTM序列过长会丢失信息的问题，而且Attention可以并行计算，Attention模型结构主要计算某个词与其他所有词的Attention分数当作对于序列的权重，根据权重就可以计算出某个词和序列中的其他词的相关性，Attention层的每一个输出都是考虑了整个序列的输出。

 

5，说一说ALBERT？

ALBERT 主要是BERT模型的进阶简化版，相对BERT参数量更少，所以对显存的要求会相对降低，而且训练更快，ALBERT 主要通过三种方式优化：

第一种就是因式分解的embedding，对词签入向量先降维再升维

第二种就是共享所有Attention层的参数

第三种就是把BERT的NSP任务变成SOP，比NSP任务更有效

 

6，说一说RoBERTa ？

RoBERTa 属于是BERT模型的改进版本，模型结构完全借用BERT架构，只是针对无监督预训练的方法做了改进，其中最大的一个改进是RoBERTa 采用动态Mask的设计，更能提升模型性能，第二个就是使用更多的训练资料，更大的batchsize

 

7，说一说Sentence-BERT？

Sentence-BERT的提出主要是因为BERT模型在相似度计算的任务上效果不好，而Sentence-BERT在模型架构和微调方式上都是针对相似度设计的，首先Sentence-BERT模型采用两个孪生的BERT模型，通过计算两个句子向量之间的相似度，并且通过交叉熵和均方差的loss 来微调模型

 

8，说一说 Key-BERT？

Key-BERT 主要是基于BERT的向量表示基础上， 通过 n-gram在整个文档中提取N个词或短语，然后通过计算余弦相似度来查找与文档最接近的词或短语描述

 

9，说一说 n-gram?

n-gram 是一种基于统计的算法，具体的算法思路就是将序列中的元素按照以N为大小的滑动窗口操作，可以得到多个长度为N的片段，	每一个片段就叫做gram，然后对所有的gram进行词频统计。

 

9，说一说Word2vec？

Word2vec 本质上就是Word embedding 的一种方式

Word2vec 是一种可以把单个词表示成高维密集向量的方法，这种方法缺点是无法考虑上下文的关系，在后来出现的Attention 模型解决了该问题 

 

9，说一说Elmo?

Elmo模型的提出，主要解决了针对word2Vec方法所表示的词向量没有办法考虑上下文的问题，Elmo的模型架构主要是基于双向的LSMT，当预测第N个词的时候，都是用长度为M的前后上下文去预测当前的token，当然它最主要的缺点是上下文长度的问题，考虑长的句子比较困难

 

10，Transformer的Decoder 和Encoder有什么本质的区别？

11，BERT的三个编码输入分别是什么？

13，BERT的预训练的任务是什么？

14，BERT的MASK采用的是什么策略？

15，ALBERT相对BERT主要做了哪些优化？

16，HMM和CRF有哪些区别？

17，CRF的两个矩阵分别是什么？

17，CRF最后的Viterbi算法原理是什么？

18，写一下Attention 的公式？

19，精确率，召回率，F1分数分别是什么？写一下它们的公式？

20，Sentence-BERT模型架构的原理是什么？

21，聚类算法的原理是什么？

22，LSTM的三个门控单元的原理是什么？

 
