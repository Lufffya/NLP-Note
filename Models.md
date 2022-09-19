## Embeddings from Language Model（ELMO）2018

- #### 概念

  - ELMO 是一个预训练的语言模型

  - ELMO 使用 双向的 LSTM 的网络架构

  

- #### 预训练任务

  - 根据前后上下文来预测句子里面的其他单词

  - 正向的 LSTM 根据单词的上文，预测单词的下文；反向的 LSTM 根据单词的下文，预测单词的上文

    

- #### 优点

  - 相对于 Word2vec 来说，ELMO 产生的词向量能够考虑上下文的关系，而不是固定的词只有固定的向量



## Attention is all your need（Transformer）2017

- #### 概念

  - Attention 公式|Attention 机制
  
    
  
- #### Attention 的公式是什么？

  - $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d_K}})V
    $$

- #### Attention 公式中除以根号DK的作用是什么？

  - 根号DK是缩放因子，主要是将Attention QK的值做归一化，使得梯度保持相对稳定
  
    

- #### Attention 机制的原理是什么？

  - 对序列中不同单词给予不同的权重，把注意力集中在一条句子中重要的部分
  
    

- #### Attention 和 RNN 的区别？

  - Attention 解决了 RNN 中序列过长所导致梯度消失/爆炸以及信息丢失的问题，Attention 可以并行的将一个词和其他任何词做注意力计算，不受序列长度的影响
  
  - Attention 序列没有前后位置关系的信息，所以一般在使用的过程中会加上额外的位置编码信息
  
    

- #### Transformer 的模型结构？

  - 从模型结构上来说，Transformer 属于从 Encoder 到 Decoder 过程的自编码模型

  - 从输入和输出来说， Transformer 属于 seq2seq 的自回归模型

    

- #### Transformer 模型执行过程？

  - Encoder：首先 Transformer 接收被 token 为数字的句子作为输入，然后将句子Emdedding成指定维度的向量，再加上标记序列位置信息（Position Encoding ）的的编码，然后进入到 Encoder 层的 Attention block，每一个 Attention block 里面会进行 Attention 计算，残差，Norm，然后再过一个Feed forword Netword，主要是将 Attention 的输出使用 glue 激活函数做非线性变换，最后在过一个 残差 和 Norm，完成一个 Attention block 的操作，通常会堆叠N层

  - Decoder：在 Decoder 里，会使用开始标记符作为 Decoder 的第一个输入，通常是 CLS，然后进入到 Decoder Attention block，Decoder Attention 和 Encoder 大致一样，唯一不同的是 Encoder 里面多了一部 Mask Attention 的操作，也就是根据 Decoder 的输入和 Encoder 的输出做 Attention 计算，并且使用 Mask 掩码，使得 Decoder 只能看到当前之间的单词序列，最后使用自回归的方式预测剩余的序列，直到预测到结束标记符，预测结束

    

- #### Transformer Encoder 和 Decoder 有什么区别？

  - 唯一的区别在于 Decoder 中的 Attention block 中引入了一个 Cross Attention 的模块，和 self Attention 不同的是， Cross Attention 采用 MASK 机制，将 Decoder 输入作为 Q，从 Encoder 中抽取 K 和 W 做注意力计算

    

- #### Mulit-Head-Attention 有什么用？

  - 将同一个序列映射在不用的向量空间中计算注意力分数，关注一组序列中不用维度的关联关系，最后再将这些信息concat起来，就是 Mulit-Head 的输出
  
    

## Bidirectional Encoder Representations from Transformers（BERT）2018

- #### 概念

  - 输入|预训练|MASK 策略|对比 ALBERT
  
    
  
- #### BERT 的模型结构？

  - BERT 采用 Transformers Encoder 的模型架构
  
  
  
- #### BERT 的输入有哪些？

  - Token Embeddings，表示句子的 Token 向量输入

  - Segment Embeddings，区分两个句子对的关系，在NSP任务中会用到

  - Position Embeddings，标记每个序列的位置信息，因为 Attention 计算无法考虑序列顺序

    

- #### BERT 预训练任务是什么？

  - MLM（Masked Language Model）随机从输入序列中遮盖一部分的单词，希望模型能正确预测被遮盖的单词

  - NSP（Next Sentence Prediction）让模型预测两个句子是不是前后句关系

    

- #### BERT MASK的策略是什么？

  - 在序列中随机MASK掉15%的Token，在15%中，有10%的几率会被替换成其他单词，10%则保持原单词不变，80%会被标记替换为[MASK]

    
  
- #### ALBERT 相对于 BERT 有哪些不同？

  - LABERT 是轻量级的 BERT 以及改进版，主要通过三种方法实现
  - 因式分解（Factorized Embedding Parameterization）将 Embedding 的矩阵做因式分解，先降维再升维到和隐含层的的维度一至
  - 权重共享（Cross-layer Parameter Sharing）
  - Sentence Order Prediction（SOP）

## Generative Pre-Training（GPT）2018

- #### 概念

  - 

