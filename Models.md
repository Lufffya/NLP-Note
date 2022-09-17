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

    

- #### Transformer 的模型结构？

  - gAttention 的出现解决了LSTM序列过长会丢失信息的问题，而且Attention可以并行计算，Attention模型结构主要计算某个词与其他所有词的Attention分数当作对于序列的权重，根据权重就可以计算出某个词和序列中的其他词的相关性，Attention层的每一个输出都是考虑了整个序列的输出。

- #### Transformer 的模型结构？

  - gg

## Bidirectional Encoder Representations from Transformers（BERT）2018

- #### 概念

  - 



## Generative Pre-Training（GPT）2018

- #### 概念

  - 

