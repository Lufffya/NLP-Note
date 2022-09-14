## Feedforward Neural Network（FNN）

- #### 概念

  - FNN 是前馈神经网络，每个神经元只和前一层的神经元之间相互连接，接收前一层输出，同一层之间的神经元彼此之间没有关联
  
    

## Convolutional Neural Network（CNN）

- #### 概念

  - CNN 是卷积神经网络，本质上是一个简化版的全连接前馈神经网络
  
  - 在网络结构中设置了对于图像处理的先验知识，比如，filter的设计可以对图像进行降维，提取一张图像中多个相同特征片段
  
    
  
- #### CNN 通过卷积计算后输出的维度大小是如何计算的？

  - $$
    height = (height_{in} - height_{kernel} + 2 * padding) / stride + 1
    $$

    

## Recurrent Neural Network（RNN）

- #### 概念

  - 梯度消失/爆炸问题|门控单元问题
  
  - RNN 是循环神经网络，是一个从序列到序列的神经网络
  
    
  
- #### 为什么 RNN 会导致梯度消失/爆炸？

  - 如果 RNN 的输入序列过长，就可能导致梯度消失/爆炸，在训练期间进行梯度计算时，过长的序列会导致权重出现指数增长或衰减
  
    
  
- #### 如何解决梯度消失/爆炸的问题？

  - 可以进行梯度截断，正则化，归一化，残差或使用更不易于造成梯度消失/爆炸的 LSTM
  
    
  
- #### LSTM 中有哪些门控单元？原理是什么？

  - LSTM 中分别有 input gate, forget gate, output gate，实质为三个向量乘以权重矩阵后，经过 sigmoid 激活函数转换为 0-1 之间数值来作为一种门控状态
  -  input gate 决定上个时刻的输出是否要输入到当前时刻
  - forget gate 决定当前时刻的输入是否要保存到memory
  - output gate 决定当前时刻的memory单元存储的值是否要输出到下一时刻
  
  

## Hidden Makov Model（HMM）

- #### 概念

  - HMM|Viterbi|HMM 和 CRF 的区别
  
    
  
- #### HMM 的原理？

  - HMM 里面分别构建两个概率模型（Transition Probability，Emission Probability）来对输入序列的词性进行联合预测，转移概率根据马尔科夫链计算出概率最大的词性序列，发射概率根据词性序列采样所有的单词序列，最终选择能让产生输入的单词序列概率最大的词性序列，并作为输出
  
    
  
- #### Viterbi 算法的作用？

  - 在 HMM 里面需要计算所有采样的词性序列和输入做概率计算，计算量比较大，使用 Viterbi 算法可能避免穷举计算
  
    
  
- #### HMM 和 CRF 的区别？

  - HMM 在训练数据集较少的时候，相对于 CRF 效果可能会比较好，因为 HMM 在概率计算过程中，会有一些先验的概率，并不完全来自于训练集，在没有采样到的数据集中有可能这种先验概率就是正确的

    

## Conditional Random Field（CRF）

- #### 概念

  - CRF|Viterbi|CRF 和 HMM 的区别
  
    
  
- #### CRF 中 Viterbi 算法原理是什么？有什么用？

- #### CRF 中的两个矩阵分别是什么？有什么用？
