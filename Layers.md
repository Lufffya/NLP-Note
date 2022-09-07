## Feedforward Neural Network（FNN）

- #### 概念

  - FNN 是前馈神经网络，每个神经元只和前一层的神经元之间相互连接，接收前一层输出，同一层之间的神经元彼此之间没有关联
  
    

## Convolutional Neural Network（CNN）

- #### 概念

  - CNN 是卷积神经网络，本质上是一个简化版的全连接前馈神经网络
  
  - 在网络结构中设置了对于图像处理的先验知识，比如，filter的设计可以对图像进行降维，提取一张图像中多个相同特征片段
  
    

## Recurrent Neural Network（RNN）

- #### 概念

  - 梯度消失/爆炸问题|门控单元问题
  
  - RNN 是循环神经网络，是一个从序列到序列的神经网络
  
    
  
- #### 为什么 RNN 会导致梯度消失/爆炸？

  - 如果 RNN 的输入序列过长，就可能导致梯度消失/爆炸，在训练期间进行梯度计算时，过长的序列会导致权重出现指数增长或衰减
  
    
  
- #### 如何解决梯度消失/爆炸的问题？

  - 可以进行梯度截断，正则化，归一化，残差或使用更不易于造成梯度消失/爆炸的 LSTM
  
    
  
- #### LSTM 中有哪些门控单元？原理是什么？

  

## Hidden Makov Model（HMM）

- #### 概念

  - HMM|HMM 和 CRF 的区别
  
    

## Conditional Random Field（CRF）

- #### 概念

  - CRF|Viterbi|CRF 和 HMM 的区别
  
    
  
- #### CRF 中 Viterbi 算法原理是什么？有什么用？

- #### CRF 中的两个矩阵分别是什么？有什么用？
