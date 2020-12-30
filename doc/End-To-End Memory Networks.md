### 《End-To-End Memory Networks》论文阅读笔记

#### 1，论文的贡献

之前提出的Memory Network由于设计的缺陷，一些函数并不是连续的，不可导，所以较难通过BP训练，并且在每层网络都需要监督。而作者提出来的End-To-End Memory Networks使用连续表示存储知识，并且在函数设计上将原有的不连续函数替换成连续函数，使之能够BP训练。

作者还将multiple hops机制应用在LSTM，并且通过实验证明这是提升性能的关键。

#### 2，核心算法

##### 2.1 Single Layer

**Input memory representation**：

输入$X = \{x_1, x_2, ... , x_N\}$，通过一个输入嵌入矩阵A将输入X转化为$M=\{m_1,m_2,...,m_N\}$，query通过嵌入矩阵B转化为$u$，然后通过内积计算得到u和输入的相关程度，即注意力分布。

$p_i = Softmax(u^T* m_i)$

**Output memory representation**：

输入$X = \{x_1, x_2, ... , x_N\}$，通过一个输出嵌入矩阵C将输入X转化为$C=\{c_1,c_2,...,c_N\}$，输出o是输出向量在注意力分布上的期望。

$o = \underset{i}{\sum}{p_ic_i}$

**Generating the final prediction**:

$\hat a = Softmax(W(o+u))$, W是一个权重矩阵(维度V*d)

下面左侧是单层模型的结构图，右侧是多层模型的结构图：

![image-20201229101716806](C:\Users\xmh\AppData\Roaming\Typora\typora-user-images\image-20201229101716806.png)

##### 2.2 Multiple Layers

在single layer的基础上，嵌入K hop操作。单层模型的堆叠方式：

- The input to layers above the first is the sum of the output $o_k$ and the input $u_k$ from layer k
  (different ways to combine $o^k$ and $u_k$ are proposed later):$u^{k+1} = u^k + o^k$ (下一层的查询的向量=当前层的查询向量+当前层的输出)
- Each layer has its own embedding matrices $A^k$, $C^k$, used to embed the inputs X.However, as
  discussed below, they are constrained to ease training and reduce the number of parameters.
  - **Adjacent**: the output embedding for one layer is the input embedding for the one above.$A^{k+1} = C^k$,最后的答案预测矩阵等于最后一层的输出嵌入矩阵，即$W^T = C^K$,the question embedding to match the input embedding of the first layer，即$B = A^1$.
  - **Layer-wise (RNN-like)**:the input and output embeddings are the same across different
    layers, $A^1 = A^2= ...=A^K$ and $C^1 = C^2= ...= C^K$.  We have found it useful to
    add a linear mapping H to the update of u between hops; that is $u^{k+1} = Hu^K + o^K$
- At the top of the network, the input to W also combines the input and the output of the top memory layer: $\hat a = Softmax(Wu^{K+1}) = Softmax(W(o^K + u^K))$.

##### 2.3 模型细节

**Sentence Representation**:词袋模型无法捕捉词的顺序信息。作者使用了position encoding(PE)机制将词顺序信息编码进表示。$m_i = \underset{j}{\sum}{l_j * Ax_{ij}}$.  $l_j$ is a column vector with the structure $l_{kj}$ = (1 − j/J) − (k/d)(1 − 2j/J) (assuming 1-based indexing)

**Temporal Encoding**: To enable our model to address them, we modify the memory vector so that $m_i = \underset{j}{\sum}{Ax_{ij} + T_{A(i)}}$, where $T_{A(i)}$ is the ith row of a special matrix TA that encodes temporal information. The output embedding is augmented in the same way with a matrix Tc
($c_i = \underset{j}{\sum}{Cx_{ij} + T_{C(i)}}$). Both $T_{A(i)}$ and $T_{C(i)} $are learned during training. They are also subject to the same sharing constraints as A and C

**Learning time invariance by injecting random noise**: we have found it helpful to add "dummy" memories to regularize TA. That is, at training time we can randomly add 10% of empty memories to the stories. We refer to this approach as random noise (RN).