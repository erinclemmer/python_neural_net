## Input Embeddings + Positional

x: input  
c: context length  
V: vocabulary  
t: token position

$$
\vec{x} \in \reals^{1 \times c},
\vec{x}_t \in V
$$


Weight embeddings  
d: hidden dimension
$$W_e \in \reals^{|V| \times d}$$

Positional embeddings
$$W_p \in \reals^{c \times d}$$

$$
X_t = 
W_e[\vec{x_t}] +
W_p[t]
$$

$$ 
X_t \in \reals^{1 \times d},
X \in \reals^{c \times d} 
$$

$$ inp(x) $$

## Attention Head
$$
W_q, W_k, W_v \in \reals^{d \times d}
$$

$$
b^Q, b^K, b_V \in \reals^{c \times d}
$$

$$
Q=XQ+b^Q \in \reals^{c \times d}
$$

$$
K=XK+b^K \in \reals^{c \times d}
$$

$$
V=XV+b^V \in \reals^{c \times d}
$$

$$
S_t=Q_t K^t \in \reals^{1 \times c}
$$

$$
S = Q K^t \in \reals^{c \times c}
$$

$$
S' = softmax(\frac{s}{\sqrt{m}})
$$

$$
O = norm(S' \times V) \in \reals^{c \times d}
$$


$$
attn(X)
$$

## Feed Forward
h: hidden layer dimension  
a(): non-linear activation function

$$
W_1 \in \reals^{d \times h}, 
W_2 \in \reals^{h \times d},
b_1\in\reals^{c \times h},
b_2\in\reals^{h \times c}
$$

$$
O_1 = a(X W_1 + b_1) \in \reals^{c \times h}
$$

$$
O_2 = O_1 W_2 + b_2 \in \reals^{c \times d}
$$

$$
FF(X)
$$

## Output
Uses We from input encoding  
Lo: logits matrix  
P: probabilities matrix  
Wu: unembedding weights

$$
W_u \in \reals^{d \times |v|}
$$

$$
L_o = X W_u^t \in \reals^{c \times |V|}
$$

$$
P = softmax(L)
$$

To determine the next token, take the index of the maximum probability for the last vector in P and use that as an index for V

$$
t_{next} = V[max\_idx(P[c])]
$$

$$O(X)$$

## Loss function
Using cross entropy

Loss per token
$$
L_i = - \sum_{j=1}^{V} y_{ij} log(\hat{y}_{ij}) = -log(\hat{y}_{ij*})
$$
\* denotes vector of expected token

TODO  
Loss vector
$$
L = L_i() \in \reals^{1 \times c}
$$

Average Loss
$$
L_i = - \frac{1}{N} \sum_{i=0}^{N} log(\hat{y}_{ij*})
$$

## Output (backwards)
$$
\frac{dL}{W_u} = 
\frac{dL}{d\hat{y}_{ij*}} \times
\frac{d\hat{y}_{ij*}}{dL_o}


$$

Derivative of loss fn
$$
\frac{dL}{d\hat{y}_{ij*}} =
-\frac{1}{\hat{y}_{ij*}}
$$

TODO  
Derivative of softmax
$$
\frac{d\hat{y}_{ij*}}{dL_o} =
\begin{cases}
\hat{y}_{ij*}
\end{cases}
$$