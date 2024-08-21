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

## Output
Uses We from input encoding
L: logits matrix
P: probabilities matrix

$$
L = X W_e^t \in \reals^{c \times |V|}
$$

$$
P = softmax(L)
$$

To determine the next token, take the index of the maximum probability for the last vector in P and use that as an index for V

$$
t_{next} = V[max\_idx(P[c])]
$$