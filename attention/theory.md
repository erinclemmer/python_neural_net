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
O = S' \times V \in \reals^{c \times d}
$$