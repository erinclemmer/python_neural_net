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
Y_1 \in \reals^{c \times h},
b_1\in\reals^{c \times h},
b_2\in\reals^{h \times c}
$$

$$
z = R W_1 + b_1
\in \reals^{c \times h}
$$

$$
Y_1 = \sigma(z)
\in \reals^{c \times h}
$$

$$
Y_2 = Y_1 W_2 + b_2 \in \reals^{c \times d}
$$

$$
FF(R): \reals^{c \times d} \rightarrow \reals^{c \times d}
$$

## Output
Uses We from input encoding  
Z: logits matrix  
y hat: probabilities matrix  
Wu: unembedding weights

$$
W_u \in \reals^{d \times |v|}, X\in \reals^{c \times d}
$$

$$
Z = X W_u^t \in \reals^{c \times |V|}
$$

$$
\sigma(z_{t,j}) =
\frac{e^{z_{t,j}}}
{\sum^V_{k=i}{e^{z_{t,k}}}}
$$
^ Softmax Function

$$
i = \{ 1, ..., c \}
$$

$$
j = \{ 1, ..., |V| \}
$$

$$

\hat{y} = 
\{
\sigma(z_{i,j}) | \forall i,j
\}
\in \reals^{c \times |V|}
$$

To determine the next token, take the index of the maximum probability for the last vector in P and use that as an index for V

$$
t_{next} = V[max\_idx(\hat{y}[c])]
$$

$$O(X): \reals^{c \times d} \rightarrow \reals^{c \times |V|}$$

## Loss function
Using cross entropy

Loss per token
$$
L_i = - \sum_{j=1}^{V} y_{ij} log(\hat{y}_{ij}) = -log(\hat{y}_{ij*})
$$
\* denotes vector of expected token

Loss vector
$$
L = L_i(\hat{y}_{ij}) \in \reals^{1 \times c}
$$

Average Loss
$$
L_i = - \frac{1}{N} \sum_{i=0}^{N} log(\hat{y}_{ij*})
$$

## Output (backwards)
$$
\frac{dL}{dW_u} = 
\frac{dL}{d\hat{y}} \times
\frac{d\hat{y}}{dz} \times
\frac{dz}{dW_u}
$$

such that

$$
W_u \in \reals^{d \times |V|},
X \in \reals^{c \times d},
\hat{y} \in \reals^{c \times |V|}
$$

Special rule for cross entropy + softmax

$$
\frac{dL}{dz} = \hat{y} - y
\in \reals^{c \times |V|}
$$
where y is one-hot vector

Derivative of output fn
$$
\frac{dz}{dW_u} = X \in \reals^{c \times d}
$$

Alltogether
$$
\frac{dL}{dW_u} = 
((\hat{y} - y)^tX)^t
\in \reals^{d \times |V|}
$$

To pass further down
$$
\frac{dL}{dX} =
\frac{dL}{d\hat{y}} \times
\frac{d\hat{y}}{dz} \times
\frac{dz}{dX}
$$

$$
\frac{dz}{dX} =
W_u
$$

$$
\frac{dL}{dX} =
(\hat{y} - y) W_u^t
\in \reals^{c \times d}
$$

## Feed Forward backwards
R: residual stream
$$
R \in \reals^{c \times d}
$$

$$
\frac{dY_2}{dW_2} = Y_1
\in \reals^{c \times h}
$$

$$
\frac{dY_2}{db_2} = 1
$$

$$
\frac{dY_2}{dW_1} =
\frac{dY_2}{dY_1} \times
\frac{dY_1}{dz} \times
\frac{dz}{dW_1}
$$

$$
\frac{dY_2}{dY_1} = W_2
$$

$$
\frac{dY_1}{dz} = \sigma'(z)
$$

$$
\frac{dz}{dW_2} =
R \in \reals^{c \times d}
$$

$$
\frac{dY_2}{dW_1} =
R W_2^t \times \sigma'(z)
\in \reals^{c \times h}
$$

$$
\frac{dY_2}{db_1} = 
\frac{dY_2}{dY_1} \times
\frac{dY_1}{dz} \times
\frac{dz}{db_1}
$$

$$
\frac{dz}{db_1} = 1
$$

$$
\frac{dY_2}{db_1} = 
R W_2^t
\in \reals^{c \times h}
$$

## Feedforward + output backwards

$$
\frac{dL}{dW_{2ff}} =
\frac{dL}{dX_o} \times
\frac{dX_o}{dW_{2ff}}
$$

$$
\frac{dL}{dX_{o}} =
(\hat{y} - y) W_u^t
\in \reals^{c \times d}
$$

$$
\frac{dX_o}{dW_{2ff}} =
Y_1 \in \reals^{c \times h}
$$

$$
\frac{dL}{dW_{2ff}} =
(((\hat{y} - y) W_u^t)^t Y_1)^t
\in \reals^{h \times d}
$$

$$
\frac{dL}{db_{2ff}} =
(\hat{y} - y) W_u^t
\in \reals^{h \times c}
$$

$$
R(r) = norm(r + f(r))
$$

where f is the FF or attn function