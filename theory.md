## Forward Pass

$$W^k \in \reals^{n \times m}, b^k \in \reals^{n}, y \in \reals^{n}$$

$$ z^l = W^l a^{l-1} + b^l$$

$$a^l  = \sigma(z^l) $$
> where sigma is activation function

## Loss Fn:
$$L=\frac{(\hat{y} - y)^2}{2}$$


## Backwards top layer
> K is last layer

### Derivative of loss w.r.t last layer bias
$$\frac{dL}{db^k} = \frac{dL}{\hat{y}} \times \frac{d\hat{y}}{dz^k} \times \frac{dz^k}{db^k}$$
$$\frac{dL}{db^k} = (\hat{y} - y) \times \sigma'(z^k) \times 1 $$
> element wise multiplication

### Delta
Intermediate value
$$\delta^k = \frac{dL}{da^{k-1}} = 
\frac{dL}{d\hat{y}} \times 
\frac{d\hat{y}}{dz^k} \times 
\frac{dz^k}{da^{k-1}}
$$

$$\delta^k = 
(\hat{y} - y) \times 
\sigma '(z^k) \times 
W^k
$$

$$\delta^k = 
\frac{dL}{db^k} \times 
W^k
$$

$$
(\frac{dL}{db^k} \in \reals^{1 \times n}) \times 
(W \in \reals^{n \times m}) = 
\delta^k \in \reals^{1 \times m}
$$

> Matrix Multiplication

### Derivative of loss w.r.t. last layer weights
$$\frac{dL}{dW^k} = 
\frac{dL}{d\hat{y}} \times 
\frac{d\hat{y}}{dz^k} \times 
\frac{dz^k}{dW^k} 
$$

$$\frac{dL}{dW^k} = 
(\hat{y} - y) \times 
\sigma '(z^k) \times 
a^{k-1}
$$

$$\frac{dL}{dW^k} = 
\frac{dL}{db^k} \times 
a^{k-1}
$$

$$
outer(\frac{dL}{db^k} \in \reals^{n}, 
a^{k-1} \in \reals^m) =
\frac{dL}{dW^k} \in \reals^{n \times m}$$
> outer product

## Backwards lower Layers
$$c = k - n$$
$$W^{c+1} \in \reals^{n \times m}, W^{c} \in \reals^{n \times o}$$

### Derivative of loss w.r.t. bias at layer k - n
$$ \frac{dL}{db^c} = 
\frac{dL}{d\hat{y}} \times 
\frac{d\hat{y}}{dz^k} \times 
\frac{dz^k}{da^{k-1}} \times
\frac{da^{k-1}}{dz^{k-1}} \times
\frac{dz^{k-1}}{da^{k-2}} \times
\frac{da^{k-2}}{dz^{k-2}} \times
... \times
\frac{da^c}{dz^c} \times
\frac{dz^c}{db^c}
$$

$$ \frac{dL}{db^c} = 
\frac{dL}{db^{c+1}} \times
\frac{dz^{c+1}}{da^c} \times
\frac{da^c}{dz^c} \times
\frac{dz^c}{db^c}
$$

$$ \frac{dL}{db^c} = 
\frac{dL}{db^{c+1}} \times
W^{c+1} \times
\sigma ' (z^c) \times
1
$$
$$ 
(\frac{dL}{db^{c+1}} \in \reals^{1 \times n}) \times 
(W^{c+1} \in \reals ^{n \times m}) = 
\vec{i} \in R^{1 \times m}
$$
> matrix multiplication
$$
(\vec{i} \in R^{1 \times m}) \times
(\sigma'(z^c) \in \reals^{1 \times m}) = 
\frac{dL}{db^{c}} \in \reals^{1 \times m}
$$
> element wise multiplication

### Delta at layer k - n
$$\delta^c = \frac{dL}{da^{c-1}} = 
\frac{dL}{d\hat{y}} \times 
\frac{d\hat{y}}{dz^k} \times 
\frac{dz^k}{da^{k-1}} \times
\frac{da^{k-1}}{dz^{k-1}} \times
\frac{dz^{k-1}}{da^{k-2}} \times
... \times
\frac{da^{c+1}}{dz^{c+1}} \times
\frac{dz^{c+1}}{da^{c}}
$$

$$\delta^c =
\delta^{c+1} \times
\frac{da^{c}}{dz^{c}} \times
\frac{dz^{c}}{da^{c-1}}
$$

$$\delta^c =
\delta^{c+1} \times
\sigma ' (z^{c}) \times
W^{c}
$$

$$ (\delta^{c+1} \in \reals^{1 \times m}) \times
(\sigma'(z^c) \in \reals^{1 \times m}) =
\vec{i} \in \reals^{1 \times m}
$$
> element wise multiplication

$$
(\vec{i} \in \reals^{1 \times m}) \times
(W^c \in \reals^{m \times o}) =
\delta^c \in \reals^{1 \times o}
$$
> matrix multiplication

### Derivative of loss w.r.t. weights at layer k - n
$$ \frac{dL}{dW^c} = 
\frac{dL}{d\hat{y}} \times
\frac{d\hat{y}}{dz^k} \times
\frac{dz^k}{da^{k-1}} \times
\frac{da^{k-1}}{dz^{k-1}} \times
\frac{dz^{k-1}}{da^{k-2}} \times
... \times
\frac{da^{c}}{dz^{c}} \times
\frac{dz^{c}}{dW^{c}}
$$

$$ \frac{dL}{dW^c} = 
\delta^{c+1} \times
\frac{da^{c}}{dz^{c}} \times
\frac{dz^{c}}{dW^{c}}
$$

$$ \frac{dL}{dW^c} = 
\delta^{c+1} \times
\sigma ' (z^c) \times
a^{c-1}
$$

$$
(\delta^{c+1} \in \reals^{1 \times m}) \times
(\sigma ' (z^c) \in \reals^{1 \times m}) =
\vec{i} \in \reals^{1 \times m}
$$
> element wise multiplication

$$
outer((\vec{i} \in \reals^{1 \times m}),
(a^{c-1} \in \reals^{1 \times o})) =
\frac{dL}{dW^c} \in \reals^{m \times o}
$$
> outer product