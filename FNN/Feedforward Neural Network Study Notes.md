# Feedforward Neural Network Study Notes

Sicong Zhao



### **Part1: The equations of backpropagation**

**Equation 1:  **The error vector in the output layer
$$
\delta^L = \nabla_a C \odot \sigma'(z^L)
$$
**Equation 2: **The relationship of error between two consecutive layer
$$
\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^{l})
$$
**Equation 3: **The derivative of cost function of biases
$$
\frac{\mathrm{d}C}{\mathrm{d}b^l} = \delta^l
$$
**Equation 4: ** The derivative of cost function of weights
$$
\frac{\mathrm{d}C}{\mathrm{d}w^l_{kj}} = \delta^l_k a^{l-1}_j
$$


###  Part2: Proof of Backpropagate Formulas



**(1) Equation 1** 
$$
\begin{align}
\delta^{L}_j &= \frac{\mathrm{d}C}{\mathrm{d}z^L_j} \\
&= \sum_{k}\frac{\mathrm{d}C}{\mathrm{d}a^L_k} \times \frac{\mathrm{d}a^L_k}{\mathrm{d}z^L_j} \\
& \text{Only when k=j, the second term could be non-zero} \\
&= \frac{\mathrm{d}C}{\mathrm{d}a^L_j} \times \frac{\mathrm{d}a^L_j}{\mathrm{d}z^L_j}   \\
&= \frac{\mathrm{d}C}{\mathrm{d}a^L_j} \sigma'(z^L_j) \\

\\

\text{The vectorized expression is:} \qquad \delta^L &= \nabla_a C \odot \sigma'(z^L)
\end{align}
$$


**(2) Equation 2** 
$$
\begin{align}

\delta^l &= ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l) \\

\end{align}
$$
First, pull out a single element of the matrix calculation.
$$
\begin{align}
\delta^l_j &= \sum_k(w^{l+1}_{kj} \delta^{l+1}_k) \sigma'(z^l_j)  \\
\end{align}
$$
Second, expand the left side.
$$
\begin{align}
\delta^l_j &= \frac{\mathrm{d}C}{\mathrm{d}z^{l+1}}\times \frac{\mathrm{d}z^{l+1}}{\mathrm{d}z^l_j}  \\
&= \sum_k \frac{\mathrm{d}C}{\mathrm{d}z^{l+1}_k}\times \frac{\mathrm{d}z^{l+1}_k}{\mathrm{d}z^l_j} \\
&= \sum_k \delta^{l+1}_k \times \frac{\mathrm{dz^{l+1}_k}}{\mathrm{d}a^l_j} \times \frac{\mathrm{d}a^l_j}{\mathrm{d}z^l_j} \\
&= \sum_k \delta^{l+1}_k \times w^{l+1}_{kj} \times \sigma'(z^l_j)
\end{align}
$$


**(3) Equation 3** 
$$
\begin{align}
\frac{\mathrm{d}C}{\mathrm{d}b^l} &= \frac{\mathrm{d}C}{\mathrm{d}z^l} \times \frac{\mathrm{d}z^l}{\mathrm{d}b^l} \\
&= \frac{\mathrm{d}C}{\mathrm{d}z^l} \\
&= \delta^l
\end{align}
$$


**(4) Equation 4** 
$$
\begin{align}
\frac{\mathrm{d}C}{\mathrm{d}w^l_{kj}} &= \sum_i \frac{\mathrm{d}C}{\mathrm{d}z^l_i} \times \frac{\mathrm{d}z^l_i}{\mathrm{d}w^l_{kj}} \\
& \text{Only when i=k, the second term could be non-zero} \\
&= \frac{\mathrm{d}C}{\mathrm{d}z^l_k} \times \frac{\mathrm{d}z^l_k}{\mathrm{d}w^l_{kj}} \\
&=  \delta^l_k \times a^{l-1}_j
\end{align}
$$


### Part3: Implementation in Python

*In following code, `L` stands for last layer, `l` stands for second to last layer*

**(1) Equation 1**
$$
\delta^L = \nabla_a C \odot \sigma'(z^L)
$$

```python
delta_L = cost_derivative(output, y) * sigmoid_prime(z_L)
```



**(2) Equation 2**
$$
\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^{l})
$$

```python
delta_l = np.dot(weights_L.transpose(), delta_L) * sigmoid_prime(z_l)
```



**(3) Euqtion 3**
$$
\frac{\mathrm{d}C}{\mathrm{d}b^l} = \delta^l
$$

```python
nabla_bias = delta_L # Just use the result of previous 2 equation
```



**(4) Equation 4**
$$
\frac{\mathrm{d}C}{\mathrm{d}w^l_{kj}} = \delta^l_k a^{l-1}_j
$$

```
nabla_weight = np.dot(delta_L, sigmoid(z_l).transpose())
```



```python
# Helper functions

def cost_derivative(out_put, y):
  return out_put - y

def sigmoid(z):
  return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))
```









