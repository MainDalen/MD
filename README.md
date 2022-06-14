
# Molecular Dynamics: 100 LJ particles - mean squared displacement and auto-diffusion with different T

# 1. Lennard Jones Potential

$$\displaystyle U(r) = 4\varepsilon \cdot \left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right]$$

```python
def ljp(r, eps=1, sig=1):
    return 4 * eps * ((sig/r)**12 - (sig/r)**6)
```

# 2. Cut Potential

$$
U(r)=
\begin{cases}
\varphi(r) - \varphi(R_c), r\le R_c\\
0, r>R_c
\end{cases}
, R_c = 2.5\sigma
$$