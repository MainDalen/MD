
# Molecular Dynamics: 100 LJ particles in 2D

## 1. Lennard Jones Potential

$$\displaystyle U(r) = 4\varepsilon \cdot \left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right]$$

## 2. Cut Potential

$$
U(r)=
\begin{cases}
\varphi(r) - \varphi(R_c), r\le R_c\\
0, r>R_c
\end{cases}
, R_c = 2.5\sigma
$$

## 3. Verlet integration

$$
r(t+\Delta t)=r(t) + v(t) \cdot \Delta t + \frac{1}{2} a(t) \Delta t^2
$$
$$
v(t+\frac{\Delta t}{2}) = v(t) + \frac{1}{2}a(t)\Delta t
$$
$$
a(t+\Delta t) = -\frac{1}{m}\nabla U(r(t+\Delta t))\\[3mm]
$$
$$
v(t+\Delta t) = v(t+\Delta t) + \frac{1}{2}a(t+\Delta t)\Delta t
$$

## 4. Velocity Scaling

$$
v(t+\frac{\Delta t}{2}) = \sqrt{\frac{T_0}{T(t)}}v(t) + \frac{1}{2}a(t)\Delta t
$$

# Code

1. [MD Simulations](MD.py)
2. [MSD Calculation](MD.ipynb)