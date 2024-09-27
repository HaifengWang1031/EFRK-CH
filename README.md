# Exponential free Runge--Kutta (EFRK) Method for Cahn-Hilliard equation

Code accompanying the paper "An exponential-free Runge–Kutta framework for developing third-order
unconditionally energy stable schemes for the Cahn-–Hilliard equation" by Haifeng Wang, Hong Zhang, Qian Xu, and Songhe Song.

## Abstract

In this work, we develop a class of up to third-order energy-stable schemes for the Cahn--Hilliard equation. Starting from Lawson's integrating factor Runge--Kutta method, which is widely used for stiff semilinear equations, the limitations of this method are discussed, such as the inability to preserve equilibrium and the oversmooth of interfacial layers of the solution's profile because of the exponential damping effects. To overcome this drawback, we approximate the exponential term using a class of sophisticated Taylor polynomials, leading to a new Runge--Kutta framework called exponential-free Runge--Kutta. By incorporating stabilization techniques, we analyze the energy stability of the proposed schemes and prove that under specific constraints on the underlying Runge--Kutta coefficients, this scheme preserves the original energy dissipation without any time-step restrictions. Furthermore, we also analyze the linear stability and establish an error estimate in the $\ell^2$ norm. A series of numerical experiments validate the high-order accuracy, mass conservation, and energy dissipation of our schemes.


## Citation

```
@ article{wang2024exponential,
title={An exponential-free Runge–Kutta framework for developing third-order unconditionally energy stable schemes for the Cahn–Hilliard equation},
author={Haifeng, Wang and Hone, Zhang and Xu, Qian and Songhe, Song},
journal={},
pages={},
year={}
publisher={}
}
```