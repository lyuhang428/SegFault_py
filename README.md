<h1 id="english">SegFault_py</h1>
<a href="#chinese">[中文]</a>

## About

Molecular DFT, plane wave DFT, HF code implemented in Python/Fortran. Fast prototype development and is intented to be integrated to [`SegFault`](https://github.com/lyuhang428/SegFault). This project is currently in its very early stage and only supports closed-shell molecular Hartree-Fock, LDA, GGA, meta-GGA calculations, and plane wave, $\Gamma$-only, norm-conversing pseudopotential DFT calculations. 

As this repository is fast development and testing purposes, no input file is needed. 


## Code structure
```txt
src/
├── dft
├── hf
└── pw
```

`hf` implements in almost pure Python, with the molecular integral engine written in C and Fortran. The C wrapper is `ctypes`, and interface to Fortran relies on `f2py`.

`dft` implements in pure Python, where only LDA functional is supported. Different from the implementation of [`SegFault`](https://github.com/lyuhang428/SegFault), this `dft` numerically solves Poisson's equation to construct Coulomb operator, which avoids the expensive 2e4c integral but the numerical solver is unfortunately still slow.

`pw` implements in pure Python, which does plane wave DFT calculations. Currently only restricted LDA/PBE, $\Gamma$-only, cubic cell, and ONCV pseudopotentials are supported.

## Validation/Benchmark
Validation against Quantum ESPRESSO (version 7.3.1)

Machine: ThinkPad T14p, 13th Gen Intel(R) Core(TM) i5-13500H, 1 threads.

Functional: PBE

PP: ONCV sg-15

Ecutwfc: 25 Ry

Cubic cell

| System | $\quad\Delta E_\text{tot}\quad$ | $\quad\Delta E_\text{1e}\quad$ | $\quad\Delta E_\text{hartree}\quad$ | $\quad\Delta E_\text{xc}\quad$ | $\quad\Delta E_\text{ewald}$ |  
| :- | :-: | :-: | :-: | :-: | :-: |
| AlCl3 | 7.1184e-06 | -4.9011e-04 | 4.9840e-04 | -1.1493e-06 | -1.6731e-08 |
| BCl3 | 1.0507e-05 | -1.8159e-03 | 1.7018e-03 | 1.2466e-04 | -9.3102e-09 |
| C2H2 | 1.2799e-05 | 8.7348e-04 | -1.0557e-03 | 1.9501e-04 | -1.1471e-08 |
| C2H4 | 1.4911e-05 | 1.0212e-03 | -1.1354e-03 | 1.2908e-04 | -1.3717e-08 |
| CH4 | 6.5361e-06 | 2.5736e-03 | -2.9805e-03 | 4.1344e-04 | -2.1372e-08 |
| CO | 1.0566e-05 | 1.4065e-03 | -1.6775e-03 | 2.8156e-04 | -2.0647e-08 |
| CO2 | 8.0224e-06 | 5.1040e-04 | -7.5096e-04 | 2.4859e-04 | -1.6963e-08 |
| H2O | 7.5373e-06 | -3.3076e-04 | 2.9978e-04 | 3.8560e-05 | -3.9385e-08 |
| H2O2 | 7.2917e-06 | 1.1640e-04 | -1.5397e-04 | 4.4908e-05 | -4.6596e-08 |
| HCN | 9.6418e-06 | -1.8021e-04 | 2.2676e-04 | -3.6897e-05 | -1.4470e-08 |
| Li2 | 6.0264e-05 | -3.7071e-04 | 4.6695e-04 | -3.5936e-05 | -4.3045e-08 |
| LiH | 3.1090e-05 | 3.1153e-04 | -6.4271e-04 | 3.6231e-04 | -4.4209e-08 |
| NaCl | -2.9635e-05 | 1.4112e-03 | -1.7801e-03 | 3.3931e-04 | -2.5562e-08 |
| NH3 | -2.1936e-06 | -3.2762e-03 | 3.9760e-03 | -7.0201e-04 | -2.7619e-08 |
| O2 | -1.5287e-03 | 9.3758e-03 | -1.4555e-02 | 3.6501e-03 | -3.7658e-08 |
| PH3 | 1.3923e-05 | 4.4414e-04 | -5.9613e-04 | 1.6596e-04 | -3.7603e-08 |
| SO2 | 8.8739e-06 | -3.2041e-03 | 4.0803e-03 | -8.6730e-04 | -7.1584e-08 |




---

<h1 id="chinese">SegFault_py</h1>
<a href="#english">[English]</a>

## 简介

Python/Fortran 实现的分子 DFT，平面波 DFT，HF 计算代码。用于为 [`SegFault`](https://github.com/lyuhang428/SegFault) 作快速原型开发和结果验证。该项目目前处于早期阶段，只能计算单点能。平面波代码目前只能处理闭壳层体系，模守恒赝势，和单 $\Gamma$点。

由于该项目主要用于测试，因此没有设计使用输入文件。


## 代码结构

```txt
src/
├── dft
├── hf
└── pw
```

## 结果验证
计算结果与 Quantum ESPRESSO (version 7.3.1) 进行比较。
硬件：ThinkPad T14p, 13th Gen Intel(R) Core(TM) i5-13500H, 单线程。
PBE泛函
ONCV sg-15 赝势
波函数动能截断值 25 Ry
立方盒子

| System | $\quad\Delta E_\text{tot}\quad$ | $\quad\Delta E_\text{1e}\quad$ | $\quad\Delta E_\text{hartree}\quad$ | $\quad\Delta E_\text{xc}\quad$ | $\quad\Delta E_\text{ewald}$ |  
| :- | :-: | :-: | :-: | :-: | :-: |
| AlCl3 | 7.1184e-06 | -4.9011e-04 | 4.9840e-04 | -1.1493e-06 | -1.6731e-08 |
| BCl3 | 1.0507e-05 | -1.8159e-03 | 1.7018e-03 | 1.2466e-04 | -9.3102e-09 |
| C2H2 | 1.2799e-05 | 8.7348e-04 | -1.0557e-03 | 1.9501e-04 | -1.1471e-08 |
| C2H4 | 1.4911e-05 | 1.0212e-03 | -1.1354e-03 | 1.2908e-04 | -1.3717e-08 |
| CH4 | 6.5361e-06 | 2.5736e-03 | -2.9805e-03 | 4.1344e-04 | -2.1372e-08 |
| CO | 1.0566e-05 | 1.4065e-03 | -1.6775e-03 | 2.8156e-04 | -2.0647e-08 |
| CO2 | 8.0224e-06 | 5.1040e-04 | -7.5096e-04 | 2.4859e-04 | -1.6963e-08 |
| H2O | 7.5373e-06 | -3.3076e-04 | 2.9978e-04 | 3.8560e-05 | -3.9385e-08 |
| H2O2 | 7.2917e-06 | 1.1640e-04 | -1.5397e-04 | 4.4908e-05 | -4.6596e-08 |
| HCN | 9.6418e-06 | -1.8021e-04 | 2.2676e-04 | -3.6897e-05 | -1.4470e-08 |
| Li2 | 6.0264e-05 | -3.7071e-04 | 4.6695e-04 | -3.5936e-05 | -4.3045e-08 |
| LiH | 3.1090e-05 | 3.1153e-04 | -6.4271e-04 | 3.6231e-04 | -4.4209e-08 |
| NaCl | -2.9635e-05 | 1.4112e-03 | -1.7801e-03 | 3.3931e-04 | -2.5562e-08 |
| NH3 | -2.1936e-06 | -3.2762e-03 | 3.9760e-03 | -7.0201e-04 | -2.7619e-08 |
| O2 | -1.5287e-03 | 9.3758e-03 | -1.4555e-02 | 3.6501e-03 | -3.7658e-08 |
| PH3 | 1.3923e-05 | 4.4414e-04 | -5.9613e-04 | 1.6596e-04 | -3.7603e-08 |
| SO2 | 8.8739e-06 | -3.2041e-03 | 4.0803e-03 | -8.6730e-04 | -7.1584e-08 |


---

[go to top](#)
