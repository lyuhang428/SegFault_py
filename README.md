# 闭壳层Hartree-Fock代码

**[English](README.md)**

`Python`+`Fortran`+`C/C++`混编Hartree-Fock闭壳层量子化学计算代码。代码易读且易修改，但是相同机器上性能比`Psi4`慢近20倍. 

分子积分计算使用`McMurchie-Davidson`算法（理论上支持任意角动量，但是只测试到`f`轨道），电子排斥积分同时提供`Rys`求积算法（最高仅支持到`d`轨道）. 

分子积分部分使用`Fortran`并通过`f2py`模块提供`Python`接口. 

计算速度上比`Psi4`慢近20倍，但是代码结构简单，易读易修改. 

`Fortran`代码部分依赖`gsl`库计算阶乘与合流超几何函数（Confluent hypergeometric function）. 若该库未添加到系统路径，需进行链接：`export LD_LIBRARY_PATH=/path/to/gsl`. 

使用`multiprocessing`模块进行并行计算，同时利用电子排斥积分八重对称性以减少计算量. 
