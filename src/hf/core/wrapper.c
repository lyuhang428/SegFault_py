// gcc -c -fPIC wrapper.c
// gcc -shared -O2 wrapper.o -lgsl -lgslcblas -lm -o libwrapper.so
// mmd3.py call this C library
// mmd3.py is faster then pure Python mmd.py, but slower than Fortran wrapped core.f90+mmd2.py
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "gsl/gsl_sf_hyperg.h"



/* signature */
double boys(const int, const double);
double hermite_gauss_e(const int, const int, const int, const double, const double, const double);
double coulomb_auxiliary_r(const int, const int, const int, const int,
                           const double, const double, const double, const double, const double);
double eri_3d(const double, const int*, const double*, 
              const double, const int*, const double*, 
              const double, const int*, const double*, 
              const double, const int*, const double*);



/* implementation */
double boys(const int n, const double T){
    return gsl_sf_hyperg_1F1(n + 0.5, n + 1.5, -T) / (2. * n + 1.);
}


double hermite_gauss_e(const int i, const int j, const int t, const double qx, const double alpha, const double beta){
    double p, q;
    p = alpha + beta;
    q = alpha * beta / p;

    if ((t < 0) || (t > (i + j)))
    {
        return 0.;
    }
    else if (i == 0 && j == 0 && t == 0)
    {
        return exp(-q * qx * qx);
    }
    else if (j == 0)
    {
        return 0.5 / p   * hermite_gauss_e(i-1, j, t-1, qx, alpha, beta) - \
        (q * qx / alpha) * hermite_gauss_e(i-1, j, t,   qx, alpha, beta) + \
        (t + 1)          * hermite_gauss_e(i-1, j, t+1, qx, alpha, beta);
    }
    else
    {
        return 0.5 / p  * hermite_gauss_e(i, j-1, t-1, qx, alpha, beta) + \
        (q * qx / beta) * hermite_gauss_e(i, j-1, t,   qx, alpha, beta) + \
        (t + 1)         * hermite_gauss_e(i, j-1, t+1, qx, alpha, beta);
    }   
}


double coulomb_auxiliary_r(const int t, const int u, const int v, const int n, const double p, const double pcx, const double pcy, const double pcz, const double rpc){
    double _T = p * rpc * rpc;
    double res = 0.;

    if (t == 0 && u == 0 && v == 0)
    {
        res = res + pow(-2. * p, n) * boys(n, _T);
    }
    else if (t == 0 && u == 0)
    {
        if (v >= 2)
        {
            res = res + (v - 1) * coulomb_auxiliary_r(t, u, v-2, n+1, p, pcx, pcy, pcz, rpc);
        }
        res = res + pcz * coulomb_auxiliary_r(t, u, v-1, n+1, p, pcx, pcy, pcz, rpc);
    }
    else if (t == 0)
    {
        if (u >= 2)
        {
            res = res + (u - 1) * coulomb_auxiliary_r(t, u-2, v, n+1, p, pcx, pcy, pcz, rpc);
        }
        res = res + pcy * coulomb_auxiliary_r(t, u-1, v, n+1, p, pcx, pcy, pcz, rpc);        
    }
    else
    {
        if (t >= 2)
        {
            res = res + (t - 1) * coulomb_auxiliary_r(t-2, u, v, n+1, p, pcx, pcy, pcz, rpc);
        }
        res = res + pcx * coulomb_auxiliary_r(t-1, u, v, n+1, p, pcx, pcy, pcz, rpc);
    }

    return res;
}


double eri_3d(const double alpha, const int* shell1, const double* center1, 
              const double beta,  const int* shell2, const double* center2, 
              const double gamma, const int* shell3, const double* center3, 
              const double delta, const int* shell4, const double* center4)
{
    const double PI = 3.14159265358979323846264338;
    int l1, m1, n1, l2, m2, n2, l3, m3, n3, l4, m4, n4;
    double p, q, PQ;
    double centerP[3], centerQ[3];
    double rpq;
    double res = 0.;

    l1 = shell1[0];
    m1 = shell1[1];
    n1 = shell1[2];

    l2 = shell2[0];
    m2 = shell2[1];
    n2 = shell2[2];

    l3 = shell3[0];
    m3 = shell3[1];
    n3 = shell3[2];

    l4 = shell4[0];
    m4 = shell4[1];
    n4 = shell4[2];

    p = alpha + beta;
    q = gamma + delta;
    PQ = p * q / (p + q);

    for (int i = 0; i < 3; i++)
    {
        centerP[i] = (alpha * center1[i] + beta  * center2[i]) / p;
        centerQ[i] = (gamma * center3[i] + delta * center4[i]) / q;
    }
    
    rpq = sqrt(pow(centerP[0] - centerQ[0], 2.) + pow(centerP[1] - centerQ[1], 2.) + pow(centerP[2] - centerQ[2], 2.));

    #pragma omp parallel for collapse(6) reduction(+:res)
    for (int t = 0; t < l1+l2+1; t++)
    {
        for (int u = 0; u < m1+m2+1; u++)
        {
            for (int v = 0; v < n1+n2+1; v++)
            {
                for (int tau = 0; tau < l3+l4+1; tau++)
                {
                    for (int nu = 0; nu < m3+m4+1; nu++)
                    {
                        for (int phi = 0; phi < n3+n4+1; phi++)
                        {
                            int parity = (tau + nu + phi) & 1;
                            double sign = parity ? -1. : 1.;
                            res = res + hermite_gauss_e(l1, l2, t,   center1[0]-center2[0], alpha, beta)  * \
                                        hermite_gauss_e(m1, m2, u,   center1[1]-center2[1], alpha, beta)  * \
                                        hermite_gauss_e(n1, n2, v,   center1[2]-center2[2], alpha, beta)  * \
                                        hermite_gauss_e(l3, l4, tau, center3[0]-center4[0], gamma, delta) * \
                                        hermite_gauss_e(m3, m4, nu,  center3[1]-center4[1], gamma, delta) * \
                                        hermite_gauss_e(n3, n4, phi, center3[2]-center4[2], gamma, delta) * \
                                        /* pow(-1, tau+nu+phi) * \ */
                                        sign * \
                                        coulomb_auxiliary_r(t+tau, u+nu, v+phi, 0, PQ, centerP[0]-centerQ[0], centerP[1]-centerQ[1], centerP[2]-centerQ[2], rpq);
                        }
                        
                    }
                    
                }
                
            }
            
        }
        
    }
    
    res = res * 2. * pow(PI, 2.5) / (p * q * sqrt(p+q));
    return res;
}

