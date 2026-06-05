#!/usr/bin/bash

pw.x -i pw.scf.in-AlCl3 > pw.scf.out-AlCl3
pw.x -i pw.scf.in-BCl3  > pw.scf.out-BCl3
pw.x -i pw.scf.in-C2H2  > pw.scf.out-C2H2
pw.x -i pw.scf.in-C2H4  > pw.scf.out-C2H4
pw.x -i pw.scf.in-CH4   > pw.scf.out-CH4
pw.x -i pw.scf.in-CO    > pw.scf.out-CO
pw.x -i pw.scf.in-CO2   > pw.scf.out-CO2

pw.x -i pw.scf.in-H2O   > pw.scf.out-H2O
pw.x -i pw.scf.in-H2O2  > pw.scf.out-H2O2
pw.x -i pw.scf.in-HCN   > pw.scf.out-HCN
pw.x -i pw.scf.in-Li2   > pw.scf.out-Li2
pw.x -i pw.scf.in-LiH   > pw.scf.out-LiH
pw.x -i pw.scf.in-NaCl  > pw.scf.out-NaCl

pw.x -i pw.scf.in-NH3   > pw.scf.out-NH3
pw.x -i pw.scf.in-O2    > pw.scf.out-O2
pw.x -i pw.scf.in-PH3   > pw.scf.out-PH3
pw.x -i pw.scf.in-SO2   > pw.scf.out-SO2
