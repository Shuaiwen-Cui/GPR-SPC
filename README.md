# GPR-SPC
Gaussian Progress Regression and Stochastic Process Control for monitoring data 

## [01-Python Version]

1. open the file in Jupyter Notebook or Vscode and run the code
2. the file will load data from '.mat' files
3. results will be printed out and visualized

## [02-C Version]

1. check the instructions at the beginning of the code
2. note to install dependencies
3. some results will be output to files
4. visualization is done by python script 'visualization.ipynb', so complie GPRSPC.c and run the executable first and use 'visualization.ipynb' to checkout the results

## [03-ARM-Cortex Version]

This code is not the version used for Xnode, but can be easily transplanted to Xnode. This version can be run on FK723M1-ZGT6 from FANKE technology, which bears a STM32H723ZGT6 chip.

Current version only has the GPR-SPC part.

## [04-Sensitivity Analysis]

Use a truss structure to simulate timber trestle bridge to analyze the sensitivity of the proposed damage detection framework.

Refer to the `analyzing` subfolder for more details.

- archive-robust: more robust setup
- archive-sensitivity: more sensitive setup (presented in the paper)

## Verification(deprecated)

deprecated, use the newer one to check the verification results. (Python on PC vs C on Arm-Cortex device)

## Verification-GraphMaking

compare the results of Python and C version, and make graphs to show the difference. The comparison in the paper can be found here.

[SENSOR]
- sensor print out is in the 'LOG' folder.
- source code and project is in 'ARM-Cortext Version/Core/GPR-SPC/GPR-SPC.c'
[PC]
- PC print out is in 'Datamanipulation' folder, check 'Graph-Gauss-Jordan.ipynb.ipynb'
