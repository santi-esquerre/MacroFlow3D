# SF-42 N3 pseudo-spectral cross-check of the FV velocity

Relative L2 error of the FV face fluxes (all faces) against the exact face averages of the pseudo-spectral reference (finer of the pair M, 2M); `selfconv` = reference M against 2M on the same faces.


## sigma^2 = 0.25 (eps = 0.5), reference pair M = 64 / 128


### seed 420001 (FV rtol 1e-12; applied-scale spread over the ladder 8.82e-16)

| N | lambda/h | err all | err x | selfconv all | FV/selfconv | order (all) | FV its | gen. match max\|dY\|/std |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 9.959e-03 | 9.111e-03 | 4.554e-10 | 21869623.8 |  | 10/10/10 | 1.3e-15 |
| 32 | 8 | 2.522e-03 | 2.307e-03 | 4.802e-10 | 5250966.3 | 1.98 | 10/10/10 | 2.7e-15 |
| 40 | 10 | 1.616e-03 | 1.479e-03 | 4.810e-10 | 3360362.3 | 1.99 | 10/10/10 | 2.2e-15 |
| 64 | 16 | 6.325e-04 | 5.788e-04 | 4.896e-10 | 1291945.4 | 2.00 | 20/20/20 | 2.2e-15 |
| 128 | 32 | 1.582e-04 | 1.448e-04 | 4.923e-10 | 321426.4 | 2.00 | 20/20/20 | 2.7e-15 |
| 256 | 64 | 3.957e-05 | 3.621e-05 | 4.931e-10 | 80254.8 | 2.00 | 20/20/20 | 5.3e-15 |

verdict E: monotone=True, order(finest pair)=2.00 (>= 1.5: True), min FV/selfconv=80254.8 (>= 10: True), FV converged=True, complete ladder=True => **PASS**

### seed 420002 (FV rtol 1e-12; applied-scale spread over the ladder 3.67e-16)

| N | lambda/h | err all | err x | selfconv all | FV/selfconv | order (all) | FV its | gen. match max\|dY\|/std |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 6.882e-03 | 5.888e-03 | 3.027e-11 | 227346871.3 |  | 10/10/10 | 8.9e-16 |
| 32 | 8 | 1.739e-03 | 1.489e-03 | 3.274e-11 | 53110144.0 | 1.98 | 10/10/10 | 1.3e-15 |
| 40 | 10 | 1.114e-03 | 9.539e-04 | 3.317e-11 | 33595784.4 | 1.99 | 10/10/10 | 1.8e-15 |
| 64 | 16 | 4.359e-04 | 3.732e-04 | 3.371e-11 | 12930894.2 | 2.00 | 10/10/10 | 2.7e-15 |
| 128 | 32 | 1.091e-04 | 9.336e-05 | 3.400e-11 | 3207828.4 | 2.00 | 20/20/20 | 2.2e-15 |
| 256 | 64 | 2.727e-05 | 2.334e-05 | 3.407e-11 | 800343.5 | 2.00 | 20/20/20 | 2.7e-15 |

verdict E: monotone=True, order(finest pair)=2.00 (>= 1.5: True), min FV/selfconv=800343.5 (>= 10: True), FV converged=True, complete ladder=True => **PASS**

sigma^2 = 0.25: PASS


## sigma^2 = 1 (eps = 1), reference pair M = 64 / 128


### seed 420001 (FV rtol 1e-12; applied-scale spread over the ladder 8.82e-16)

| N | lambda/h | err all | err x | selfconv all | FV/selfconv | order (all) | FV its | gen. match max\|dY\|/std |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 2.641e-02 | 2.477e-02 | 3.065e-08 | 861705.1 |  | 20/20/20 | 1.3e-15 |
| 32 | 8 | 6.731e-03 | 6.306e-03 | 3.255e-08 | 206819.7 | 1.97 | 20/20/20 | 2.7e-15 |
| 40 | 10 | 4.318e-03 | 4.045e-03 | 3.201e-08 | 134891.5 | 1.99 | 20/20/20 | 2.2e-15 |
| 64 | 16 | 1.691e-03 | 1.584e-03 | 3.328e-08 | 50813.5 | 1.99 | 20/20/20 | 2.2e-15 |
| 128 | 32 | 4.233e-04 | 3.964e-04 | 3.350e-08 | 12636.0 | 2.00 | 20/20/20 | 2.7e-15 |
| 256 | 64 | 1.059e-04 | 9.914e-05 | 3.356e-08 | 3154.6 | 2.00 | 20/20/20 | 5.3e-15 |

verdict E: monotone=True, order(finest pair)=2.00 (>= 1.5: True), min FV/selfconv=3154.6 (>= 10: True), FV converged=True, complete ladder=True => **PASS**

### seed 420002 (FV rtol 1e-12; applied-scale spread over the ladder 3.67e-16)

| N | lambda/h | err all | err x | selfconv all | FV/selfconv | order (all) | FV its | gen. match max\|dY\|/std |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 1.708e-02 | 1.486e-02 | 5.584e-09 | 3059125.3 |  | 20/20/20 | 8.9e-16 |
| 32 | 8 | 4.358e-03 | 3.793e-03 | 6.038e-09 | 721680.7 | 1.97 | 20/20/20 | 1.3e-15 |
| 40 | 10 | 2.796e-03 | 2.433e-03 | 6.110e-09 | 457555.6 | 1.99 | 10/10/10 | 1.8e-15 |
| 64 | 16 | 1.095e-03 | 9.531e-04 | 6.216e-09 | 176155.9 | 1.99 | 20/20/20 | 2.7e-15 |
| 128 | 32 | 2.741e-04 | 2.386e-04 | 6.269e-09 | 43726.7 | 2.00 | 20/20/20 | 2.2e-15 |
| 256 | 64 | 6.855e-05 | 5.967e-05 | 6.283e-09 | 10911.3 | 2.00 | 20/20/20 | 2.7e-15 |

verdict E: monotone=True, order(finest pair)=2.00 (>= 1.5: True), min FV/selfconv=10911.3 (>= 10: True), FV converged=True, complete ladder=True => **PASS**

sigma^2 = 1: PASS


## sigma^2 = 2.25 (eps = 1.5), reference pair M = 96 / 192


### seed 420001 (FV rtol 1e-12; applied-scale spread over the ladder 8.82e-16)

| N | lambda/h | err all | err x | selfconv all | FV/selfconv | order (all) | FV its | gen. match max\|dY\|/std |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 4.622e-02 | 4.406e-02 | 5.170e-11 | 894078824.4 |  | 20/20/20 | 1.3e-15 |
| 32 | 8 | 1.183e-02 | 1.125e-02 | 5.676e-11 | 208405330.4 | 1.97 | 20/20/20 | 2.7e-15 |
| 40 | 10 | 7.593e-03 | 7.222e-03 | 5.743e-11 | 132228538.0 | 1.99 | 20/20/20 | 2.2e-15 |
| 64 | 16 | 2.976e-03 | 2.829e-03 | 5.863e-11 | 50758233.3 | 1.99 | 20/20/20 | 2.2e-15 |
| 128 | 32 | 7.451e-04 | 7.083e-04 | 5.913e-11 | 12601129.1 | 2.00 | 20/20/20 | 2.7e-15 |
| 256 | 64 | 1.863e-04 | 1.771e-04 | 5.931e-11 | 3142188.0 | 2.00 | 20/20/20 | 5.3e-15 |

verdict E: monotone=True, order(finest pair)=2.00 (>= 1.5: True), min FV/selfconv=3142188.0 (>= 10: True), FV converged=True, complete ladder=True => **PASS**

### seed 420002 (FV rtol 1e-12; applied-scale spread over the ladder 3.67e-16)

| N | lambda/h | err all | err x | selfconv all | FV/selfconv | order (all) | FV its | gen. match max\|dY\|/std |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 2.863e-02 | 2.551e-02 | 5.430e-12 | 5272255048.6 |  | 20/20/20 | 8.9e-16 |
| 32 | 8 | 7.376e-03 | 6.576e-03 | 6.138e-12 | 1201618280.9 | 1.96 | 20/20/20 | 1.3e-15 |
| 40 | 10 | 4.738e-03 | 4.225e-03 | 6.256e-12 | 757455388.3 | 1.98 | 20/20/20 | 1.8e-15 |
| 64 | 16 | 1.859e-03 | 1.657e-03 | 6.417e-12 | 289644673.5 | 1.99 | 20/20/20 | 2.7e-15 |
| 128 | 32 | 4.656e-04 | 4.152e-04 | 6.513e-12 | 71491027.6 | 2.00 | 20/20/20 | 2.2e-15 |
| 256 | 64 | 1.165e-04 | 1.038e-04 | 6.539e-12 | 17808312.8 | 2.00 | 20/20/20 | 2.7e-15 |

verdict E: monotone=True, order(finest pair)=2.00 (>= 1.5: True), min FV/selfconv=17808312.8 (>= 10: True), FV converged=True, complete ladder=True => **PASS**

sigma^2 = 2.25: PASS


## sigma^2 = 4 (eps = 2), reference pair M = 128 / 256


### seed 420001 (FV rtol 1e-12; applied-scale spread over the ladder 8.82e-16)

| N | lambda/h | err all | err x | selfconv all | FV/selfconv | order (all) | FV its | gen. match max\|dY\|/std |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 6.584e-02 | 6.302e-02 | 2.041e-13 | 322646350001.6 |  | 20/20/20 | 1.3e-15 |
| 32 | 8 | 1.695e-02 | 1.619e-02 | 2.299e-13 | 73736212975.2 | 1.96 | 20/20/20 | 2.7e-15 |
| 40 | 10 | 1.089e-02 | 1.040e-02 | 2.377e-13 | 45803040611.8 | 1.98 | 20/20/20 | 2.2e-15 |
| 64 | 16 | 4.269e-03 | 4.075e-03 | 2.565e-13 | 16646064604.7 | 1.99 | 20/20/20 | 2.2e-15 |
| 128 | 32 | 1.069e-03 | 1.021e-03 | 3.034e-13 | 3525107844.8 | 2.00 | 20/20/20 | 2.7e-15 |
| 256 | 64 | 2.675e-04 | 2.553e-04 | 3.540e-13 | 755638704.3 | 2.00 | 20/20/20 | 5.3e-15 |

verdict E: monotone=True, order(finest pair)=2.00 (>= 1.5: True), min FV/selfconv=755638704.3 (>= 10: True), FV converged=True, complete ladder=True => **PASS**

### seed 420002 (FV rtol 1e-12; applied-scale spread over the ladder 3.67e-16)

| N | lambda/h | err all | err x | selfconv all | FV/selfconv | order (all) | FV its | gen. match max\|dY\|/std |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 4.034e-02 | 3.684e-02 | 8.235e-13 | 48977259553.4 |  | 20/20/20 | 8.9e-16 |
| 32 | 8 | 1.051e-02 | 9.608e-03 | 7.760e-13 | 13537292795.6 | 1.94 | 20/20/20 | 1.3e-15 |
| 40 | 10 | 6.759e-03 | 6.182e-03 | 7.830e-13 | 8631583795.0 | 1.98 | 20/20/20 | 1.8e-15 |
| 64 | 16 | 2.655e-03 | 2.429e-03 | 8.367e-13 | 3173520841.0 | 1.99 | 20/20/20 | 2.7e-15 |
| 128 | 32 | 6.657e-04 | 6.091e-04 | 9.497e-13 | 700932897.7 | 2.00 | 20/20/20 | 2.2e-15 |
| 256 | 64 | 1.665e-04 | 1.524e-04 | 1.125e-12 | 147995417.2 | 2.00 | 20/20/20 | 2.7e-15 |

verdict E: monotone=True, order(finest pair)=2.00 (>= 1.5: True), min FV/selfconv=147995417.2 (>= 10: True), FV converged=True, complete ladder=True => **PASS**

sigma^2 = 4: PASS

