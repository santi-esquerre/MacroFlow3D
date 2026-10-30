# MMS order study of the Dirichlet head solve (SF-42 N2)

Verdict D: **PASS** (threshold 1.80 over the two finest grid pairs; gated: head L2, head max and flux max at least 2 cells from the Dirichlet faces, interior flux L2).

## eps = 0.5

| ny | nx | mg | PCG it | conv | rel res | head L2 | head max (all cells) | head L2 (>=2 cells from Dirichlet faces) | head max (>=2 cells from Dirichlet faces) | interior flux L2 | flux max (>=2 cells from Dirichlet faces) | interior flux max (all interior faces) | Dirichlet faces L2 | Dirichlet faces max | first interior layer L2 | first interior layer max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 64 | 3 | 20 | True | 6.808e-18 | 2.341e-03 | 7.673e-03 | 2.415e-03 | 7.673e-03 | 6.744e-03 | 2.927e-02 | 2.927e-02 | 5.045e-03 | 1.771e-02 | 2.815e-03 | 1.664e-02 |
| 32 | 128 | 4 | 20 | True | 6.382e-17 | 5.819e-04 | 1.909e-03 | 5.909e-04 | 1.909e-03 | 1.678e-03 | 7.412e-03 | 7.412e-03 | 1.260e-03 | 4.357e-03 | 7.270e-04 | 4.222e-03 |
| 64 | 256 | 5 | 20 | True | 2.508e-16 | 1.453e-04 | 4.768e-04 | 1.464e-04 | 4.768e-04 | 4.190e-04 | 1.851e-03 | 1.851e-03 | 3.149e-04 | 1.097e-03 | 1.861e-04 | 1.079e-03 |
| 128 | 512 | 6 | 20 | True | 7.358e-16 | 3.630e-05 | 1.192e-04 | 3.644e-05 | 1.192e-04 | 1.047e-04 | 4.636e-04 | 4.636e-04 | 7.873e-05 | 2.747e-04 | 4.718e-05 | 2.723e-04 |
| 256 | 1024 | 7 | 20 | True | 1.755e-15 | 9.074e-06 | 2.979e-05 | 9.091e-06 | 2.979e-05 | 2.617e-05 | 1.159e-04 | 1.159e-04 | 1.968e-05 | 6.868e-05 | 1.188e-05 | 6.837e-05 |

Observed orders (consecutive pairs):

| metric | gated | 16-32 | 32-64 | 64-128 | 128-256 | rel. error (finest) |
|---|---|---|---|---|---|---|
| head L2 | yes | 2.01 | 2.00 | 2.00 | 2.00 | 3.489e-06 |
| head max (all cells) | no | 2.01 | 2.00 | 2.00 | 2.00 | 1.146e-05 |
| head L2 (>=2 cells from Dirichlet faces) | no | 2.03 | 2.01 | 2.01 | 2.00 | 3.497e-06 |
| head max (>=2 cells from Dirichlet faces) | yes | 2.01 | 2.00 | 2.00 | 2.00 | 1.146e-05 |
| interior flux L2 | yes | 2.01 | 2.00 | 2.00 | 2.00 | 2.270e-05 |
| flux max (>=2 cells from Dirichlet faces) | yes | 1.98 | 2.00 | 2.00 | 2.00 | 1.004e-04 |
| interior flux max (all interior faces) | no | 1.98 | 2.00 | 2.00 | 2.00 | 1.005e-04 |
| Dirichlet faces L2 | no | 2.00 | 2.00 | 2.00 | 2.00 | 1.639e-05 |
| Dirichlet faces max | no | 2.02 | 1.99 | 2.00 | 2.00 | 5.720e-05 |
| first interior layer L2 | no | 1.95 | 1.97 | 1.98 | 1.99 | 1.715e-05 |
| first interior layer max | no | 1.98 | 1.97 | 1.99 | 1.99 | 9.871e-05 |

## eps = 1

| ny | nx | mg | PCG it | conv | rel res | head L2 | head max (all cells) | head L2 (>=2 cells from Dirichlet faces) | head max (>=2 cells from Dirichlet faces) | interior flux L2 | flux max (>=2 cells from Dirichlet faces) | interior flux max (all interior faces) | Dirichlet faces L2 | Dirichlet faces max | first interior layer L2 | first interior layer max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 64 | 3 | 20 | True | 3.631e-18 | 2.599e-03 | 8.620e-03 | 2.676e-03 | 8.620e-03 | 8.894e-03 | 4.536e-02 | 4.536e-02 | 8.166e-03 | 3.337e-02 | 4.496e-03 | 3.052e-02 |
| 32 | 128 | 4 | 20 | True | 3.549e-17 | 6.456e-04 | 2.147e-03 | 6.549e-04 | 2.147e-03 | 2.210e-03 | 1.138e-02 | 1.138e-02 | 2.043e-03 | 8.337e-03 | 1.194e-03 | 7.972e-03 |
| 64 | 256 | 5 | 20 | True | 1.257e-16 | 1.611e-04 | 5.379e-04 | 1.623e-04 | 5.379e-04 | 5.515e-04 | 2.873e-03 | 2.873e-03 | 5.109e-04 | 2.086e-03 | 3.107e-04 | 2.035e-03 |
| 128 | 512 | 6 | 20 | True | 3.499e-16 | 4.027e-05 | 1.345e-04 | 4.041e-05 | 1.345e-04 | 1.378e-04 | 7.178e-04 | 7.178e-04 | 1.278e-04 | 5.214e-04 | 7.940e-05 | 5.149e-04 |
| 256 | 1024 | 7 | 20 | True | 8.303e-16 | 1.007e-05 | 3.362e-05 | 1.008e-05 | 3.362e-05 | 3.445e-05 | 1.795e-04 | 1.795e-04 | 3.194e-05 | 1.304e-04 | 2.008e-05 | 1.295e-04 |

Observed orders (consecutive pairs):

| metric | gated | 16-32 | 32-64 | 64-128 | 128-256 | rel. error (finest) |
|---|---|---|---|---|---|---|
| head L2 | yes | 2.01 | 2.00 | 2.00 | 2.00 | 3.871e-06 |
| head max (all cells) | no | 2.01 | 2.00 | 2.00 | 2.00 | 1.293e-05 |
| head L2 (>=2 cells from Dirichlet faces) | no | 2.03 | 2.01 | 2.01 | 2.00 | 3.879e-06 |
| head max (>=2 cells from Dirichlet faces) | yes | 2.01 | 2.00 | 2.00 | 2.00 | 1.293e-05 |
| interior flux L2 | yes | 2.01 | 2.00 | 2.00 | 2.00 | 2.636e-05 |
| flux max (>=2 cells from Dirichlet faces) | yes | 1.99 | 1.99 | 2.00 | 2.00 | 1.372e-04 |
| interior flux max (all interior faces) | no | 1.99 | 1.99 | 2.00 | 2.00 | 1.373e-04 |
| Dirichlet faces L2 | no | 2.00 | 2.00 | 2.00 | 2.00 | 2.370e-05 |
| Dirichlet faces max | no | 2.00 | 2.00 | 2.00 | 2.00 | 9.675e-05 |
| first interior layer L2 | no | 1.91 | 1.94 | 1.97 | 1.98 | 2.586e-05 |
| first interior layer max | no | 1.94 | 1.97 | 1.98 | 1.99 | 1.668e-04 |

## eps = 1.5

| ny | nx | mg | PCG it | conv | rel res | head L2 | head max (all cells) | head L2 (>=2 cells from Dirichlet faces) | head max (>=2 cells from Dirichlet faces) | interior flux L2 | flux max (>=2 cells from Dirichlet faces) | interior flux max (all interior faces) | Dirichlet faces L2 | Dirichlet faces max | first interior layer L2 | first interior layer max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 64 | 3 | 20 | True | 1.932e-18 | 3.029e-03 | 9.606e-03 | 3.112e-03 | 9.606e-03 | 1.266e-02 | 7.446e-02 | 7.446e-02 | 1.321e-02 | 5.911e-02 | 7.245e-03 | 5.339e-02 |
| 32 | 128 | 4 | 20 | True | 1.993e-17 | 7.519e-04 | 2.405e-03 | 7.620e-04 | 2.405e-03 | 3.140e-03 | 1.839e-02 | 1.839e-02 | 3.303e-03 | 1.499e-02 | 1.941e-03 | 1.423e-02 |
| 64 | 256 | 5 | 20 | True | 7.109e-17 | 1.877e-04 | 6.035e-04 | 1.889e-04 | 6.035e-04 | 7.836e-04 | 4.610e-03 | 4.610e-03 | 8.262e-04 | 3.734e-03 | 5.075e-04 | 3.631e-03 |
| 128 | 512 | 6 | 20 | True | 1.938e-16 | 4.690e-05 | 1.508e-04 | 4.705e-05 | 1.508e-04 | 1.958e-04 | 1.153e-03 | 1.153e-03 | 2.066e-04 | 9.347e-04 | 1.300e-04 | 9.216e-04 |
| 256 | 1024 | 7 | 20 | True | 4.650e-16 | 1.172e-05 | 3.770e-05 | 1.174e-05 | 3.770e-05 | 4.895e-05 | 2.883e-04 | 2.883e-04 | 5.164e-05 | 2.337e-04 | 3.293e-05 | 2.320e-04 |

Observed orders (consecutive pairs):

| metric | gated | 16-32 | 32-64 | 64-128 | 128-256 | rel. error (finest) |
|---|---|---|---|---|---|---|
| head L2 | yes | 2.01 | 2.00 | 2.00 | 2.00 | 4.508e-06 |
| head max (all cells) | no | 2.00 | 1.99 | 2.00 | 2.00 | 1.450e-05 |
| head L2 (>=2 cells from Dirichlet faces) | no | 2.03 | 2.01 | 2.01 | 2.00 | 4.517e-06 |
| head max (>=2 cells from Dirichlet faces) | yes | 2.00 | 1.99 | 2.00 | 2.00 | 1.450e-05 |
| interior flux L2 | yes | 2.01 | 2.00 | 2.00 | 2.00 | 3.123e-05 |
| flux max (>=2 cells from Dirichlet faces) | yes | 2.02 | 2.00 | 2.00 | 2.00 | 1.837e-04 |
| interior flux max (all interior faces) | no | 2.02 | 2.00 | 2.00 | 2.00 | 1.839e-04 |
| Dirichlet faces L2 | no | 2.00 | 2.00 | 2.00 | 2.00 | 3.295e-05 |
| Dirichlet faces max | no | 1.98 | 2.01 | 2.00 | 2.00 | 1.491e-04 |
| first interior layer L2 | no | 1.90 | 1.94 | 1.96 | 1.98 | 3.649e-05 |
| first interior layer max | no | 1.91 | 1.97 | 1.98 | 1.99 | 2.571e-04 |

## eps = 2

| ny | nx | mg | PCG it | conv | rel res | head L2 | head max (all cells) | head L2 (>=2 cells from Dirichlet faces) | head max (>=2 cells from Dirichlet faces) | interior flux L2 | flux max (>=2 cells from Dirichlet faces) | interior flux max (all interior faces) | Dirichlet faces L2 | Dirichlet faces max | first interior layer L2 | first interior layer max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 64 | 3 | 20 | True | 1.335e-18 | 3.639e-03 | 1.059e-02 | 3.735e-03 | 1.059e-02 | 1.874e-02 | 1.224e-01 | 1.224e-01 | 2.095e-02 | 1.002e-01 | 1.147e-02 | 9.002e-02 |
| 32 | 128 | 4 | 20 | True | 1.171e-17 | 9.034e-04 | 2.658e-03 | 9.150e-04 | 2.658e-03 | 4.640e-03 | 3.066e-02 | 3.066e-02 | 5.236e-03 | 2.579e-02 | 3.077e-03 | 2.437e-02 |
| 64 | 256 | 5 | 20 | True | 3.855e-17 | 2.255e-04 | 6.676e-04 | 2.269e-04 | 6.676e-04 | 1.158e-03 | 7.693e-03 | 7.693e-03 | 1.309e-03 | 6.425e-03 | 8.055e-04 | 6.230e-03 |
| 128 | 512 | 6 | 20 | True | 1.084e-16 | 5.634e-05 | 1.669e-04 | 5.652e-05 | 1.669e-04 | 2.892e-04 | 1.926e-03 | 1.926e-03 | 3.273e-04 | 1.608e-03 | 2.065e-04 | 1.583e-03 |
| 256 | 1024 | 7 | 20 | True | 2.848e-16 | 1.408e-05 | 4.173e-05 | 1.411e-05 | 4.173e-05 | 7.230e-05 | 4.818e-04 | 4.818e-04 | 8.184e-05 | 4.020e-04 | 5.232e-05 | 3.989e-04 |

Observed orders (consecutive pairs):

| metric | gated | 16-32 | 32-64 | 64-128 | 128-256 | rel. error (finest) |
|---|---|---|---|---|---|---|
| head L2 | yes | 2.01 | 2.00 | 2.00 | 2.00 | 5.416e-06 |
| head max (all cells) | no | 1.99 | 1.99 | 2.00 | 2.00 | 1.605e-05 |
| head L2 (>=2 cells from Dirichlet faces) | no | 2.03 | 2.01 | 2.01 | 2.00 | 5.426e-06 |
| head max (>=2 cells from Dirichlet faces) | yes | 1.99 | 1.99 | 2.00 | 2.00 | 1.605e-05 |
| interior flux L2 | yes | 2.01 | 2.00 | 2.00 | 2.00 | 3.686e-05 |
| flux max (>=2 cells from Dirichlet faces) | yes | 2.00 | 1.99 | 2.00 | 2.00 | 2.453e-04 |
| interior flux max (all interior faces) | no | 2.00 | 1.99 | 2.00 | 2.00 | 2.456e-04 |
| Dirichlet faces L2 | no | 2.00 | 2.00 | 2.00 | 2.00 | 4.346e-05 |
| Dirichlet faces max | no | 1.96 | 2.01 | 2.00 | 2.00 | 2.135e-04 |
| first interior layer L2 | no | 1.90 | 1.93 | 1.96 | 1.98 | 4.831e-05 |
| first interior layer max | no | 1.89 | 1.97 | 1.98 | 1.99 | 3.684e-04 |

## Algebraic-error control (rtol 1e-12 against 1e-10, finest grid)

- ny=256 eps=0.5 rtol=1e-12: max relative change 0.00e+00 (first interior layer max), negligible
- ny=256 eps=1.5 rtol=1e-12: max relative change 0.00e+00 (first interior layer max), negligible
- ny=256 eps=1 rtol=1e-12: max relative change 0.00e+00 (first interior layer max), negligible
- ny=256 eps=2 rtol=1e-12: max relative change 0.00e+00 (first interior layer max), negligible
