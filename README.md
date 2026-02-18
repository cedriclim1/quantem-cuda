# quantem-cuda
repository for developing custom CUDA kernels that are meant to work in conjunction with the `electronmicroscopoy/quantem` repo. This repo is intended to hold optional, additional functions that support and extend the main `quantem` code--it is not meant as an optional dependency that is required or called anywhere with `quantem` itself. 

This repo is also a place for us to develop and test various methods for implementing custom kernels. There is existing code that uses `CuPy` [RawKernel](https://docs.cupy.dev/en/stable/reference/kernel.html), but we would prefer to have entirely torch-native solutions that don't require additional dependencies, as then we could easily include the code in the main repo. 

Long term, we would also like to support writing hardcoded custom gradients in `quantem`, e.g. as is done in [gsplat](https://github.com/nerfstudio-project/gsplat), but this will likely be a bit tricky to integrate with our current packaging. 
