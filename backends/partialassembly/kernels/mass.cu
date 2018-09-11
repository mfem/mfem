
extern "C" __global__ void evector_kernel(int size, const double* U, double* V){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) 
        V[i] += U[i];
}