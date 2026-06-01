// #include <cuda_runtime.h>
#include <stdio.h>

__global__ void mostrarIndices(void) {
  printf("Device: threadIdx: (%d, %d, %d)  blockIdx: (%d, %d, %d)  "
         "blockDim: (%d, %d, %d)  gridDim: (%d, %d, %d)\n",
         threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y,
         blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y,
         gridDim.z);
}

int main(int argc, char **argv) {

  // definir número de threads por bloque, número de bloques
  dim3 block(3, 3);
  dim3 grid(2, 2);

  // mostrar dimensiones del lado del host
  printf("Host: (grid.x , grid.y , grid.z ) = (%d, %d, %d)\n", grid.x, grid.y,
         grid.z);
  printf("Host: (block.x, block.y, block.z) = (%d, %d, %d)\n", block.x, block.y,
         block.z);

  // mostrar indices y dimensiones del lado del GPU
  mostrarIndices<<<grid, block>>>();

  // reinicializar el device (limpiar)
  cudaDeviceReset();

  return (0);
}
