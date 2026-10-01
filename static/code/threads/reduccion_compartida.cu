#include <stdio.h>
#include <stdlib.h>

__global__ void reduccion_memoria_compartida(float *salida, const float *data,
                                             unsigned int N) {
  unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

  // declaramos memoria compartida
  extern __shared__ float s_data[];

  // copiamos los datos de la memoria global a la memoria compartida
  s_data[threadIdx.x] = (idx_x < N) ? data[idx_x] : 0.f;

  // sincronizamos para asegurar que todos los datos estén copiados
  // antes de comenzar con el cálculo
  __syncthreads();

  // calcular la reducción, incrementando el stride hasta que es igual
  // a (uno menos) el tamaño de los bloques
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    // un subconjunto de threads suman sus valores con otro valor
    // a una distancia "stride" del thread actual.
    if ((idx_x % (stride * 2)) == 0)
      s_data[threadIdx.x] += s_data[threadIdx.x + stride];

    // Ya que la suma se realiza en la memoria compartida
    // hay que sincronizar para tener todo actualizado
    __syncthreads();
  }

  // Primer thread de cada bloque guarda el resultado del bloque en OTRO
  // arreglo, usando el índice del **bloque**. Escribirlo en data[blockIdx.x]
  // pisaría datos que otro bloque todavía no copia a su memoria compartida.
  if (threadIdx.x == 0)
    salida[blockIdx.x] = s_data[0];
}

void inicializar_numeros(float *data, int size) {
  for (int i = 0; i < size; i++) {
    // Generamos números aleatorios pequeños
    // El "0xFF" es un número en hexadecimal y está combinado con
    // rand() usando el "bitwise operator" & para obtener números
    // entre 0 y 255.
    data[i] = (float)(rand() & 0xFF) / (float)RAND_MAX;
  }
}

float resultado_cpu(float *data, int N) {
  float resultado = 0.f;
  for (int i = 0; i < N; i++) {
    resultado += data[i];
  }

  return resultado;
}

int main() {
  float *h_array;
  float *d_array, *d_salida;

  unsigned int N = 1 << 25;
  unsigned int Nt = N;
  unsigned int n_threads = 1024;

  float resultado_host, resultado_gpu;

  srand(2019);

  // Asignar memoria en el host
  h_array = (float *)malloc(N * sizeof(float));

  // Iniciliazar valores de h_array con números aleatorios
  inicializar_numeros(h_array, N);

  // Asignar memoria en el GPU y copiar datos
  cudaMalloc((void **)&d_array, N * sizeof(float));
  cudaMalloc((void **)&d_salida, (N + n_threads - 1) / n_threads * sizeof(float));
  cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

  // Calcular reducción en el GPU: cada pasada lee de "entrada" y escribe un
  // resultado por bloque en "salida"; después se intercambian los punteros
  float *entrada = d_array, *salida = d_salida;
  while (N > 1) {
    int n_bloques = (N + n_threads - 1) / n_threads;
    reduccion_memoria_compartida<<<n_bloques, n_threads,
                                   n_threads * sizeof(float), 0>>>(salida,
                                                                   entrada, N);
    N = n_bloques;
    float *tmp = entrada;
    entrada = salida;
    salida = tmp;
  }

  // Copiar resultado del GPU
  cudaMemcpy(&resultado_gpu, &entrada[0], sizeof(float),
             cudaMemcpyDeviceToHost);

  // Calcular reducción en el CPU (secuencial)
  // Ojo: usamos Nt ya que N está modificado en la invocación del kernel
  resultado_host = resultado_cpu(h_array, Nt);
  printf("host: %f, device %f\n", resultado_host, resultado_gpu);

  // Liberar memoria
  cudaFree(d_array);
  cudaFree(d_salida);
  free(h_array);

  return 0;
}
