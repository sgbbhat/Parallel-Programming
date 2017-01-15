#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

__global__ void opt_2dhistoKernel(uint32_t *input_device, int input_size, uint32_t *device_bins)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

     if (device_bins[input_device[i]] <255)
        {
                atomicAdd(&(device_bins[input_device[i]]), 1);
        }
        i += stride;
        __syncthreads();
}


void opt_2dhisto(uint32_t *input, int height, int width, uint32_t *bins)
{
	// size of the input image
	int size = height * width;
	dim3 dimBlock(BLOCK_SIZE, 1,1);
	dim3 dimGrid( ((height * width +  dimBlock.x - 1) / dimBlock.x) ,1,1);

	cudaMemset(bins, 0, sizeof(uint32_t) * HISTO_WIDTH);
  
	int num_kernel_calls = (dimGrid.x)/GRID_SIZE_MAX;

	for(int i=0; i<num_kernel_calls; i++)
	{
		opt_2dhistoKernel<<<GRID_SIZE_MAX, dimBlock>>>(input, GRID_SIZE_MAX * BLOCK_SIZE, bins);
		cudaDeviceSynchronize(); 
        	// pass the next set of inputs (pointer, so just add the address)
        	input += GRID_SIZE_MAX * BLOCK_SIZE;      
        	// keep track of elements left to compute
        	// to be used when only one kernel call to be done
        	size      -= GRID_SIZE_MAX * BLOCK_SIZE;
        	dimGrid.x -= GRID_SIZE_MAX;
	}
 
	opt_2dhistoKernel<<<dimGrid, dimBlock>>>(input, size, bins);
	cudaDeviceSynchronize(); 	
}


uint8_t *AllocateDeviceMemory(int histo_width, int histo_height, int size_of_element)
{
  uint8_t *t_memory;
  cudaMalloc((void **)&t_memory, histo_width * histo_height * size_of_element);
  return t_memory;
}

void* AllocateDevice(size_t size){
	void* ret;
	cudaMalloc(&ret, size);
	return ret;
}


void CopyToDevice(uint32_t *device, uint32_t *host, uint32_t input_height, uint32_t input_width, int size_of_element)
{
  const size_t x_size_padded = (input_width + 128) & 0xFFFFFF80;
  size_t row_size = input_width * size_of_element;

  for(int i=0; i<input_height; i++)
     {
        cudaMemcpy(device, host, row_size, cudaMemcpyHostToDevice);
        device += input_width;
        host += (x_size_padded);
     }
}

void CopyToHost(uint32_t *host, uint32_t *device, int size)
{
    cudaMemcpy(host,device, size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < HISTO_WIDTH * HISTO_HEIGHT; i++)
        host[i] = host[i]>255?255:host[i];
}


void cuda_memset(uint32_t *ptr, uint32_t value, uint32_t byte_count)
{
  cudaMemset((void *)ptr, value, (size_t)byte_count);
}

void FreeDevice(void* D_device){
	cudaFree(D_device);
}

/* Include below the implementation of any other functions you need */

