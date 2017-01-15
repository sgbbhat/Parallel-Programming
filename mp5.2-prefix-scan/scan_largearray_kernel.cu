#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

float ** BlockSums;
float ** BlockSumsSummed;
float ** HostSums;


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
// MP4.2 - You can use any other block size you wish.
#define BLOCK_SIZE 256

// MP4.2 - Host Helper Functions (allocate your own data structure...)


#ifdef ZERO_BANK_CONFLICTS 
#define CONFLICT_FREE_OFFSET(n) \ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
#else 
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS) 
#endif

// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.

//Kernel function for prefix exclusive scan algorithm 

__global__ void kernal_scan(float* d_odata,float* d_idata, int numElements,float* auxArray)
{
	__shared__ float scan_array[2*BLOCK_SIZE+(2*BLOCK_SIZE/16)];
	unsigned int t=threadIdx.x;
	unsigned int start=2*blockDim.x*blockIdx.x;
	if(blockIdx.x==gridDim.x-1)
	{
		if(numElements%2*BLOCK_SIZE==0)
		{                       
			scan_array[t+CONFLICT_FREE_OFFSET(t)]=d_idata[t+start];
			scan_array[t+blockDim.x+CONFLICT_FREE_OFFSET(t+blockDim.x)]=d_idata[t+blockDim.x+start];
        	}
        	else
		{
			if (numElements%(2*BLOCK_SIZE)>255)		
			{
				if(t < numElements%BLOCK_SIZE)       
				{
					scan_array[t+CONFLICT_FREE_OFFSET(t)]=d_idata[t+start];
					scan_array[t+blockDim.x+CONFLICT_FREE_OFFSET(t+blockDim.x)]=d_idata[t+blockDim.x+start];
				}
				else                                
				{
					scan_array[t+CONFLICT_FREE_OFFSET(t)]=d_idata[t+start];
					scan_array[t+blockDim.x+CONFLICT_FREE_OFFSET(t+blockDim.x)]=0.0f;
				}
			}
			else
			{
				if(t < numElements % BLOCK_SIZE)
				{
					scan_array[t+CONFLICT_FREE_OFFSET(t)]=d_idata[t+start];
					scan_array[t+blockDim.x+CONFLICT_FREE_OFFSET(t+blockDim.x)]=0.0f;
				}
				else
				{
					scan_array[t+CONFLICT_FREE_OFFSET(t)]=0.0f;                               
					scan_array[t+blockDim.x+CONFLICT_FREE_OFFSET(t+blockDim.x)]=0.0f;
				}
			}
		}
	}
	else
	{
		scan_array[t+CONFLICT_FREE_OFFSET(t)]=d_idata[t+start];
		scan_array[t+blockDim.x+CONFLICT_FREE_OFFSET(t+blockDim.x)]=d_idata[t+blockDim.x+start];
	}
	
	// Reduction algorithm
	__syncthreads();
	int stride = 1;
	while(stride <= BLOCK_SIZE)
	{
		int index = (threadIdx.x+1)*stride*2 - 1;
		if(index < 2*BLOCK_SIZE)
		scan_array[index+CONFLICT_FREE_OFFSET(index)] += scan_array[index-stride + CONFLICT_FREE_OFFSET(index-stride)];
		stride = stride*2;
		__syncthreads();
	}
	
	//Postreduction steps
	if (threadIdx.x==0)
		scan_array[2*blockDim.x-1+CONFLICT_FREE_OFFSET(2*blockDim.x-1)] = 0;
	stride = BLOCK_SIZE;
	while(stride > 0) 
	{
		int index = (threadIdx.x+1)*stride*2 - 1;
		if(index < 2* BLOCK_SIZE) 
		{
			float temp = scan_array[index+CONFLICT_FREE_OFFSET(index)];
			scan_array[index+CONFLICT_FREE_OFFSET(index)] += scan_array[index-stride+CONFLICT_FREE_OFFSET(index-stride)];
			scan_array[index-stride+CONFLICT_FREE_OFFSET(index-stride)] = temp; 
   		} 
		stride = stride / 2;
		__syncthreads(); 
	}

	if ((start + t) < numElements)  
		d_odata[start + t] = scan_array[t+CONFLICT_FREE_OFFSET(t)];

        if ((start + blockDim.x + t) < numElements)
		d_odata[start + blockDim.x + t] = scan_array[blockDim.x + t+CONFLICT_FREE_OFFSET(t+blockDim.x)];
	__syncthreads();

	// Making the first thread calculate the elements for auxillary array
	if(t==0)
	{
		if(numElements%(2*BLOCK_SIZE)!=0)
		{	
			if(blockIdx.x == (gridDim.x-1))
			{	
				auxArray[blockIdx.x]= d_odata[start+2*blockDim.x-1]+ d_idata[start+(numElements%(2*BLOCK_SIZE))-1];		
			}
			else
			{
				auxArray[blockIdx.x]=d_odata[start+2*blockDim.x-1]+ d_idata[start+2*blockDim.x-1];
			}
		}
		else
		{ 
			auxArray[blockIdx.x]=d_odata[start+2*blockDim.x-1]+ d_idata[start+2*blockDim.x-1];
		} 
	}	
}

// Kernel function for vector addition
__global__ void kernal_constant_addition(float *array_n,float *array_n_minus_1)
{
	int t = threadIdx.x;
	unsigned int start = 2*blockDim.x*blockIdx.x;
	float add_element;
	add_element = array_n[blockIdx.x];
	array_n_minus_1[start + t]=array_n_minus_1[start + t]+add_element;
	array_n_minus_1[start + blockDim.x + t] = array_n_minus_1[start + blockDim.x + t]+add_element;
}

void prescanArray(float *outArray, float *inArray, int numElements)
{
	dim3 block_dim(BLOCK_SIZE,1,1);
	dim3 grid_dim1(ceil((float)numElements/(BLOCK_SIZE*2)),1,1);
	float *auxArray;
	cudaMalloc((void**)&auxArray,(grid_dim1.x)*sizeof(float));       
	kernal_scan<<<grid_dim1,block_dim>>>(outArray,inArray,numElements,auxArray); 
	cudaDeviceSynchronize();
     
        if(grid_dim1.x > 1)
	{
		int new_numElements=grid_dim1.x;
		dim3 grid_dim2(ceil((float)new_numElements/(BLOCK_SIZE*2)),1,1);
		float *auxArray1, *outArray1;
		cudaMalloc((void**)&auxArray1,(grid_dim2.x)*sizeof(float));
		cudaMalloc((void**)&outArray1,(new_numElements)*sizeof(float));
		kernal_scan<<<grid_dim2,block_dim>>>(outArray1,auxArray,new_numElements,auxArray1); 
		cudaDeviceSynchronize();
		if(grid_dim2.x > 1) 
		{
			int new_numElements1=grid_dim2.x;
			dim3 grid_dim3(ceil((float)new_numElements1/(BLOCK_SIZE*2)),1,1);
			float *auxArray2, *outArray2;
			cudaMalloc((void**)&auxArray2,(grid_dim3.x)*sizeof(float));
			cudaMalloc((void**)&outArray2,(new_numElements1)*sizeof(float));
	   		kernal_scan<<<grid_dim3,block_dim>>>(outArray2,auxArray1,new_numElements1,auxArray2);
	   		cudaDeviceSynchronize();
			kernal_constant_addition<<<grid_dim2,block_dim>>>(outArray2,outArray1);
			cudaDeviceSynchronize();
		}
		kernal_constant_addition<<<grid_dim1,block_dim>>>(outArray1,outArray);
	}
	cudaDeviceSynchronize();
}
#endif // _PRESCAN_CU_
