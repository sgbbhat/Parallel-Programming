#ifndef OPT_KERNEL
#define OPT_KERNEL

#define BLOCK_SIZE 512

#define GRID_SIZE_MAX 65535

void opt_2dhisto(uint32_t *input, int input_height, int input_width, uint32_t *device_bins);

/* Include below the function headers of any other functions that you implement */

uint8_t *AllocateDeviceMemory(int histo_width, int histo_height, int size_of_element);

void CopyToDevice(uint32_t *device, uint32_t *host, uint32_t input_height, uint32_t input_width, int size_of_element);

void CopyToHost(uint32_t *host, uint32_t *device, int size);


void* AllocateDevice(size_t size);

void CopyToDevice(void* D_device, void* D_host, size_t size);

void CopyFromDevice(void* D_host, void* D_device, size_t size);

void FreeDevice(void* D_device); 

void cuda_memset(uint32_t *ptr, uint32_t value, uint32_t byte_count);

/* Include below the function headers of any other functions that you implement */


#endif
