//#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "cudnn.h"

/*
一个典型的CUDA编程结构包括5个主要步骤。 
1.分配GPU内存。 
2.从CPU内存中拷贝数据到GPU内存。
3.调用CUDA内核函数来完成程序指定的运算。 
4.将数据从GPU拷回CPU内存。 
5.释放GPU内存空间。
*/


void SumArraysOnHost(float* A, float* B, float* C, const int N)
{
	for (int idx = 0; idx < N; ++idx) {
		C[idx] = A[idx] + B[idx];
	}
}


void InitialData(float* dataPtr, int size) 
{
	// generate different seed for random number
	time_t t;
	srand((unsigned int)time(&t));

	for (int idx = 0; idx < size; ++idx) {
		dataPtr[idx] = (float)(rand() & 0xFF) / 10.0f;
	}
}


// https://developer.nvidia.com/blog/even-easier-introduction-cuda/
__global__ void SumArrayOnDevice(float* A, float* B, float *C)
{
	//for (int idx = 0; idx < N; ++idx) {
	//	C[idx] = A[idx] + B[idx];
	//}
	int idx = threadIdx.x;
	C[idx] = A[idx] + B[idx];
}

void checkResult(float* hostData, float* deviceData, const int N)
{
	double epsilon = 1.0E-8;
	int match = 1;
	for (int idx = 0; idx < N; ++idx) {
		if (std::abs(hostData[idx] - deviceData[idx]) > epsilon) {
			match = 0;
			printf("Data don't match!\n");
			printf("hostData %5.2f, device %5.2f at current %d\n",
				hostData[idx], deviceData[idx], idx);
			break;
		}
	}
	if (match) {
		printf("Data is matched.\n");
	}
	return;
}

// check cuda api for debug
#define CHECK(call)                                                                              \
{                                                                                                \
	const cudaError_t error = call;                                                              \
	if (error != cudaSuccess)                                                                    \
	{                                                                                            \
		printf("Error: %s: %d, ", __FILE__, __LINE__);                                           \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));                       \
		exit(1);                                                                                 \
	}                                                                                            \
}

int main(int argc, char *argv[])
{ 
	int nElem = 1024;
	size_t nBytes = nElem * sizeof(float);

	float* hA, * hB, * hC;
	hA = (float*)malloc(nBytes);
	hB = (float*)malloc(nBytes);
	hC = (float*)malloc(nBytes);

	InitialData(hA, nElem);
	InitialData(hB, nElem);

	SumArraysOnHost(hA, hB, hC, nElem);

	printf("hC[5]: %f\n", hC[5]);

	// Gpu
	float* dA, * dB, * dC;
	cudaMalloc((float**)&dA, nBytes);
	cudaMalloc((float**)&dB, nBytes);
	cudaMalloc((float**)&dC, nBytes);

	cudaMemcpy(dA, hA, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, nBytes, cudaMemcpyHostToDevice);

	dim3 blockSize(nElem);
	dim3 gridSize((nElem + blockSize.x - 1) / blockSize.x);
	SumArrayOnDevice <<<gridSize, blockSize >>> (dA, dB, dC);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	// CHECK(cudaDeviceSynchronize());  // debug

	float* dData;
	dData = (float*)malloc(nBytes);
	cudaMemcpy(dData, dC, nBytes, cudaMemcpyDeviceToHost);

	checkResult(hC, dData, nElem);

	free(dData);
	free(hA);
	free(hB);
	free(hC);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);


	return 0;
}
