//#include <iostream>
#include <stdio.h>

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

// 修饰符__global__告诉编译器这个函数将会从CPU中调用，然后在GPU上执行
__global__ void helloFromGPU(void)
{
	//std::cout << "Hello world from GPU!" << std::endl;  // Error: identifier "std::cout" is undefined in device code
	printf("Hello world from GPU %d!\n", threadIdx.x);
}

int main(void)
{ 
    //std::cout << "Hello world from CPU!" << std::endl;
	printf("Hello world from CPU!\n");
	// kernel_function<<<grid dimensions, block dimensions, dynamic shared memory, stream ID>>>
	helloFromGPU <<<1, 10>>> ();   // 启动内核函数,三重尖括号意味着从主线程到设备端代码的调用,三重尖括号里面的参数是执行配置，用来说明使用多少线程来执行内核函数。在这个例子中，有10个GPU线程被调用
	//cudaDeviceReset();  //用来显式地释放和清空当前进程中与当前设备有关的所有资源
	cudaDeviceSynchronize();  // cudaDeviceReset在多线程应用程序中使用不安全，可采用cudaDeviceSynchronize

	return 0;
}
