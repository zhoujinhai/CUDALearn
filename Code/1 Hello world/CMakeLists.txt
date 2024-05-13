cmake_minimum_required(VERSION 3.9.0)

project(hello VERSION 1.0)

# >>> build type 
# set(CMAKE_BUILD_TYPE "Debug")				# 指定生成的版本
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# <<<


# >>> CXX11 
set(CMAKE_CXX_STANDARD 11)				# C++ 11 编译器
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# <<< CXX11

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
set(CUDA_INCLUDE "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")


project(${PROJECT_NAME} LANGUAGES CXX CUDA) #使用CUDA, 在这里会检查nvcc路径是否正确

add_executable(${PROJECT_NAME} Hello.cu)
target_include_directories(${PROJECT_NAME}  
PRIVATE  
    ${CUDA_INCLUDE}
)

target_link_directories(${PROJECT_NAME} PUBLIC ${CUDA_LIB_DIR})

target_link_libraries(${PROJECT_NAME} 
PUBLIC
	cudart
	# cudnn
) 



    


