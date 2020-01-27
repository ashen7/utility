/*
 * =====================================================================================
 *
 *       Filename:  matrix_gpu.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年01月04日 16时42分49秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "matrix_gpu.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include <map>
#include <algorithm>

#include <glog/logging.h>

#include "matrix_math_function.hpp"

namespace calculate {
namespace cuda {

extern const int kDepth;
extern const int kHeight;
extern const int kWidth;

//device端和host端函数
__device__ __host__ double Max(double a, double b) {
    return (a > b ? a : b);
} 

//核函数 矩阵相乘 点积
__global__ void DotProductKernel(double* a, double* b, double* c, 
                                 int left_rows, int left_cols, int right_cols) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < left_rows 
            && col < right_cols) {
        c[row * right_cols + col] = 0;
        for (int i = 0; i < left_cols; i++) {
            c[row * right_cols + col] += a[row * left_cols + i] * b[i * right_cols + col];
        }
    }
}

//核函数 矩阵相加
__global__ void AddKernel(double* a, double* b, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = a[thread_id] + b[thread_id];
}

//核函数 矩阵相加
__global__ void AddKernel(double* a, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] += a[thread_id];
}

//核函数 矩阵相加
__global__ void AddKernel(double a[][kWidth], double b[][kWidth], double c[][kWidth]) {
	const unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
	if (ix < kHeight && iy < kWidth) {
        c[ix][iy] = a[ix][iy] + b[ix][iy];
    }
}

//核函数 矩阵相加
__global__ void AddKernel(double a[kDepth][kHeight][kWidth], double b[kDepth][kHeight][kWidth], double c[kDepth][kHeight][kWidth]) {
	const unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
	const unsigned int iz = threadIdx.z + blockDim.z * blockIdx.z;
	if (ix < kDepth && iy < kHeight && iz < kWidth) {
	    c[ix][iy][iz] = a[ix][iy][iz] + b[ix][iy][iz];
	}
}

//核函数 矩阵相减
__global__ void SubtractKernel(double* a, double* b, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = a[thread_id] - b[thread_id];
}

//核函数 矩阵相减
__global__ void SubtractKernel(double* a, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = a[thread_id] - c[thread_id];
}


//核函数 矩阵hadamark积
__global__ void HadamarkProductKernel(double* a, double* b, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = a[thread_id] * b[thread_id];
}

//核函数 矩阵hadamark积
__global__ void HadamarkProductKernel(double* a, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] *= a[thread_id];
}


//核函数 矩阵同乘一个值
__global__ void ValueMulMatrixKernel(double* a, double value, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = a[thread_id] * value;
}

//核函数 矩阵同乘一个值
__global__ void ValueMulMatrixKernel(double value, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] *= value;
}


//核函数 一个值减去一个矩阵
__global__ void ValueSubMatrixKernel(double value, double* a, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = value - a[thread_id];
}

//核函数 一个值减去一个矩阵
__global__ void ValueSubMatrixKernel(double value, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = value - c[thread_id];
}


//核函数 矩阵同除一个值
__global__ void MatrixDivValueKernel(double* a, double value, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = a[thread_id] / value;
}

//核函数 矩阵同除一个值
__global__ void MatrixDivValueKernel(double value, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] /= value;
}

//核函数 求和
__global__ void SumKernel(double* c, double* value) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    *value += c[thread_id];
}

//核函数 均方误差
__global__ void MeanSquareError(double* a, double* b, double* value) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    *value += (pow((b[thread_id] - a[thread_id]), 2) / 2);
}

//核函数 sigmoid前向激活函数
__global__ void SigmoidForwardKernel(double* a, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = 1.0 / (1.0 + exp(-a[thread_id])); 
}

//核函数 sigmoid前向激活函数
__global__ void SigmoidForwardKernel(double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = 1.0 / (1.0 + exp(-c[thread_id])); 
}

//核函数 sigmoid反向激活函数
__global__ void SigmoidBackwardKernel(double* a, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = a[thread_id] * (1.0 - a[thread_id]); 
}

//核函数 sigmoid反向激活函数
__global__ void SigmoidBackwardKernel(double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = c[thread_id] * (1.0 - c[thread_id]); 
}

//核函数 ReLu前向激活函数
__global__ void ReLuForwardKernel(double* a, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = Max(0.0, a[thread_id]);
}

//核函数 ReLu前向激活函数
__global__ void ReLuForwardKernel(double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = Max(0.0, c[thread_id]);
}

//核函数 ReLu反向激活函数
__global__ void ReLuBackwardKernel(double* a, double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = a[thread_id] > 0 ? 1.0 : 0.0; 
}

//核函数 ReLu反向激活函数
__global__ void ReLuBackwardKernel(double* c) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = c[thread_id] > 0 ? 1.0 : 0.0; 
}


//初始化CUDA
bool InitializeCUDA() {
    int device_count;
    cudaGetDeviceCount(&device_count);  //得到device的数目
    if (0 == device_count) {
        LOG(ERROR) << "there is no gpu device";
        return false;
    }
    int i;
    for (i = 0; i < device_count; i++) {
        //cuda存放设备信息的结构体 
        cudaDeviceProp device_prop;
        if (cudaSuccess == cudaGetDeviceProperties(&device_prop, i)) {
            if (device_prop.major >= 1) {
                break;
            }
        }
    }
    if (i == device_count) {
        LOG(ERROR) << "there is no gpu device supporting CUDA";
        return false;
    }
    //设置device
    cudaSetDevice(i);

    return true;
}

//打印GPU硬件信息
void GpuInfoShow() {
    int device_count;
    cudaGetDeviceCount(&device_count);  //得到device的数目
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        LOG(INFO) << "================================================================";
        LOG(INFO) << "使用GPU device：" << i << ": " << device_prop.name;
        LOG(INFO) << "设备全局内存总量：" << device_prop.totalGlobalMem / 1024 / 1024 << "MB";
        LOG(INFO) << "SM数量(一个线程块对应一个物理上的sm)：" << device_prop.multiProcessorCount;
        LOG(INFO) << "每个线程块的共享内存大小：" << device_prop.sharedMemPerBlock / 1024.0 << "KB";
        LOG(INFO) << "设备上一个线程块中可用的32位寄存器数量：" << device_prop.regsPerBlock;
        LOG(INFO) << "每个SM的最大线程数：" << device_prop.maxThreadsPerMultiProcessor;
        LOG(INFO) << "每个SM的最大线程束数：" << device_prop.maxThreadsPerMultiProcessor / 32;
        LOG(INFO) << "设备上多处理器的数量：" << device_prop.multiProcessorCount; 
        LOG(INFO) << "================================================================";
    }
}

//检查cuda调用函数是否成功
int CheckCudaError() {
    cudaError_t cuda_error = cudaGetLastError();
    if (cudaSuccess != cuda_error) {
        LOG(ERROR) << cudaGetErrorString(cuda_error);
        return -1;
    }

    return 0;
}

//得到block size
int GetBlockSize(int rows, int cols) {
    if (rows <= 0) {
        LOG(ERROR) << "get block size failed";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get block size failed";
        return -1;
    }
    int block_size = 0;
    int min_value = std::min(rows, cols);

    if (min_value >= 512) {
        block_size = 512;
    } else if (min_value >= 256) {
        block_size = 256;
    } else if (min_value >= 128) {
        block_size = 128;
    } else if (min_value >= 64) {
        block_size = 64;
    } else if (min_value >= 32) {
        block_size = 32;
    } else if (min_value >= 16) {
        block_size = 16;
    } else if (min_value >= 8) {
        block_size = 8;
    } else if (min_value >= 4) {
        block_size = 4;
    } else if (min_value >= 2) {
        block_size = 2;
    } else {
        block_size = 1;
    }
    
    return block_size;
}

int GetBlockSize(int rows, int cols, std::tuple<int, int>& shape) {
    if (rows <= 0) {
        LOG(ERROR) << "get block size failed";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get block size failed";
        return -1;
    }
    int block_size_x = 0;
    int block_size_y = 0;
    int min_value = std::min(rows, cols);

    if (min_value >= 512) {
        block_size_x = 32;
        block_size_y = 16;
    } else if (min_value >= 256) {
        block_size_x = 16;
        block_size_y = 16;
    } else if (min_value >= 128) {
        block_size_x = 16;
        block_size_y = 8;
    } else if (min_value >= 64) {
        block_size_x = 8;
        block_size_y = 8;
    } else if (min_value >= 32) {
        block_size_x = 8;
        block_size_y = 4;
    } else if (min_value >= 16) {
        block_size_x = 4;
        block_size_y = 4;
    } else if (min_value >= 8) {
        block_size_x = 4;
        block_size_y = 2;
    } else if (min_value >= 4) {
        block_size_x = 2;
        block_size_y = 2;
    } else if (min_value >= 2) {
        block_size_x = 2;
        block_size_y = 1;
    } else {
        block_size_x = 1;
        block_size_y = 1;
    }
    
    shape = std::make_tuple(block_size_x, block_size_y); 
    return 0;
}

//得到block size
int GetBlockSize() {
    int block_size = 0;
    int min_value = std::min(kDepth, kHeight);
    min_value = std::min(min_value, kWidth);

    if (min_value >= 512) {
        block_size = 512;
    } else if (min_value >= 256) {
        block_size = 256;
    } else if (min_value >= 128) {
        block_size = 128;
    } else if (min_value >= 64) {
        block_size = 64;
    } else if (min_value >= 32) {
        block_size = 32;
    } else if (min_value >= 16) {
        block_size = 16;
    } else if (min_value >= 8) {
        block_size = 8;
    } else if (min_value >= 4) {
        block_size = 4;
    } else if (min_value >= 2) {
        block_size = 2;
    } else {
        block_size = 1;
    }
    
    return block_size;
}


//2d矩阵相乘 点积
int DotProduct(const Matrix2d& left_matrix, 
               const Matrix2d& right_matrix, 
               Matrix2d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d left_array;
    Matrix1d right_array;
    int left_rows;
    int left_cols;
    int right_cols;
    if (-1 == Matrix::Reshape(left_matrix, right_matrix, 
                              left_rows, left_cols, right_cols, 
                              left_array, right_array)) {
        LOG(ERROR) << "cuda matrix dot product failed";
        return -1;
    }
    //初始化结果矩阵
    Matrix1d result_array(left_rows * right_cols);
    //计算block size
    std::tuple<int, int> shape;
    int block_size_x;
    int block_size_y;
    if (-1 == GetBlockSize(left_rows, right_cols, shape)) {
        LOG(ERROR) << "cuda matrix dot product failed";
        return -1;
    }
    std::tie(block_size_x, block_size_y) = shape;

    double* device_a = NULL;
    double* device_b = NULL;
    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(double) * left_rows * left_cols);
    cudaMalloc((void**)&device_b, sizeof(double) * left_cols * right_cols);
    cudaMalloc((void**)&device_c, sizeof(double) * left_rows * right_cols);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix dot product failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_a, &left_array[0], sizeof(double) * left_rows * left_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, &right_array[0], sizeof(double) * left_cols * right_cols, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix dot product failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size_x, block_size_y);                                  //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((left_rows + dim_block.x - 1) / dim_block.x, 
                   (right_cols + dim_block.y - 1) / dim_block.y);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    DotProductKernel<<<dim_grid, dim_block>>>(device_a, device_b, device_c, 
                                              left_rows, left_cols, right_cols);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * left_rows * right_cols, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix dot product failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix dot product failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, left_rows, right_cols, result_matrix);

    return 0;
}

//2d矩阵相加 
int Add(const Matrix2d& left_matrix, 
        const Matrix2d& right_matrix, 
        Matrix2d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d left_array;
    Matrix1d right_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(left_matrix, right_matrix, rows, cols,
                              left_array, right_array)) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    }

    double* device_a = NULL;
    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(double) * total_size);
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_a, &left_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, &right_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    AddKernel<<<dim_grid, dim_block>>>(device_a, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&left_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(left_array, rows, cols, result_matrix);

    return 0;
}

//3d矩阵相加 
int Add(const Matrix3d& left_matrix, 
        const Matrix3d& right_matrix, 
        Matrix3d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d left_array;
    Matrix1d right_array;
    int depth;
    int height;
    int width;
    if (-1 == Matrix::Reshape(left_matrix, right_matrix, depth, height, width, 
                              left_array, right_array)) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    }

    //初始化保存结果的1d矩阵 
    const int total_size = depth * height * width;
    //计算block size
    const int block_size = GetBlockSize(height, width);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    }

    double* device_a = NULL;
    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(double) * total_size);
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_a, &left_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, &right_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    AddKernel<<<dim_grid, dim_block>>>(device_a, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&left_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(left_array, depth, height, width, result_matrix);

    return 0;
}

int AddFixed(const Matrix3d& left_matrix, 
             const Matrix3d& right_matrix, 
             Matrix3d& result_matrix) {
    //check source matrix
    if (!Matrix::MatrixCheck(left_matrix, kDepth, kHeight, kWidth, true)) {
        LOG(ERROR) << "cuda matrix add failed, source matrix is wrong";
        return -1;
    }
    if (!Matrix::MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "cuda matrix add failed, source matrix is not equal";
        return -1;
    }

    //create 3d matrix
    const int block_size = GetBlockSize();
    if (-1 == block_size) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    }
    double left_array[kDepth][kHeight][kWidth] = { 0.0 };
    double right_array[kDepth][kHeight][kWidth] = { 0.0 };
    double result_array[kDepth][kHeight][kWidth] = { 0.0 };
    
    for (int i = 0; i < kDepth; i++) {
        for (int j = 0; j < kHeight; j++) {
            for (int k = 0; k < kWidth; k++) {
                left_array[i][j][k] = left_matrix[i][j][k];    
                right_array[i][j][k] = right_matrix[i][j][k];
            }
        }
    }

    const int total_size = kDepth * kHeight * kWidth;
    result_matrix = Matrix3d(kDepth, Matrix2d(kHeight, Matrix1d(kWidth)));

    double (*device_a)[kHeight][kWidth] = NULL;
    double (*device_b)[kHeight][kWidth] = NULL;
    double (*device_c)[kHeight][kWidth] = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(double) * total_size);
    cudaMalloc((void**)&device_b, sizeof(double) * total_size);
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_a, left_array, sizeof(double) * total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, right_array, sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    dim3 dim_block(block_size, block_size, block_size);       //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((kDepth + dim_block.x - 1) / dim_block.x, 
                  (kHeight + dim_block.y - 1) / dim_block.y, 
                  (kWidth + dim_block.z - 1) / dim_block.z);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    AddKernel<<<dim_grid, dim_block>>>(device_a, device_b, device_c);
    
    cudaMemcpy(result_array, device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };

    for (int i = 0; i < kDepth; i++) {
        for (int j = 0; j < kHeight; j++) {
            for (int k = 0; k < kWidth; k++) {
                result_matrix[i][j][k] = result_array[i][j][k];    
            }
        }
    }
    
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix add failed";
        return -1;
    };

    return 0;
}

//2d矩阵相减 
int Subtract(const Matrix2d& left_matrix, 
             const Matrix2d& right_matrix, 
             Matrix2d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d left_array;
    Matrix1d right_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(left_matrix, right_matrix, rows, cols,
                              left_array, right_array)) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    }

    double* device_a = NULL;
    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(double) * total_size);
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_a, &left_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, &right_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    SubtractKernel<<<dim_grid, dim_block>>>(device_a, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&left_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(left_array, rows, cols, result_matrix);

    return 0;
}

//3d矩阵相减 
int Subtract(const Matrix3d& left_matrix, 
             const Matrix3d& right_matrix, 
             Matrix3d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d left_array;
    Matrix1d right_array;
    int depth;
    int height;
    int width;
    if (-1 == Matrix::Reshape(left_matrix, right_matrix, depth, height, width, 
                              left_array, right_array)) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    }

    //初始化保存结果的1d矩阵 
    const int total_size = depth * height * width;
    //计算block size
    const int block_size = GetBlockSize(height, width);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    }

    double* device_a = NULL;
    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(double) * total_size);
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_a, &left_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, &right_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    SubtractKernel<<<dim_grid, dim_block>>>(device_a, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&left_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix subtract failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(left_array, depth, height, width, result_matrix);

    return 0;
}

//2d矩阵 hadamark积 
int HadamarkProduct(const Matrix2d& left_matrix, 
                    const Matrix2d& right_matrix, 
                    Matrix2d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d left_array;
    Matrix1d right_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(left_matrix, right_matrix, rows, cols,
                              left_array, right_array)) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    }

    double* device_a = NULL;
    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(double) * total_size);
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_a, &left_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, &right_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    HadamarkProductKernel<<<dim_grid, dim_block>>>(device_a, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&left_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(left_array, rows, cols, result_matrix);

    return 0;
}

//3d矩阵 hadamark积 
int HadamarkProduct(const Matrix3d& left_matrix, 
                    const Matrix3d& right_matrix, 
                    Matrix3d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d left_array;
    Matrix1d right_array;
    int depth;
    int height;
    int width;
    if (-1 == Matrix::Reshape(left_matrix, right_matrix, depth, height, width, 
                              left_array, right_array)) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    }

    //初始化保存结果的1d矩阵 
    const int total_size = depth * height * width;
    //计算block size
    const int block_size = GetBlockSize(height, width);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    }

    double* device_a = NULL;
    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(double) * total_size);
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_a, &left_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, &right_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    HadamarkProductKernel<<<dim_grid, dim_block>>>(device_a, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&left_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix hadamark product failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(left_array, depth, height, width, result_matrix);

    return 0;
}

//一个值乘以一个2d矩阵 
int ValueMulMatrix(double value,  
                   const Matrix2d& source_matrix, 
                   Matrix2d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(source_matrix, 
                              rows, cols,
                              result_array)) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    ValueMulMatrixKernel<<<dim_grid, dim_block>>>(value, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, rows, cols, result_matrix);

    return 0;
}

//一个值乘以一个3d矩阵 
int ValueMulMatrix(double value,  
                   const Matrix3d& source_matrix, 
                   Matrix3d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int depth;
    int height;
    int width;
    if (-1 == Matrix::Reshape(source_matrix, 
                              depth, height, width, 
                              result_array)) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    }

    //初始化保存结果的1d矩阵 
    const int total_size = depth * height * width;
    //计算block size
    const int block_size = GetBlockSize(height, width);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    ValueMulMatrixKernel<<<dim_grid, dim_block>>>(value, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value multiply matrix failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, depth, height, width, result_matrix);

    return 0;
}

//一个值减去一个2d矩阵 
int ValueSubMatrix(double value,  
                   const Matrix2d& source_matrix, 
                   Matrix2d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(source_matrix, 
                              rows, cols,
                              result_array)) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    ValueSubMatrixKernel<<<dim_grid, dim_block>>>(value, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, rows, cols, result_matrix);

    return 0;
}

//一个值减去一个3d矩阵 
int ValueSubMatrix(double value,  
                   const Matrix3d& source_matrix, 
                   Matrix3d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int depth;
    int height;
    int width;
    if (-1 == Matrix::Reshape(source_matrix, 
                              depth, height, width, 
                              result_array)) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    }

    //初始化保存结果的1d矩阵 
    const int total_size = depth * height * width;
    //计算block size
    const int block_size = GetBlockSize(height, width);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    ValueSubMatrixKernel<<<dim_grid, dim_block>>>(value, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda value subtract matrix failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, depth, height, width, result_matrix);

    return 0;
}

//一个2d矩阵同除以一个值 
int MatrixDivValue(const Matrix2d& source_matrix, 
                   double value,  
                   Matrix2d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(source_matrix, 
                              rows, cols,
                              result_array)) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    MatrixDivValueKernel<<<dim_grid, dim_block>>>(value, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, rows, cols, result_matrix);

    return 0;
}

//一个3d矩阵同除以一个值 
int MatrixDivValue(const Matrix3d& source_matrix, 
                   double value,  
                   Matrix3d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int depth;
    int height;
    int width;
    if (-1 == Matrix::Reshape(source_matrix, 
                              depth, height, width, 
                              result_array)) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    }

    //初始化保存结果的1d矩阵 
    const int total_size = depth * height * width;
    //计算block size
    const int block_size = GetBlockSize(height, width);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    MatrixDivValueKernel<<<dim_grid, dim_block>>>(value, device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda matrix divide value failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, depth, height, width, result_matrix);

    return 0;
}

//sigmoid激活函数的前向计算
int SigmoidForward(const Matrix2d& source_matrix, 
                   Matrix2d& result_matrix) { 
    //reshape 成1d矩阵
    Matrix1d result_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(source_matrix, 
                              rows, cols,
                              result_array)) {
        LOG(ERROR) << "cuda activator sigmoid forward failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda activator sigmoid forward failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator sigmoid forward failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator sigmoid forward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    SigmoidForwardKernel<<<dim_grid, dim_block>>>(device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator sigmoid forward failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator sigmoid forward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, rows, cols, result_matrix);

    return 0;
}

//sigmoid激活函数的反向计算
int SigmoidBackward(const Matrix2d& result_matrix, 
                    Matrix2d& delta_array) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(result_matrix, 
                              rows, cols,
                              result_array)) {
        LOG(ERROR) << "cuda activator sigmoid backward failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda activator sigmoid backward failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator sigmoid backward failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator sigmoid backward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    SigmoidBackwardKernel<<<dim_grid, dim_block>>>(device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator sigmoid backward failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator sigmoid backward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, rows, cols, delta_array);

    return 0;
}

//ReLu激活函数的2d前向计算
int ReLuForward2d(const Matrix2d& source_matrix, 
                  Matrix2d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(source_matrix, 
                              rows, cols,
                              result_array)) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    ReLuForwardKernel<<<dim_grid, dim_block>>>(device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, rows, cols, result_matrix);

    return 0;
}

//ReLu激活函数的3d前向计算
int ReLuForward3d(const Matrix3d& source_matrix, 
                  Matrix3d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int depth;
    int height;
    int width;
    if (-1 == Matrix::Reshape(source_matrix, 
                              depth, height, width, 
                              result_array)) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    }

    //初始化保存结果的1d矩阵 
    const int total_size = depth * height * width;
    //计算block size
    const int block_size = GetBlockSize(height, width);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    ReLuForwardKernel<<<dim_grid, dim_block>>>(device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu forward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, depth, height, width, result_matrix);

    return 0;

}

//ReLu激活函数的3d反向计算
int ReLuBackward2d(const Matrix2d& source_matrix, 
                   Matrix2d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int rows;
    int cols;
    if (-1 == Matrix::Reshape(source_matrix, 
                              rows, cols,
                              result_array)) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    }
    const int total_size = rows * cols;
    //计算block size
    const int block_size = GetBlockSize(rows, cols);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                   //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);  //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    ReLuBackwardKernel<<<dim_grid, dim_block>>>(device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, rows, cols, result_matrix);

    return 0;
    
}

//ReLu激活函数的3d反向计算
int ReLuBackward3d(const Matrix3d& source_matrix, 
                   Matrix3d& result_matrix) {
    //reshape 成1d矩阵
    Matrix1d result_array;
    int depth;
    int height;
    int width;
    if (-1 == Matrix::Reshape(source_matrix, 
                              depth, height, width, 
                              result_array)) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    }

    //初始化保存结果的1d矩阵 
    const int total_size = depth * height * width;
    //计算block size
    const int block_size = GetBlockSize(height, width);
    if (-1 == block_size) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    }

    double* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_c, sizeof(double) * total_size);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    };
    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_c, &result_array[0], sizeof(double) * total_size, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size);                                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((total_size + dim_block.x - 1) / dim_block.x);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    ReLuBackwardKernel<<<dim_grid, dim_block>>>(device_c);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_c, sizeof(double) * total_size, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    };
    
    //释放gpu显存
    cudaFree(device_c);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda activator relu backward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, depth, height, width, result_matrix);

    return 0;
}







}       //namespace cuda
}       //namespace calculate

