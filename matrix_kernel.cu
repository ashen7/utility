/*
 * =====================================================================================
 *
 *       Filename:  matrix_kernel.cu
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
#include "matrix_kernel.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cudnn.h"

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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>

#include "matrix_math_function.hpp"

#define OPENMP_THREADS_NUMBER 6   //openmp并行线程数量

namespace moon {
namespace calculate {
namespace cuda {

//带参宏 检查cudnn调用函数是否成功
#define CheckCUDNN(expression) {                            \
    cudnnStatus_t status = (expression);                    \
    if (status != CUDNN_STATUS_SUCCESS) {                   \
        LOG(FATAL) << "Error on line " << __LINE__          \
                   << ": " << cudnnGetErrorString(status);  \
    }                                                       \
}

//检查cuda调用函数是否成功
int CheckCudaError() {
    cudaError_t cuda_error = cudaGetLastError();
    if (cudaSuccess != cuda_error) {
        LOG(ERROR) << "Error on line " << __LINE__  
                   << ": " << cudaGetErrorString(cuda_error);
        return -1;
    }
    return 0;
}

extern const int kDepth;
extern const int kHeight;
extern const int kWidth;

//device端和host端函数
__device__ __host__ __forceinline__ double Max(double a, double b) {
    return (a > b ? a : b);
} 

//device端和host端函数
__device__ __host__ __forceinline__ double ReLuBackward(double a) {
    return (a > 0 ? 1.0 : 0.0); 
} 

//核函数 矩阵相乘 点积
__global__ void DotProductKernel(double* a, double* b, double* c, 
                                 int left_rows, int left_cols, int right_cols) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < left_rows 
            && col < right_cols) {
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

//核函数 全连接层的前向计算 有dropout a = f(w .* x + b)
__global__ void FullConnectedForwardKernel(double* weights_array, double* input_array, 
                                           double* binomial_array, double* result_array,
                                           int left_rows, int left_cols, 
                                           int right_cols, double p) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int thread_id = row * right_cols + col;
    if (row < left_rows 
            && col < right_cols) {
        for (int i = 0; i < left_cols; i++) {
            result_array[thread_id] += weights_array[row * left_cols + i] *
                                       input_array[i * right_cols + col];
        }
        result_array[thread_id] = ((1.0 / (1.0 + exp(-result_array[thread_id]))) * 
                                  binomial_array[thread_id]) / (1.0 - p); 
    }
}

//核函数 全连接层的前向计算 a = f(w .* x + b)
__global__ void FullConnectedForwardKernel(double* weights_array, double* input_array, 
                                           double* result_array, int left_rows,
                                           int left_cols, int right_cols) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int thread_id = row * right_cols + col;
    if (row < left_rows 
            && col < right_cols) {
        for (int i = 0; i < left_cols; i++) {
            result_array[thread_id] += weights_array[row * left_cols + i] *
                                       input_array[i * right_cols + col];
        }
        result_array[thread_id] = 1.0 / (1.0 + exp(-result_array[thread_id])); 
    }
}

//核函数 全连接层的反向计算 有dropout
//delta=x * (1-x) * wT .* delta_array, w梯度=delta_array .* xT, b梯度=delta_array
__global__ void FullConnectedBackwardKernel(double* weights_transpose_array, double* delta_array, 
                                            double* input_array, double* binomial_array,
                                            double* result_array, int left_rows,
                                            int left_cols, int right_cols, double p) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int thread_id = row * right_cols + col;
    if (row < left_rows 
            && col < right_cols) {
        result_array[thread_id] = 0.0;
        for (int i = 0; i < left_cols; i++) {
            result_array[thread_id] += weights_transpose_array[row * left_cols + i] *
                                       delta_array[i * right_cols + col];
        }
        result_array[thread_id] = result_array[thread_id] *
                                  input_array[thread_id] *
                                  (1.0 - input_array[thread_id]) *
                                  binomial_array[thread_id] / (1.0 - p) *
                                  ReLuBackward(input_array[thread_id]);
    }
}

//核函数 全连接层的反向计算 有dropout
//delta=x * (1-x) * wT .* delta_array, w梯度=delta_array .* xT, b梯度=delta_array
__global__ void FullConnectedBackwardKernel(double* weights_transpose_array, double* delta_array, 
                                            double* input_array, double* result_array, 
                                            int left_rows, int left_cols, int right_cols) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int thread_id = row * right_cols + col;
    if (row < left_rows 
            && col < right_cols) {
        result_array[thread_id] = 0.0;
        for (int i = 0; i < left_cols; i++) {
            result_array[thread_id] += weights_transpose_array[row * left_cols + i] *
                                       delta_array[i * right_cols + col];
        }
        result_array[thread_id] = result_array[thread_id] *
                                  input_array[thread_id] *
                                  (1.0 - input_array[thread_id]);
    }
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
    int max_value = std::max(rows, cols);

    if (max_value >= 512) {
        block_size = 512;
    } else if (max_value >= 256) {
        block_size = 256;
    } else if (max_value >= 128) {
        block_size = 128;
    } else if (max_value >= 64) {
        block_size = 64;
    } else if (max_value >= 32) {
        block_size = 32;
    } else if (max_value >= 16) {
        block_size = 16;
    } else if (max_value >= 8) {
        block_size = 8;
    } else if (max_value >= 4) {
        block_size = 4;
    } else if (max_value >= 2) {
        block_size = 2;
    } else {
        block_size = 1;
    }
    
    return block_size;
}

//得到block size
int GetBlockSize(int depth, int height, int width) {
    if (depth <= 0) {
        LOG(ERROR) << "get block size failed";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get block size failed";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get block size failed";
        return -1;
    }
    int block_size = 0;
    int max_value = std::max(depth, height);
    max_value = std::max(max_value, width);

    if (max_value >= 512) {
        block_size = 512;
    } else if (max_value >= 256) {
        block_size = 256;
    } else if (max_value >= 128) {
        block_size = 128;
    } else if (max_value >= 64) {
        block_size = 64;
    } else if (max_value >= 32) {
        block_size = 32;
    } else if (max_value >= 16) {
        block_size = 16;
    } else if (max_value >= 8) {
        block_size = 8;
    } else if (max_value >= 4) {
        block_size = 4;
    } else if (max_value >= 2) {
        block_size = 2;
    } else {
        block_size = 1;
    }
    
    return block_size;
}

//得到2d的block size
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
    int max_value = std::max(rows, cols);

    if (max_value >= 512) {
        block_size_x = 32;
        block_size_y = 16;
    } else if (max_value >= 256) {
        block_size_x = 16;
        block_size_y = 16;
    } else if (max_value >= 128) {
        block_size_x = 16;
        block_size_y = 8;
    } else if (max_value >= 64) {
        block_size_x = 8;
        block_size_y = 8;
    } else if (max_value >= 32) {
        block_size_x = 8;
        block_size_y = 4;
    } else if (max_value >= 16) {
        block_size_x = 4;
        block_size_y = 4;
    } else if (max_value >= 8) {
        block_size_x = 4;
        block_size_y = 2;
    } else if (max_value >= 4) {
        block_size_x = 2;
        block_size_y = 2;
    } else if (max_value >= 2) {
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
    //cudaMemset(device_c, 0, sizeof(double) * left_rows * right_cols);
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
    Matrix::MatrixShow(result_array); 
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
    const int block_size = GetBlockSize(depth, height, width);
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
    const int block_size = GetBlockSize(depth, height, width);
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
    const int block_size = GetBlockSize(depth, height, width);
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
    const int block_size = GetBlockSize(depth, height, width);
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
    const int block_size = GetBlockSize(depth, height, width);
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
    const int block_size = GetBlockSize(depth, height, width);
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
    const int block_size = GetBlockSize(depth, height, width);
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
    const int block_size = GetBlockSize(depth, height, width);
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

//全连接层的前向计算
int FullConnectedForward(const Matrix2d& weights_array, 
                         const Matrix2d& input_array, 
                         const Matrix2d& biases_array,
                         Matrix2d& binomial_array, 
                         Matrix2d& output_array, 
                         bool is_input_layer,  
                         bool dropout, double p) {
    if (0 == weights_array.size()
            || 0 == input_array.size()
            || 0 == biases_array.size()) {
        LOG(ERROR) << "cuda full connected layer forward failed, input source matrix is empty";
        return -1;
    }
    //reshape 成1d矩阵
    Matrix1d _weights_array;
    Matrix1d _input_array;
    Matrix1d _biases_array;
    Matrix1d _binomial_array;
    int left_rows;
    int left_cols;
    int right_cols;
    if (-1 == Matrix::Reshape(weights_array, input_array, 
                              left_rows, left_cols, right_cols, 
                              _weights_array, _input_array)) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    }

    //初始化结果矩阵
    Matrix1d result_array(left_rows * right_cols);
    //计算block size
    std::tuple<int, int> shape;
    int block_size_x;
    int block_size_y;
    if (-1 == GetBlockSize(left_rows, right_cols, shape)) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    }
    std::tie(block_size_x, block_size_y) = shape;

    double* device_weight_array = NULL;
    double* device_input_array = NULL;
    double* device_binomial_array = NULL;
    double* device_result_array = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_weight_array, sizeof(double) * left_rows * left_cols);
    cudaMalloc((void**)&device_input_array, sizeof(double) * left_cols * right_cols);
    cudaMalloc((void**)&device_binomial_array, sizeof(double) * left_rows * right_cols);
    cudaMalloc((void**)&device_result_array, sizeof(double) * left_rows * right_cols);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    };

    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_weight_array, &_weights_array[0], sizeof(double) * left_rows * left_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_array, &_input_array[0], sizeof(double) * left_cols * right_cols, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size_x, block_size_y);                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((left_rows + dim_block.x - 1) / dim_block.x, 
                  (right_cols + dim_block.y - 1) / dim_block.y);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    if (is_input_layer && dropout) {
        //伯努利分布 得到0 1数组 
        if (-1 == Random::Binomial(1, p, Matrix::GetShape(biases_array),
                                   binomial_array)) {
            LOG(ERROR) << "cuda full connected layer forward failed, drop out occur error";
            return -1;
        }
        if (-1 == Matrix::Reshape(biases_array, binomial_array, 
                                  left_rows, right_cols, 
                                  _biases_array, _binomial_array)) {
            LOG(ERROR) << "cuda full connected layer forward failed";
            return -1;
        }
        //这里得到了二项分布矩阵 copy to device
        cudaMemcpy(device_binomial_array, &_binomial_array[0], sizeof(double) * left_rows * right_cols, cudaMemcpyHostToDevice);
        cudaMemcpy(device_result_array, &_biases_array[0], sizeof(double) * left_rows * right_cols, cudaMemcpyHostToDevice);
        FullConnectedForwardKernel<<<dim_grid, dim_block>>>(device_weight_array, device_input_array, 
                                                            device_binomial_array, device_result_array, 
                                                            left_rows, left_cols, right_cols, p); 
    } else {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    }

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_result_array, sizeof(double) * left_rows * right_cols, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    };
    //释放gpu显存
    cudaFree(device_weight_array);
    cudaFree(device_input_array);
    cudaFree(device_binomial_array);
    cudaFree(device_result_array);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, left_rows, right_cols, output_array);
    
    return 0;
}

//全连接层的前向计算
int FullConnectedForward(const Matrix2d& weights_array, 
                         const Matrix2d& input_array, 
                         const Matrix2d& biases_array,
                         Matrix2d& output_array, 
                         bool is_input_layer) { 
    if (0 == weights_array.size()
            || 0 == input_array.size()
            || 0 == biases_array.size()) {
        LOG(ERROR) << "cuda full connected layer forward failed, input source matrix is empty";
        return -1;
    }
    //reshape 成1d矩阵
    Matrix1d _weights_array;
    Matrix1d _input_array;
    Matrix1d _biases_array;
    int left_rows;
    int left_cols;
    int right_cols;
    if (-1 == Matrix::Reshape(weights_array, input_array, 
                              left_rows, left_cols, right_cols, 
                              _weights_array, _input_array)) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    }
    if (-1 == Matrix::Reshape(biases_array, 
                              _biases_array)) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    }

    //初始化结果矩阵
    Matrix1d result_array(left_rows * right_cols);
    //计算block size
    std::tuple<int, int> shape;
    int block_size_x;
    int block_size_y;
    if (-1 == GetBlockSize(left_rows, right_cols, shape)) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    }
    std::tie(block_size_x, block_size_y) = shape;

    double* device_weight_array = NULL;
    double* device_input_array = NULL;
    double* device_result_array = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_weight_array, sizeof(double) * left_rows * left_cols);
    cudaMalloc((void**)&device_input_array, sizeof(double) * left_cols * right_cols);
    cudaMalloc((void**)&device_result_array, sizeof(double) * left_rows * right_cols);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    };

    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_weight_array, &_weights_array[0], sizeof(double) * left_rows * left_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_array, &_input_array[0], sizeof(double) * left_cols * right_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_result_array, &_biases_array[0], sizeof(double) * left_rows * right_cols, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size_x, block_size_y);                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((left_rows + dim_block.x - 1) / dim_block.x, 
                  (right_cols + dim_block.y - 1) / dim_block.y);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)

    if (is_input_layer) {
        FullConnectedForwardKernel<<<dim_grid, dim_block>>>(device_weight_array, device_input_array, 
                                                            device_result_array, left_rows,
                                                            left_cols, right_cols); 
    } else {
        FullConnectedForwardKernel<<<dim_grid, dim_block>>>(device_weight_array, device_input_array, 
                                                            device_result_array, left_rows,
                                                            left_cols, right_cols); 
    }

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_result_array, sizeof(double) * left_rows * right_cols, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    };
    //释放gpu显存
    cudaFree(device_weight_array);
    cudaFree(device_input_array);
    cudaFree(device_result_array);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer forward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, left_rows, right_cols, output_array);

    return 0;
}

//全连接层的反向计算
int FullConnectedBackward(const Matrix2d& output_delta_array, 
                          const Matrix2d& weights_array, 
                          const Matrix2d& input_array,
                          const Matrix2d& binomial_array, 
                          Matrix2d& delta_array, 
                          Matrix2d& weights_gradient_array, 
                          Matrix2d& biases_gradient_array, 
                          bool is_input_layer, 
                          bool dropout, double p) { 
    if (0 == output_delta_array.size()
            || 0 == weights_array.size()
            || 0 == input_array.size()
            || 0 == binomial_array.size()
            || 0 == weights_gradient_array.size()
            || 0 == biases_gradient_array.size()) {
        LOG(ERROR) << "cuda full connected layer backward failed, input source matrix is empty";
        return -1;
    }
    //计算本层误差项
    Matrix2d weights_transpose_array;
    Matrix1d _weights_transpose_array;
    Matrix1d _output_delta_array;
    Matrix1d _input_array;
    Matrix1d _binomial_array;
    
    int left_rows;
    int left_cols;
    int right_cols;
    int rows;
    int cols;
    if (-1 == Matrix::Transpose(weights_array, weights_transpose_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    if (-1 == Matrix::Reshape(weights_transpose_array, output_delta_array, 
                              left_rows, left_cols, right_cols, 
                              _weights_transpose_array, _output_delta_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    if (-1 == Matrix::Reshape(input_array, binomial_array, 
                              rows, cols, 
                              _input_array, _binomial_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }

    //初始化结果矩阵
    Matrix1d result_array(left_rows * right_cols);
    //计算block size
    std::tuple<int, int> shape;
    int block_size_x;
    int block_size_y;
    if (-1 == GetBlockSize(left_rows, right_cols, shape)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    std::tie(block_size_x, block_size_y) = shape;

    double* device_weight_transpose_array = NULL;
    double* device_output_delta_array = NULL;
    double* device_input_array = NULL;
    double* device_binomial_array = NULL;
    double* device_result_array = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_weight_transpose_array, sizeof(double) * left_rows * left_cols);
    cudaMalloc((void**)&device_output_delta_array, sizeof(double) * left_cols * right_cols);
    cudaMalloc((void**)&device_input_array, sizeof(double) * rows * cols);
    cudaMalloc((void**)&device_binomial_array, sizeof(double) * rows * cols);
    cudaMalloc((void**)&device_result_array, sizeof(double) * left_rows * right_cols);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };

    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_weight_transpose_array, &_weights_transpose_array[0], sizeof(double) * left_rows * left_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output_delta_array, &_output_delta_array[0], sizeof(double) * left_cols * right_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_array, &_input_array[0], sizeof(double) * rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_binomial_array, &_binomial_array[0], sizeof(double) * rows * cols, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size_x, block_size_y);                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((left_rows + dim_block.x - 1) / dim_block.x, 
                  (right_cols + dim_block.y - 1) / dim_block.y);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)

    if (!is_input_layer && dropout) {
        FullConnectedBackwardKernel<<<dim_grid, dim_block>>>(device_weight_transpose_array, device_output_delta_array, 
                                                             device_input_array, device_binomial_array, 
                                                             device_result_array, left_rows, left_cols, 
                                                             right_cols, p);
    }

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_result_array, sizeof(double) * left_rows * right_cols, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };
    //释放gpu显存
    cudaFree(device_weight_transpose_array);
    cudaFree(device_input_array);
    cudaFree(device_binomial_array);
    cudaFree(device_result_array);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, left_rows, right_cols, delta_array);
    
    //计算梯度
    Matrix2d input_transpose_array;
    Matrix1d _input_transpose_array;
    Matrix1d _weights_gradient_array;
    if (-1 == Matrix::Transpose(input_array, input_transpose_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    if (-1 == Matrix::Reshape(output_delta_array, input_transpose_array, 
                              left_rows, left_cols, right_cols, 
                              _output_delta_array, _input_transpose_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        LOG(WARNING) << "5";
        return -1;
    }
    if (-1 == Matrix::Reshape(weights_gradient_array, _weights_gradient_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        LOG(WARNING) << "6";
        return -1;
    }
    //初始化结果矩阵
    result_array = Matrix1d(left_rows * right_cols);
    //计算block size
    if (-1 == GetBlockSize(left_rows, right_cols, shape)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    std::tie(block_size_x, block_size_y) = shape;

    double* device_input_transpose_array = NULL;
    double* device_weights_gradient_array = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_input_transpose_array, sizeof(double) * left_rows * left_cols);
    cudaMalloc((void**)&device_weights_gradient_array, sizeof(double) * left_rows * right_cols);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };

    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_input_transpose_array, &_input_transpose_array[0], sizeof(double) * left_rows * left_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_weights_gradient_array, &_weights_gradient_array[0], sizeof(double) * left_rows * right_cols, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 block(block_size_x, block_size_y);                 //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 grid((left_rows + block.x - 1) / block.x, 
              (right_cols + block.y - 1) / block.y);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    DotProductKernel<<<grid, block>>>(device_output_delta_array, device_input_transpose_array, 
                                      device_weights_gradient_array, left_rows, left_cols, right_cols);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_weights_gradient_array, sizeof(double) * left_rows * right_cols, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };
    cudaFree(device_output_delta_array);
    cudaFree(device_input_transpose_array);
    cudaFree(device_weights_gradient_array);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, left_rows, right_cols, weights_gradient_array);
    Matrix::Add(biases_gradient_array, output_delta_array, biases_gradient_array);

    return 0;
}

//全连接层的反向计算
int FullConnectedBackward(const Matrix2d& output_delta_array, 
                          const Matrix2d& weights_array, 
                          const Matrix2d& input_array,
                          Matrix2d& delta_array, 
                          Matrix2d& weights_gradient_array, 
                          Matrix2d& biases_gradient_array) { 
    if (0 == output_delta_array.size()
            || 0 == weights_array.size()
            || 0 == input_array.size()
            || 0 == weights_gradient_array.size()
            || 0 == biases_gradient_array.size()) {
        LOG(ERROR) << "cuda full connected layer backward failed, input source matrix is empty";
        return -1;
    }
    //计算本层误差项
    Matrix2d weights_transpose_array;
    Matrix1d _weights_transpose_array;
    Matrix1d _output_delta_array;
    Matrix1d _input_array;
    
    int left_rows;
    int left_cols;
    int right_cols;
    if (-1 == Matrix::Transpose(weights_array, weights_transpose_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    if (-1 == Matrix::Reshape(weights_transpose_array, output_delta_array, 
                              left_rows, left_cols, right_cols, 
                              _weights_transpose_array, _output_delta_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    if (-1 == Matrix::Reshape(input_array, _input_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }

    //初始化结果矩阵
    Matrix1d result_array(left_rows * right_cols);
    //计算block size
    std::tuple<int, int> shape;
    int block_size_x;
    int block_size_y;
    if (-1 == GetBlockSize(left_rows, right_cols, shape)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    std::tie(block_size_x, block_size_y) = shape;

    double* device_weight_transpose_array = NULL;
    double* device_output_delta_array = NULL;
    double* device_input_array = NULL;
    double* device_result_array = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_weight_transpose_array, sizeof(double) * left_rows * left_cols);
    cudaMalloc((void**)&device_output_delta_array, sizeof(double) * left_cols * right_cols);
    cudaMalloc((void**)&device_input_array, sizeof(double) * left_rows * right_cols);
    cudaMalloc((void**)&device_result_array, sizeof(double) * left_rows * right_cols);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };

    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_weight_transpose_array, &_weights_transpose_array[0], sizeof(double) * left_rows * left_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output_delta_array, &_output_delta_array[0], sizeof(double) * left_cols * right_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_array, &_input_array[0], sizeof(double) * left_rows * right_cols, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 dim_block(block_size_x, block_size_y);                     //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((left_rows + dim_block.x - 1) / dim_block.x, 
                  (right_cols + dim_block.y - 1) / dim_block.y);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    FullConnectedBackwardKernel<<<dim_grid, dim_block>>>(device_weight_transpose_array, device_output_delta_array, 
                                                         device_input_array, device_result_array,
                                                         left_rows, left_cols, right_cols);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_result_array, sizeof(double) * left_rows * right_cols, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };
    //释放gpu显存
    cudaFree(device_weight_transpose_array);
    cudaFree(device_input_array);
    cudaFree(device_result_array);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, left_rows, right_cols, delta_array);
    
    //计算梯度
    Matrix2d input_transpose_array;
    Matrix1d _input_transpose_array;
    Matrix1d _weights_gradient_array;
    if (-1 == Matrix::Transpose(input_array, input_transpose_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    if (-1 == Matrix::Reshape(output_delta_array, input_transpose_array, 
                              left_rows, left_cols, right_cols, 
                              _output_delta_array, _input_transpose_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        LOG(WARNING) << "5";
        return -1;
    }
    if (-1 == Matrix::Reshape(weights_gradient_array, _weights_gradient_array)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        LOG(WARNING) << "6";
        return -1;
    }
    //初始化结果矩阵
    result_array = Matrix1d(left_rows * right_cols);
    //计算block size
    if (-1 == GetBlockSize(left_rows, right_cols, shape)) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    }
    std::tie(block_size_x, block_size_y) = shape;

    double* device_input_transpose_array = NULL;
    double* device_weights_gradient_array = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_input_transpose_array, sizeof(double) * left_cols * right_cols);
    cudaMalloc((void**)&device_weights_gradient_array, sizeof(double) * left_rows * right_cols);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };

    //将内存中的值复制到GPU显存中
    cudaMemcpy(device_input_transpose_array, &_input_transpose_array[0], sizeof(double) * left_cols * right_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_weights_gradient_array, &_weights_gradient_array[0], sizeof(double) * left_rows * right_cols, cudaMemcpyHostToDevice);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 异步执行 cpu会继续往下运行 
    dim3 block(block_size_x, block_size_y);                 //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 grid((left_rows + block.x - 1) / block.x, 
              (right_cols + block.y - 1) / block.y);    //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    DotProductKernel<<<grid, block>>>(device_output_delta_array, device_input_transpose_array, 
                                      device_weights_gradient_array, left_rows, left_cols, right_cols);

    //GPU计算任务完成后 将数据传输回CPU 阻塞式API 这里会把gpu运算完成的结果全部拷完才往下继续运行
    cudaMemcpy(&result_array[0], device_weights_gradient_array, sizeof(double) * left_rows * right_cols, cudaMemcpyDeviceToHost);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };
    cudaFree(device_output_delta_array);
    cudaFree(device_input_transpose_array);
    cudaFree(device_weights_gradient_array);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cuda full connected layer backward failed";
        return -1;
    };
    //得到结果
    Matrix::Reshape(result_array, left_rows, right_cols, weights_gradient_array);
    Matrix::Add(biases_gradient_array, output_delta_array, biases_gradient_array);

    return 0;
}

//导入图片
cv::Mat LoadImage(std::string image_path) {
    cv::Mat image = cv::imread(image_path.c_str(), CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32FC3);
    //数据值归一化到0, 1之间
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    
    return image;
}

//保存图片
void SaveImage(std::string image_path, float* data, 
               int height, int width) {
    cv::Mat image(height, width, CV_32FC3, data);
    cv::threshold(image, image, 
                  /*threshold*/0, 
                  /*maxval*/0, 
                  /*mode*/cv::THRESH_TOZERO);
    //范围归一化到0, 255之间
    cv::normalize(image, image, 0.0, 255.0, cv::NORM_MINMAX);
    image.convertTo(image, CV_8UC3);
    cv::imwrite(image_path.c_str(), image);
}

//cudnn卷积操作 得到图像边缘
int ImageEdge(std::string input_image_path, 
              std::string output_image_path) {
    bool with_sigmoid = true;
    cv::Mat image = LoadImage(input_image_path);
    //1. handle
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    //2. 输入张量的描述 输入4d张量 NHWC为batch_size height width channels 
    cudnnTensorDescriptor_t input_descriptor;
    CheckCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CheckCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, 
                                          /*format*/CUDNN_TENSOR_NHWC,  
                                          /*dataType*/CUDNN_DATA_DOUBLE, 
                                          /*batch_size*/1, 
                                          /*channels*/3, 
                                          /*image_height*/image.rows, 
                                          /*image_width*/image.cols));
    //3. 卷积核的描述(形状 格式) NCHW为batch_size channels height width 
    cudnnFilterDescriptor_t filter_descriptor;
    CheckCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CheckCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor, 
                                          /*dataType*/CUDNN_DATA_DOUBLE, 
                                          /*format*/CUDNN_TENSOR_NCHW, 
                                          /*out_channels*/3, 
                                          /*in_channels*/3, 
                                          /*kernel_height*/3, 
                                          /*kernel_width*/3));
    //4. 卷积操作的描述(步长 填充)
    cudnnConvolutionDescriptor_t convolution_descriptor;
    CheckCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CheckCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 
                                               /*pad_height*/1, 
                                               /*pad_width*/1, 
                                               /*vertical_stride*/1, 
                                               /*horizontal_stride*/1, 
                                               /*dilation_height*/1, 
                                               /*dilation_width*/1, 
                                               /*mode*/CUDNN_CROSS_CORRELATION, 
                                               /*computeType*/CUDNN_DATA_DOUBLE));

    //5. 输出张量的描述 输入4d张量 NHWC为batch_size height width channels 
    cudnnTensorDescriptor_t output_descriptor;
    CheckCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CheckCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, 
                                          /*format*/CUDNN_TENSOR_NHWC,  
                                          /*dataType*/CUDNN_DATA_DOUBLE, 
                                          /*batch_size*/1, 
                                          /*channels*/3, 
                                          /*image_height*/image.rows, 
                                          /*image_width*/image.cols));
    //6. 卷积算法的描述(选择算法)
    /*
      cudnn_tion_fwd_algo_gemm 将卷积建模为显示矩阵乘法
      cudnn_tion_fwd_algo_fft  使用快速傅里叶变换fft进行卷积
      cudnn_tion_fwd_algo_winograd 使用Winograd算法执行卷积
    */
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    CheckCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                   input_descriptor, 
                                                   filter_descriptor, 
                                                   convolution_descriptor, 
                                                   output_descriptor, 
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 
                                                   /*memoryLimitlnBytes*/0, 
                                                   &convolution_algorithm));
    //7. 计算cuDNN需要多少内存
    size_t workspace_size;
    //计算卷积后图像的维度
    int batch_size;
    int channels;
    int height;
    int width;
    CheckCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, 
                                                     input_descriptor, 
                                                     filter_descriptor, 
                                                     &batch_size, 
                                                     &channels, 
                                                     &height, 
                                                     &width));
    LOG(INFO) << "输出图像(深度 高 宽): " << channels << "*"
              << height << "*" << width;
    CheckCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, 
                                                       input_descriptor, 
                                                       filter_descriptor, 
                                                       convolution_descriptor, 
                                                       output_descriptor, 
                                                       convolution_algorithm, 
                                                       &workspace_size));
    LOG(INFO) << "CUDNN 需要显存大小: " << workspace_size / 1048576.0 << "MB";
    
    //8. 分配内存
    void* device_workspace = NULL;
    float* device_input = NULL;
    float* device_filter = NULL;
    float* device_output = NULL;
    size_t image_size = batch_size * channels * height * width * sizeof(float);
    const float filter[3][3] = {
        {1, 1, 1}, 
        {1, -8, 1}, 
        {1, 1, 1}
    };
    float filters[3][3][3][3];  //NCHW
    for (int d = 0; d < 3; d++) {
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 3; h++) {
                for (int w = 0; w < 3; w++) {
                    filters[d][c][h][w] = filter[h][w];
                }
            }
        }
    }
    cudaMalloc((void**)&device_workspace, workspace_size);
    cudaMalloc((void**)&device_input, image_size);
    cudaMalloc((void**)&device_filter, sizeof(filters));
    cudaMalloc((void**)&device_output, image_size);

    //host to device
    cudaMemcpy(device_input, image.ptr<float>(0), image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_filter, filters, sizeof(filters), cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, image_size);
    
    //9. 执行前向卷积
    const float alpha = 1.0f;   //对卷积结果x*w进行缩放
    const float beta = 0.0f;    //对输出y进行缩放
    CheckCUDNN(cudnnConvolutionForward(cudnn, 
                                       &alpha, 
                                       input_descriptor, 
                                       device_input, 
                                       filter_descriptor, 
                                       device_filter, 
                                       convolution_descriptor, 
                                       convolution_algorithm, 
                                       device_workspace,   //如果不需要额外内存的卷积算法 这里为nullptr
                                       workspace_size, 
                                       &beta, 
                                       output_descriptor, 
                                       device_output));  
    
    if (with_sigmoid) {
        //激活函数的描述
        cudnnActivationDescriptor_t activation_descriptor;
        CheckCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
        CheckCUDNN(cudnnSetActivationDescriptor(activation_descriptor, 
                                                CUDNN_ACTIVATION_SIGMOID, 
                                                CUDNN_PROPAGATE_NAN, 
                                                /*relu_coef*/0)); 
        //前向sigmoid激活函数
        CheckCUDNN(cudnnActivationForward(cudnn, 
                                          activation_descriptor, 
                                          &alpha, 
                                          output_descriptor, 
                                          device_output, 
                                          &beta, 
                                          output_descriptor, 
                                          device_output));
        cudnnDestroyActivationDescriptor(activation_descriptor);
    } 

    float* output = new float[image_size];
    //输出结果送回host端
    cudaMemcpy(output, device_output, image_size, cudaMemcpyDeviceToHost);
    SaveImage(output_image_path, output, height, width);

    //10. 释放内存 
    delete []output;
    cudaFree(device_workspace);
    cudaFree(device_input);
    cudaFree(device_filter);
    cudaFree(device_output);
    
    //摧毁描述符
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);

    //摧毁cudnn句柄
    cudnnDestroy(cudnn);

    return 0;
}

//cudnn卷积层前向计算
int ConvolutionalForwardww(const Matrix3d& input_array, const Matrix4d& filter_array, 
                            Matrix3d& output_array, int batch_size, 
                            int input_channels, int input_height, int input_width, 
                            int filter_number, int filter_height, int filter_width, 
                            int zero_padding, int stride, 
                            int output_height, int output_width) { 
    //1. handle
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    //2. 输入张量的描述 输入4d张量 NCHW为batch_size channels height width  
    cudnnTensorDescriptor_t input_descriptor;
    CheckCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CheckCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, 
                                          /*format*/CUDNN_TENSOR_NCHW,  
                                          /*dataType*/CUDNN_DATA_DOUBLE, 
                                          /*batch_size*/batch_size, 
                                          /*channels*/input_channels, 
                                          /*image_height*/input_height, 
                                          /*image_width*/input_width));
    //3. 卷积核的描述(形状 格式) NCHW为batch_size channels height width 
    cudnnFilterDescriptor_t filter_descriptor;
    CheckCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CheckCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor, 
                                          /*dataType*/CUDNN_DATA_DOUBLE, 
                                          /*format*/CUDNN_TENSOR_NCHW, 
                                          /*out_channels*/filter_number, 
                                          /*in_channels*/input_channels, 
                                          /*kernel_height*/filter_height, 
                                          /*kernel_width*/filter_width));
    //4. 卷积操作的描述(步长 填充)
    cudnnConvolutionDescriptor_t convolution_descriptor;
    CheckCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CheckCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 
                                               /*pad_height*/zero_padding, 
                                               /*pad_width*/zero_padding, 
                                               /*vertical_stride*/stride, 
                                               /*horizontal_stride*/stride, 
                                               /*dilation_height*/1, 
                                               /*dilation_width*/1, 
                                               /*mode*/CUDNN_CROSS_CORRELATION, 
                                               /*computeType*/CUDNN_DATA_DOUBLE));

    //5. 输出张量的描述 输入4d张量 NCHW为batch_size channels height width  
    cudnnTensorDescriptor_t output_descriptor;
    CheckCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CheckCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, 
                                          /*format*/CUDNN_TENSOR_NCHW, 
                                          /*dataType*/CUDNN_DATA_DOUBLE, 
                                          /*batch_size*/batch_size, 
                                          /*channels*/filter_number, 
                                          /*image_height*/output_height, 
                                          /*image_width*/output_width));
    //6. 卷积算法的描述(选择算法)
    /*
      cudnn_tion_fwd_algo_gemm 将卷积建模为显示矩阵乘法
      cudnn_tion_fwd_algo_fft  使用快速傅里叶变换fft进行卷积
      cudnn_tion_fwd_algo_winograd 使用Winograd算法执行卷积
    */
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    CheckCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                   input_descriptor, 
                                                   filter_descriptor, 
                                                   convolution_descriptor, 
                                                   output_descriptor, 
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 
                                                   /*memoryLimitlnBytes*/0, 
                                                   &convolution_algorithm));
    //7. 计算cuDNN需要多少内存
    size_t workspace_size;
    CheckCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, 
                                                       input_descriptor, 
                                                       filter_descriptor, 
                                                       convolution_descriptor, 
                                                       output_descriptor, 
                                                       convolution_algorithm, 
                                                       &workspace_size));
    LOG(INFO) << "CUDNN 需要显存大小: " << workspace_size / 1048576.0 << "MB";

    //8. 分配内存
    void* device_workspace = NULL;
    float* device_input = NULL;
    float* device_filter = NULL;
    float* device_output = NULL;
    
    //输入 和 卷积核
    //double input[input_channels][input_height][input_width] = { 0.0 };
    double input[input_channels * input_height * input_width] = { 0.0 };
    double filter[filter_number][input_channels][filter_height][filter_width] = { 0.0 };
    Matrix1d output(filter_number * output_height * output_width);
    //std::shared_ptr<double> output(new double[filter_number * output_height * output_width], [](double* data) { 
    //        delete []data;
    //});
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < filter_number; i++) {
            for (int j = 0; j < input_channels; j++) {
                for (int k = 0; k < filter_height; k++) {
                    for (int z = 0; z < filter_width; z++) {
                        filter[i][j][z][k] = filter_array[i][j][z][k];
                    }
                }
            }
        }

        #pragma omp for schedule(static) 
        for (int i = 0; i < input_channels; i++) {
            for (int j = 0; j < input_height; j++) {
                for (int k = 0; k < input_width; k++) {
                    input[i * input_height * input_width + j * input_width + k] = input_array[i][j][k];
                }
            }
        }
    }
    
    cudaMalloc((void**)&device_workspace, workspace_size);
    cudaMalloc((void**)&device_input, sizeof(double)  * batch_size * input_channels * input_height * input_width);
    cudaMalloc((void**)&device_filter, sizeof(double) * filter_number * input_channels * filter_height * filter_width);
    cudaMalloc((void**)&device_output, sizeof(double) * batch_size * filter_number * output_height * output_width);

    //host to device
    cudaMemcpy(device_input, input, sizeof(double) * batch_size * input_channels * input_height * input_width, cudaMemcpyHostToDevice);
    cudaMemcpy(device_filter, filter, sizeof(double) * filter_number * input_channels * filter_height * filter_width, cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, sizeof(double) * batch_size * filter_number * output_height * output_width);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cudnn convolution forward algorithm failed";
        return -1;
    };
    
    //9. 执行前向卷积
    const float alpha = 1.0f;   //对卷积结果x*w进行缩放
    const float beta = 0.0f;    //对输出y进行缩放
    CheckCUDNN(cudnnConvolutionForward(cudnn, 
                                       &alpha, 
                                       input_descriptor, 
                                       device_input, 
                                       filter_descriptor, 
                                       device_filter, 
                                       convolution_descriptor, 
                                       convolution_algorithm, 
                                       device_workspace,   //如果不需要额外内存的卷积算法 这里为nullptr
                                       workspace_size, 
                                       &beta, 
                                       output_descriptor, 
                                       device_output));  
    
    //激活函数的描述
    cudnnActivationDescriptor_t activation_descriptor;
    CheckCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    CheckCUDNN(cudnnSetActivationDescriptor(activation_descriptor, 
                                            CUDNN_ACTIVATION_SIGMOID, 
                                            CUDNN_PROPAGATE_NAN, 
                                            /*relu_coef*/0)); 
    //前向ReLu激活函数
    CheckCUDNN(cudnnActivationForward(cudnn, 
                                      activation_descriptor, 
                                      &alpha, 
                                      output_descriptor, 
                                      device_output, 
                                      &beta, 
                                      output_descriptor, 
                                      device_output));
    cudnnDestroyActivationDescriptor(activation_descriptor);

    //输出结果送回host端
    cudaMemcpy(&output[0], device_output, filter_number * output_height * output_width, cudaMemcpyDeviceToHost);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cudnn convolution forward algorithm failed";
        return -1;
    };
    Matrix::MatrixShow(output);
    //10. 释放内存 
    cudaFree(device_workspace);
    cudaFree(device_input);
    cudaFree(device_filter);
    cudaFree(device_output);
    
    //摧毁描述符
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);

    //摧毁cudnn句柄
    cudnnDestroy(cudnn);

    //结果传回
    if (-1 == Matrix::Reshape(output, filter_number, 
                              output_height, output_width, 
                              output_array)) {
        LOG(ERROR) << "cudnn convolution forward algorithm failed";
        return -1;
    } 

    return 0;
}

//cudnn卷积层前向计算
int ConvolutionalForward(const Matrix3d& input_array, const Matrix4d& filter_array, 
                         Matrix3d& output_array, int batch_size, 
                         int input_channels, int input_height, int input_width, 
                         int filter_number, int filter_height, int filter_width, 
                         int zero_padding, int stride, 
                         int output_height, int output_width) { 
    //1. handle
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    //2. 输入张量的描述 输入4d张量 NCHW为batch_size channels height width  
    cudnnTensorDescriptor_t input_descriptor;
    CheckCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CheckCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, 
                                          /*format*/CUDNN_TENSOR_NCHW,  
                                          /*dataType*/CUDNN_DATA_DOUBLE, 
                                          /*batch_size*/batch_size, 
                                          /*channels*/1,  
                                          /*image_height*/28,  
                                          /*image_width*/28));
    //3. 卷积核的描述(形状 格式) NCHW为batch_size channels height width 
    cudnnFilterDescriptor_t filter_descriptor;
    CheckCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    CheckCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor, 
                                          /*dataType*/CUDNN_DATA_DOUBLE, 
                                          /*format*/CUDNN_TENSOR_NCHW, 
                                          /*out_channels*/3, 
                                          /*in_channels*/1, 
                                          /*kernel_height*/3, 
                                          /*kernel_width*/3));
    //4. 卷积操作的描述(步长 填充)
    cudnnConvolutionDescriptor_t convolution_descriptor;
    CheckCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CheckCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 
                                               /*pad_height*/1, 
                                               /*pad_width*/1, 
                                               /*vertical_stride*/1, 
                                               /*horizontal_stride*/1, 
                                               /*dilation_height*/1, 
                                               /*dilation_width*/1, 
                                               /*mode*/CUDNN_CROSS_CORRELATION, 
                                               /*computeType*/CUDNN_DATA_DOUBLE));

    //计算卷积后图像的维度
    int _batch_size;
    int _channels;
    int _height;
    int _width;
    CheckCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, 
                                                     input_descriptor, 
                                                     filter_descriptor, 
                                                     &_batch_size, 
                                                     &_channels, 
                                                     &_height, 
                                                     &_width));
    LOG(INFO) << "输出图像(深度 高 宽): " << _channels << "*"
              << _height << "*" << _width;

    //2. 输出张量的描述 输入4d张量 NCHW为batch_size channels height width  
    cudnnTensorDescriptor_t output_descriptor;
    CheckCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    CheckCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, 
                                          /*format*/CUDNN_TENSOR_NCHW, 
                                          /*dataType*/CUDNN_DATA_DOUBLE, 
                                          /*batch_size*/batch_size, 
                                          /*channels*/_channels, 
                                          /*image_height*/_height, 
                                          /*image_width*/_width));
    //6. 卷积算法的描述(选择算法)
    /*
      cudnn_tion_fwd_algo_gemm 将卷积建模为显示矩阵乘法
      cudnn_tion_fwd_algo_fft  使用快速傅里叶变换fft进行卷积
      cudnn_tion_fwd_algo_winograd 使用Winograd算法执行卷积
    */
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    CheckCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                   input_descriptor, 
                                                   filter_descriptor, 
                                                   convolution_descriptor, 
                                                   output_descriptor, 
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 
                                                   /*memoryLimitlnBytes*/0, 
                                                   &convolution_algorithm));
    //7. 计算cuDNN需要多少内存
    size_t workspace_size;
    CheckCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, 
                                                       input_descriptor, 
                                                       filter_descriptor, 
                                                       convolution_descriptor, 
                                                       output_descriptor, 
                                                       convolution_algorithm, 
                                                       &workspace_size));
    LOG(INFO) << "CUDNN 需要显存大小: " << workspace_size / 1048576.0 << "MB";

    //8. 分配内存
    void* device_workspace = NULL;
    float* device_input = NULL;
    float* device_filter = NULL;
    float* device_output = NULL;
    
    //输入 和 卷积核
    //double input[input_channels][input_height][input_width] = { 0.0 };
    double input[input_channels * input_height * input_width] = { 0.0 };
    Matrix1d output(_channels * _height * _width);
    //std::shared_ptr<double> output(new double[filter_number * output_height * output_width], [](double* data) { 
    //        delete []data;
    //});
    const float filters[3][3] = {
        {1, 1, 1}, 
        {1, -8, 1}, 
        {1, 1, 1}
    };
    float filter[3][1][3][3];  //NCHW
    for (int d = 0; d < 3; d++) {
        for (int c = 0; c < 1; c++) {
            for (int h = 0; h < 3; h++) {
                for (int w = 0; w < 3; w++) {
                    filter[d][c][h][w] = filters[h][w];
                }
            }
        }
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < input_channels; i++) {
            for (int j = 0; j < input_height; j++) {
                for (int k = 0; k < input_width; k++) {
                    input[i * input_height * input_width + j * input_width + k] = input_array[i][j][k];
                    //input[i][j][k] = input_array[i][j][k];
                }
            }
        }
    }
    
    cudaMalloc((void**)&device_workspace, workspace_size);
    cudaMalloc((void**)&device_input, sizeof(double)  * batch_size * input_channels * input_height * input_width);
    cudaMalloc((void**)&device_filter, sizeof(double) * 3*3*3 );
    cudaMalloc((void**)&device_output, sizeof(double) * batch_size * _channels * _height * _width);

    //host to device
    cudaMemcpy(device_input, input, sizeof(double) * batch_size * input_channels * input_height * input_width, cudaMemcpyHostToDevice);
    cudaMemcpy(device_filter, filter, sizeof(double) * 3*3*3, cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, sizeof(double) * batch_size * _channels * _height * _width);
    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cudnn convolution forward algorithm failed";
        return -1;
    };
    
    //9. 执行前向卷积
    const float alpha = 0.0f;   //对卷积结果x*w进行缩放
    const float beta = 0.0f;    //对输出y进行缩放
    CheckCUDNN(cudnnConvolutionForward(cudnn, 
                                       &alpha, 
                                       input_descriptor, 
                                       device_input, 
                                       filter_descriptor, 
                                       device_filter, 
                                       convolution_descriptor, 
                                       convolution_algorithm, 
                                       device_workspace,   //如果不需要额外内存的卷积算法 这里为nullptr
                                       workspace_size, 
                                       &beta, 
                                       output_descriptor, 
                                       device_output));  
    
    //激活函数的描述
    cudnnActivationDescriptor_t activation_descriptor;
    CheckCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    CheckCUDNN(cudnnSetActivationDescriptor(activation_descriptor, 
                                            CUDNN_ACTIVATION_SIGMOID, 
                                            CUDNN_PROPAGATE_NAN, 
                                            /*relu_coef*/0)); 
    //前向ReLu激活函数
    CheckCUDNN(cudnnActivationForward(cudnn, 
                                      activation_descriptor, 
                                      &alpha, 
                                      output_descriptor, 
                                      device_output, 
                                      &beta, 
                                      output_descriptor, 
                                      device_output));
    cudnnDestroyActivationDescriptor(activation_descriptor);

    //输出结果送回host端
    cudaMemcpy(&output[0], device_output, _channels * _height * _width, cudaMemcpyDeviceToHost);

    if (-1 == CheckCudaError()) {
        LOG(ERROR) << "cudnn convolution forward algorithm failed";
        return -1;
    };
    Matrix::MatrixShow(output);
    //10. 释放内存 
    cudaFree(device_workspace);
    cudaFree(device_input);
    cudaFree(device_filter);
    cudaFree(device_output);
    
    //摧毁描述符
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);

    //摧毁cudnn句柄
    cudnnDestroy(cudnn);


    return 0;
}








}       //namespace cuda
}       //namespace calculate
}       //namespace moon

