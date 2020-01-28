/*
 * =====================================================================================
 *
 *       Filename:  matrix_gpu.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年01月24日 15时23分07秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef CALCULATE_MATRIX_GPU_H_
#define CALCULATE_MATRIX_GPU_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>
#include <memory>
#include <map>
#include <tuple>

namespace calculate {
namespace cuda {
//类型别名
typedef std::vector<double> Matrix1d;
typedef std::vector<std::vector<double>> Matrix2d;
typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
typedef std::vector<std::vector<std::vector<std::vector<double>>>> Matrix4d;
typedef std::vector<uint8_t> ImageMatrix1d;
typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;
typedef std::vector<std::vector<std::vector<std::vector<uint8_t>>>> ImageMatrix4d;

//global variable 
const int kDepth = 50;
const int kHeight = 100;
const int kWidth = 100;

//初始化CUDA
bool InitializeCUDA();

//打印GPU硬件信息
void GpuInfoShow();

//检查cuda调用函数是否成功
int CheckCudaError();

//2d矩阵点积
int DotProduct(const Matrix2d& left_matrix, 
               const Matrix2d& right_matrix, 
               Matrix2d& result_matrix);
int DotProductw(const Matrix2d& left_matrix, 
               const Matrix2d& right_matrix, 
               Matrix2d& result_matrix);

//2d矩阵hadamark积
int HadamarkProduct(const Matrix2d& left_matrix, 
                    const Matrix2d& right_matrix, 
                    Matrix2d& result_matrix);

//3d矩阵hadamark积
int HadamarkProduct(const Matrix3d& left_matrix, 
                    const Matrix3d& right_matrix, 
                    Matrix3d& result_matrix);

//2d矩阵相加 
int Add(const Matrix2d& left_matrix, 
        const Matrix2d& right_matrix, 
        Matrix2d& result_matrix);

//3d矩阵相加 
int Add(const Matrix3d& left_matrix, 
        const Matrix3d& right_matrix, 
        Matrix3d& result_matrix);

//3d矩阵相加 
int AddFixed(const Matrix3d& left_matrix, 
             const Matrix3d& right_matrix, 
             Matrix3d& result_matrix);

//2d矩阵相减
int Subtract(const Matrix2d& left_matrix, 
             const Matrix2d& right_matrix, 
             Matrix2d& result_matrix);

//3d矩阵相减
int Subtract(const Matrix3d& left_matrix, 
             const Matrix3d& right_matrix, 
             Matrix3d& result_matrix);

//一个值乘以2d矩阵
int ValueMulMatrix(double value, 
                   const Matrix2d& source_matrix, 
                   Matrix2d& result_matrix);

//一个值乘以3d矩阵
int ValueMulMatrix(double value,  
                   const Matrix3d& source_matrix, 
                   Matrix3d& result_matrix);

//一个值减去一个2d矩阵
int ValueSubMatrix(double value, 
                   const Matrix2d& source_matrix, 
                   Matrix2d& result_matrix);

//一个值减去一个3d矩阵
int ValueMulMatrix(double value,  
                   const Matrix3d& source_matrix, 
                   Matrix3d& result_matrix);

//2d矩阵除以一个值
int MatrixDivValue(const Matrix2d& source_matrix, 
                   double value, 
                   Matrix2d& result_matrix);

//3d矩阵除以一个值
int MatrixDivValue(const Matrix3d& source_matrix, 
                   double value, 
                   Matrix3d& result_matrix);

//sigmoid激活函数的前向计算
int SigmoidForward(const Matrix2d& source_matrix, 
                   Matrix2d& result_matrix);

//sigmoid激活函数的反向计算
int SigmoidBackward(const Matrix2d& result_matrix, 
                    Matrix2d& delta_array);

//sigmoid激活函数的反向计算
int SigmoidImageBackward(const ImageMatrix2d& result_matrix, 
                         Matrix2d& delta_array);

//ReLu激活函数的前向计算
int ReLuForward2d(const Matrix2d& source_matrix, 
                  Matrix2d& result_matrix);

//ReLu激活函数的前向计算
int ReLuForward3d(const Matrix3d& source_matrix, 
                  Matrix3d& result_matrix);

//ReLu激活函数的反向计算
int ReLuBackward2d(const Matrix2d& source_matrix, 
                   Matrix2d& result_matrix);

//ReLu激活函数的反向计算
int ReLuBackward3d(const Matrix3d& source_matrix, 
                   Matrix3d& result_matrix);

//ReLu激活函数的反向计算
int ReLuImageBackward3d(const ImageMatrix3d& source_matrix, 
                        Matrix3d& result_matrix);

//全连接层的前向计算
int FullConnectedLayerForward(const Matrix2d& weights_array, 
                              const Matrix2d& input_array, 
                              const Matrix2d& biases_array,
                              Matrix2d& binomial_array, 
                              Matrix2d& output_array, 
                              bool is_input_layer = false,  
                              bool dropout = false, double p = 0.0); 

//全连接层的前向计算
int FullConnectedLayerForward(const Matrix2d& weights_array, 
                              const Matrix2d& input_array, 
                              const Matrix2d& biases_array,
                              Matrix2d& output_array, 
                              bool is_input_layer = false);  

//全连接层的反向计算
int FullConnectedLayerBackward(const Matrix2d& output_delta_array, 
                               const Matrix2d& weights_array, 
                               const Matrix2d& input_array,
                               const Matrix2d& binomial_array, 
                               Matrix2d& delta_array, 
                               Matrix2d& weights_gradient_array, 
                               Matrix2d& biases_gradient_array, 
                               bool is_input_layer =false, 
                               bool dropout = false, double p = 0.0); 

//全连接层的反向计算
int FullConnectedLayerBackward(const Matrix2d& output_delta_array, 
                               const Matrix2d& weights_array, 
                               const Matrix2d& input_array,
                               Matrix2d& delta_array, 
                               Matrix2d& weights_gradient_array, 
                               Matrix2d& biases_gradient_array); 




}          //namespace cuda
}          //namespace calculate

#endif     //CALCULATE_MATRIX_GPU_HPP_
