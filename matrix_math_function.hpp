/*
 * =====================================================================================
 *
 *       Filename:  matrix_math_function.hpp
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

#ifndef MOON_CALCULATE_MATRIX_CPU_HPP_
#define MOON_CALCULATE_MATRIX_CPU_HPP_

#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <vector>
#include <algorithm>
#include <map>
#include <tuple>
#include <memory>
#include <chrono>

#include <glog/logging.h>
#include <omp.h>

#define OPENMP_THREADS_NUMBER 6   //openmp并行线程数量

//模板类 
namespace moon {
namespace calculate {
namespace matrix {

template <typename DataType=double>
struct Matrix {
    //类型别名
    //主要是double 和uint8_t两种类型 
    typedef std::vector<DataType> Matrix1d;
    typedef std::vector<std::vector<DataType>> Matrix2d;
    typedef std::vector<std::vector<std::vector<DataType>>> Matrix3d;
    typedef std::vector<std::vector<std::vector<std::vector<DataType>>>> Matrix4d;
    typedef std::vector<std::vector<std::vector<std::vector<std::vector<DataType>>>>> Matrix5d;
    
    // 检查1d矩阵是否正确
    static bool MatrixCheck(const Matrix1d& left_matrix, 
                            bool is_write_logging);

    // 检查2d矩阵行列是否正确 结果矩阵不打印日志 源矩阵打印 
    static bool MatrixCheck(const std::vector<std::vector<uint8_t>>& matrix, 
                            bool is_write_logging);

    // 检查2d矩阵行列是否正确 结果矩阵不打印日志 源矩阵打印 
    static bool MatrixCheck(const std::vector<std::vector<double>>& matrix, 
                            bool is_write_logging);

    // 检查3d矩阵深度 行 列是否正确 结果矩阵不打印日志 源矩阵打印 
    static bool MatrixCheck(const std::vector<std::vector<std::vector<uint8_t>>>& matrix, 
                            bool is_write_logging);

    // 检查3d矩阵深度 行 列是否正确 结果矩阵不打印日志 源矩阵打印 
    static bool MatrixCheck(const std::vector<std::vector<std::vector<double>>>& matrix, 
                            bool is_write_logging);

    // 检查4d矩阵深度 行 列是否正确 结果矩阵不打印日志 源矩阵打印 
    static bool MatrixCheck(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix, 
                            bool is_write_logging);

    // 检查4d矩阵深度 行 列是否正确 结果矩阵不打印日志 源矩阵打印 
    static bool MatrixCheck(const std::vector<std::vector<std::vector<std::vector<double>>>>& matrix, 
                            bool is_write_logging);

    // 检查5d矩阵深度 行 列是否正确 结果矩阵不打印日志 源矩阵打印 
    static bool MatrixCheck(const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& matrix, 
                            bool is_write_logging);

    // 检查两个2d矩阵行 列是否正确
    static bool MatrixCheck(const Matrix2d& left_matrix, 
                            const Matrix2d& right_matrix, 
                            bool is_write_logging);

    // 检查两个3d矩阵 深度 行 列是否正确
    static bool MatrixCheck(const Matrix3d& left_matrix, 
                            const Matrix3d& right_matrix, 
                            bool is_write_logging);

    // 检查两个4d矩阵 深度 行 列是否正确
    static bool MatrixCheck(const Matrix4d& left_matrix, 
                            const Matrix4d& right_matrix, 
                            bool is_write_logging);

    // 检查两个2d矩阵行 列是否正确
    static bool MatrixCheck(const std::vector<std::vector<uint8_t>>& left_matrix, 
                            const std::vector<std::vector<double>>& right_matrix, 
                            bool is_write_logging);

    // 检查两个3d矩阵 深度 行 列是否正确
    static bool MatrixCheck(const std::vector<std::vector<std::vector<uint8_t>>>& left_matrix, 
                            const std::vector<std::vector<std::vector<double>>>& right_matrix, 
                            bool is_write_logging);

    // 检查两个4d矩阵 深度 行 列是否正确
    static bool MatrixCheck(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& left_matrix, 
                            const std::vector<std::vector<std::vector<std::vector<double>>>>& right_matrix, 
                            bool is_write_logging);

    // 检查2d矩阵行 列 是否是 期望的 行 列  
    static bool MatrixCheck(const std::vector<std::vector<uint8_t>>& matrix, 
                            int32_t rows, int32_t cols, 
                            bool is_write_logging);

    // 检查2d矩阵行 列 是否是 期望的 行 列  
    static bool MatrixCheck(const std::vector<std::vector<double>>& matrix, 
                            int32_t rows, int32_t cols, 
                            bool is_write_logging);

    // 检查3d矩阵深度 行 列 是否是 期望的 深度 行 列  
    static bool MatrixCheck(const std::vector<std::vector<std::vector<uint8_t>>>& matrix,
                            int32_t depth, int32_t height, int32_t width, 
                            bool is_write_logging);

    // 检查3d矩阵深度 行 列 是否是 期望的 深度 行 列  
    static bool MatrixCheck(const std::vector<std::vector<std::vector<double>>>& matrix,
                            int32_t depth, int32_t height, int32_t width, 
                            bool is_write_logging);

    // 检查4d矩阵深度 行 列 是否是 期望的 深度 行 列  
    static bool MatrixCheck(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix,
                            int32_t batch_size, int32_t depth, int32_t height, int32_t width, 
                            bool is_write_logging);

    // 检查4d矩阵深度 行 列 是否是 期望的 深度 行 列  
    static bool MatrixCheck(const std::vector<std::vector<std::vector<std::vector<double>>>>& matrix,
                            int32_t batch_size, int32_t depth, int32_t height, int32_t width, 
                            bool is_write_logging);

    // 检查3d矩阵是否值为全零
    static bool IsEmptyMatrix(const std::vector<std::vector<std::vector<uint8_t>>>& matrix);

    // 检查3d矩阵是否值为全零
    static bool IsEmptyMatrix(const std::vector<std::vector<std::vector<double>>>& matrix);

    // 检查4d矩阵是否值为全零
    static bool IsEmptyMatrix(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix);

    // 检查4d矩阵是否值为全零
    static bool IsEmptyMatrix(const std::vector<std::vector<std::vector<std::vector<double>>>>& matrix);

    // 返回元祖 2d矩阵的形状(高, 宽)
    static std::tuple<int32_t, int32_t> GetShape(const std::vector<std::vector<uint8_t>>& source_matrix);

    // 返回元祖 2d矩阵的形状(高, 宽)
    static std::tuple<int32_t, int32_t> GetShape(const std::vector<std::vector<double>>& source_matrix);

    // 返回元祖 3d矩阵的形状(深度, 高, 宽)
    static std::tuple<int32_t, int32_t, int32_t> GetShape(const std::vector<std::vector<std::vector<uint8_t>>>& source_matrix);

    // 返回元祖 3d矩阵的形状(深度, 高, 宽)
    static std::tuple<int32_t, int32_t, int32_t> GetShape(const std::vector<std::vector<std::vector<double>>>& source_matrix);

    // 返回元祖 4d矩阵的形状(批大小, 深度, 高, 宽)
    static std::tuple<int32_t, int32_t, int32_t, int32_t> GetShape(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& source_matrix);

    // 返回元祖 4d矩阵的形状(批大小, 深度, 高, 宽)
    static std::tuple<int32_t, int32_t, int32_t, int32_t> GetShape(const std::vector<std::vector<std::vector<std::vector<double>>>>& source_matrix);

    // 返回元祖 5d矩阵的形状(批大小, 深度, 高, 宽)
    static std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t> GetShape(const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& source_matrix);

    // 打印2d矩阵的形状(高 宽)
    static void ShapeShow(const Matrix2d& source_matrix);

    // 打印3d矩阵的形状(深度 高 宽)
    static void ShapeShow(const Matrix3d& source_matrix);
    
    // 打印4d矩阵的形状(批大小 深度 高 宽)
    static void ShapeShow(const Matrix4d& source_matrix);

    // 返回一个浮点型的2d矩阵
    static std::vector<std::vector<double>> ToDouble(const std::vector<std::vector<uint8_t>>& matrix);

    // 返回一个浮点型的3d矩阵
    static std::vector<std::vector<std::vector<double>>> ToDouble(const std::vector<std::vector<std::vector<uint8_t>>>& matrix);

    // 返回一个浮点型的4d矩阵
    static std::vector<std::vector<std::vector<std::vector<double>>>> ToDouble(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix);

    // 创建2维矩阵 初始值为0
    static int8_t CreateZeros(int32_t rows, int32_t cols, 
                              Matrix2d& matrix);

    // 创建2维矩阵 初始值为0
    static int8_t CreateZeros(const std::tuple<int32_t, int32_t>& shape, 
                              Matrix2d& matrix);

    // 创建3维矩阵 初始值为0
    static int8_t CreateZeros(int32_t depth, int32_t height, int32_t width, 
                              Matrix3d& matrix);

    // 创建3维矩阵 初始值为0
    static int8_t CreateZeros(const std::tuple<int32_t, int32_t, int32_t>& shape, 
                              Matrix3d& matrix);

    // 创建4维矩阵 初始值为0
    static int8_t CreateZeros(int32_t batch_size, int32_t depth, int32_t height, int32_t width, 
                              Matrix4d& matrix);

    // 创建4维矩阵 初始值为0
    static int8_t CreateZeros(const std::tuple<int32_t, int32_t, int32_t, int32_t>& shape, 
                              std::vector<std::vector<std::vector<std::vector<double>>>>& matrix);

    // 创建4维矩阵 初始值为0
    static int8_t CreateZeros(const std::tuple<int32_t, int32_t, int32_t, int32_t>& shape, 
                              std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix);

    // 创建5维矩阵 初始值为0
    static int8_t CreateZeros(int32_t batch_size, int32_t filter_number, int32_t depth, int32_t height, int32_t width, 
                              Matrix5d& matrix);

    // 创建5维矩阵 初始值为0
    static int8_t CreateZeros(const std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t>& shape, 
                              Matrix5d& matrix);

    // 创建2维矩阵 初始值为1
    static int8_t CreateOnes(int32_t rows, int32_t cols, 
                             Matrix2d& matrix);

    // 创建2维矩阵 初始值为1
    static int8_t CreateOnes(const std::tuple<int32_t, int32_t>& shape, 
                             Matrix2d& matrix);

    // 创建3维矩阵 初始值为1
    static int8_t CreateOnes(int32_t depth, int32_t height, int32_t width,  
                             Matrix3d& matrix);

    // 创建3维矩阵 初始值为1
    static int8_t CreateOnes(const std::tuple<int32_t, int32_t, int32_t>& shape, 
                             Matrix3d& matrix);

    // 打印二维矩阵
    static void MatrixShow(const Matrix1d& matrix);

    // 打印二维矩阵
    static void MatrixShow(const Matrix2d& matrix);
    
    // 打印三维矩阵
    static void MatrixShow(const Matrix3d& matrix);

    // 打印四维矩阵
    static void MatrixShow(const Matrix4d& matrix);

    // 打印图像二维矩阵
    static void ImageMatrixShow(const std::vector<std::vector<uint8_t>>& matrix);

    // 打印图像二维矩阵
    static void ImageMatrixShow(const std::vector<std::vector<std::vector<uint8_t>>>& matrix);

    // 2d矩阵相乘  dot product 点积
    static int8_t DotProduct(const Matrix2d& left_matrix, 
                             const Matrix2d& right_matrix, 
                             Matrix2d& result_matrix); 

    // 3d矩阵相乘  dot product 点积
    static int8_t DotProduct(const Matrix3d& left_matrix, 
                             const Matrix3d& right_matrix, 
                             Matrix2d& result_matrix); 

    // 3d矩阵相乘  dot product 点积
    static int8_t DotProduct(const Matrix3d& left_matrix, 
                             const Matrix3d& right_matrix, 
                             Matrix3d& result_matrix); 

    // 2d矩阵相乘 函数重载 输入为uint8_t类型 输出为double类型
    static int8_t DotProduct(const std::vector<std::vector<double>>& left_matrix, 
                             const std::vector<std::vector<uint8_t>>& right_matrix, 
                             std::vector<std::vector<double>>& result_matrix); 

    // 2d矩阵对应位置相乘 hadamark积 
    static int8_t HadamarkProduct(const Matrix2d& left_matrix, 
                                  const Matrix2d& right_matrix, 
                                  Matrix2d& result_matrix);

    // 3d矩阵对应位置相乘 hadamark积 
    static int8_t HadamarkProduct(const Matrix3d& left_matrix, 
                                  const Matrix2d& right_matrix, 
                                  Matrix3d& result_matrix);

    // 3d矩阵对应位置相乘 hadamark积 
    static int8_t HadamarkProduct(const Matrix3d& left_matrix, 
                                  const Matrix3d& right_matrix, 
                                  Matrix3d& result_matrix);

    // 4d矩阵对应位置相乘 hadamark积 
    static int8_t HadamarkProduct(const Matrix4d& left_matrix, 
                                  const Matrix3d& right_matrix, 
                                  Matrix4d& result_matrix);

    // 4d矩阵对应位置相乘 hadamark积 
    static int8_t HadamarkProduct(const Matrix4d& left_matrix, 
                                  const Matrix4d& right_matrix, 
                                  Matrix4d& result_matrix);

    // 2d矩阵对应位置相乘 hadamark积 函数重载 输入uint8_t 输出double
    static int8_t HadamarkProduct(const std::vector<std::vector<uint8_t>>& left_matrix, 
                                  const std::vector<std::vector<double>>& right_matrix, 
                                  std::vector<std::vector<double>>& result_matrix); 

    // 3d矩阵对应位置相乘 hadamark积 函数重载 输入uint8_t 输出double
    static int8_t HadamarkProduct(const std::vector<std::vector<std::vector<uint8_t>>>& left_matrix, 
                                  const std::vector<std::vector<std::vector<double>>>& right_matrix, 
                                  std::vector<std::vector<std::vector<double>>>& result_matrix); 

    // 4d矩阵对应位置相乘 hadamark积 函数重载 输入uint8_t 输出double
    static int8_t HadamarkProduct(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& left_matrix, 
                                  const std::vector<std::vector<std::vector<double>>>& right_matrix, 
                                  std::vector<std::vector<std::vector<std::vector<double>>>>& result_matrix); 

    // 2d矩阵相加 
    static int8_t Add(const Matrix2d& left_matrix, 
                      const Matrix2d& right_matrix, 
                      Matrix2d& result_matrix);

    // 3d矩阵相加 
    static int8_t Add(const Matrix3d& left_matrix, 
                      const Matrix3d& right_matrix, 
                      Matrix3d& result_matrix);

    // 2d矩阵相减
    static int8_t Subtract(const Matrix2d& left_matrix, 
                           const Matrix2d& right_matrix, 
                           Matrix2d& result_matrix);

    // 3d矩阵相减
    static int8_t Subtract(const Matrix3d& left_matrix, 
                           const Matrix3d& right_matrix, 
                           Matrix3d& result_matrix);

    // 2d矩阵reshape 成1d矩阵  
    static int8_t Reshape(const Matrix2d& source_matrix,
                          Matrix1d& result_matrix);

    // 2d矩阵reshape 成1d矩阵  
    static int8_t Reshape(const Matrix2d& source_matrix,
                          int32_t& rows, int32_t& cols, 
                          Matrix1d& result_matrix);

    // 2个2d矩阵reshape 成2个1d矩阵  
    static int8_t Reshape(const Matrix2d& left_matrix,
                          const Matrix2d& right_matrix, 
                          int32_t& rows, int32_t& cols, 
                          Matrix1d& result_left_matrix, 
                          Matrix1d& result_right_matrix); 

    // 2个2d矩阵reshape 成2个1d矩阵  
    static int8_t Reshape(const Matrix2d& left_matrix,
                          const Matrix2d& right_matrix, 
                          int32_t& left_rows, int32_t& left_cols, 
                          int32_t& right_cols, 
                          Matrix1d& result_left_matrix, 
                          Matrix1d& result_right_matrix); 

    // 3d矩阵reshape 成1d矩阵  
    static int8_t Reshape(const Matrix3d& source_matrix,
                          Matrix1d& result_matrix);

    // 3d矩阵reshape 成1d矩阵  
    static int8_t Reshape(const Matrix3d& source_matrix,
                          int32_t& depth, int32_t& height, 
                          int32_t& width, 
                          Matrix1d& result_matrix);

    // 2个3d矩阵reshape 成2个1d矩阵  
    static int8_t Reshape(const Matrix3d& left_matrix,
                          const Matrix3d& right_matrix, 
                          int32_t& depth, int32_t& height, 
                          int32_t& width, 
                          Matrix1d& result_left_matrix, 
                          Matrix1d& result_right_matrix); 

    // 4d矩阵reshape 成1d矩阵  
    static int8_t Reshape(const Matrix4d& source_matrix,
                          Matrix1d& result_matrix);
    
    // 1d矩阵reshape 成2d矩阵
    static int8_t Reshape(const Matrix1d& source_matrix,
                          int32_t rows, int32_t cols, 
                          Matrix2d& result_matrix);

    // 1d矩阵reshape 成3d矩阵
    static int8_t Reshape(const Matrix1d& source_matrix,
                          int32_t depth, int32_t height, 
                          int32_t width, 
                          Matrix3d& result_matrix);

    // 2d矩阵reshape 成2d矩阵 
    static int8_t Reshape(const Matrix2d& source_matrix,
                          int32_t rows, int32_t cols, 
                          Matrix2d& result_matrix);

    // 2d矩阵reshape 成3d矩阵  
    static int8_t Reshape(const Matrix2d& source_matrix,
                          int32_t depth, int32_t height, 
                          int32_t width, 
                          Matrix3d& result_matrix);

    // 3d矩阵reshape 成2d矩阵  
    static int8_t Reshape(const Matrix3d& source_matrix,
                          int32_t rows, int32_t cols, 
                          Matrix2d& result_matrix);

    // 4d矩阵reshape 成3d矩阵  
    static int8_t Reshape(const Matrix4d& source_matrix,
                          int32_t depth, int32_t rows, int32_t cols, 
                          Matrix3d& result_matrix);

    // 3d矩阵reshape 成4d矩阵  
    static int8_t Reshape(const Matrix3d& source_matrix,
                          int32_t batch_size, int32_t depth,
                          int32_t rows, int32_t cols, 
                          Matrix4d& result_matrix);

    // 2维矩阵的装置矩阵
    static int8_t Transpose(const Matrix2d& source_matrix, 
                            Matrix2d& result_matrix);

    // 3维矩阵的装置矩阵
    static int8_t Transpose(const Matrix3d& source_matrix, 
                            Matrix3d& result_matrix);

    // 1个值 乘以 一个2d矩阵
    static int8_t ValueMulMatrix(DataType value,  
                                 const Matrix2d& source_matrix, 
                                 Matrix2d& result_matrix);

    // 1个值 乘以 一个3d矩阵
    static int8_t ValueMulMatrix(DataType value,  
                                 const Matrix3d& source_matrix, 
                                 Matrix3d& result_matrix);

    // 1个值 减去 一个2d矩阵
    static int8_t ValueSubMatrix(DataType value,  
                                 const Matrix2d& source_matrix, 
                                 Matrix2d& result_matrix);

    // 1个值 减去 一个3d矩阵
    static int8_t ValueSubMatrix(DataType value,  
                                 const Matrix3d& source_matrix, 
                                 Matrix3d& result_matrix);

    // 一个2d矩阵 同除以 一个值
    static int8_t MatrixDivValue(const Matrix2d& source_matrix, 
                                 DataType value, 
                                 Matrix2d& result_matrix);

    // 一个3d矩阵 同除以 一个值
    static int8_t MatrixDivValue(const Matrix3d& source_matrix, 
                                 DataType value, 
                                 Matrix3d& result_matrix);

    // 计算2d矩阵的和
    static double Sum(const std::vector<std::vector<uint8_t>>& source_matrix);

    // 计算2d矩阵的和
    static double Sum(const std::vector<std::vector<double>>& source_matrix);

    // 计算3d矩阵的和
    static double Sum(const std::vector<std::vector<std::vector<uint8_t>>>& source_matrix);

    // 计算3d矩阵的和
    static double Sum(const std::vector<std::vector<std::vector<double>>>& source_matrix);

    // 得到2d矩阵的最大值
    static double Max(const Matrix2d& source_matrix);

    // 得到2d矩阵的最大值所在的行 列 索引 
    static std::tuple<int32_t, int32_t> GetMaxIndex(const Matrix2d& source_matrix);

    // 得到2d矩阵的最大值所在的行 列 索引 
    static std::vector<std::tuple<int32_t, int32_t>> GetMaxIndex(const Matrix3d& source_matrix);

    // 得到2d矩阵中值为0的所有索引
    static int8_t GetZeroIndex(const Matrix2d& source_matrix, 
                               std::vector<std::tuple<int32_t, int32_t>>& result_matrix);

    // 2dSoftMax层
    static int8_t SoftMax(const Matrix2d& source_matrix, 
                          Matrix2d& result_matrix);

    // 3dSoftMax层
    static int8_t SoftMax(const Matrix3d& source_matrix, 
                          Matrix3d& result_matrix);

    // 3dSoftMaxLoss 交叉熵损失函数 因为概率和信息量成反比 所以加负号 -(label*ln(output))的求和
    static double CrossEntropyLoss(const Matrix3d& output_array, 
                                   const Matrix3d& label, 
                                   Matrix3d& output_delta_array);

    // 3dSoftMaxLoss 交叉熵损失函数
    static double CrossEntropyLoss(const Matrix3d& output_array, 
                                   const Matrix3d& label);

    // 计算2d均方误差
    static double MeanSquareError(const Matrix2d& output_matrix, 
                                  const Matrix2d& label);

    // 计算3d均方误差
    static double MeanSquareError(const Matrix3d& output_matrix, 
                                  const Matrix3d& label);

    // 计算4d均方误差
    static double MeanSquareError(const Matrix4d& output_matrix, 
                                  const Matrix4d& label);

    // 得到两2d矩阵不相等的值个数
    static int32_t NotEqualTotal(const Matrix2d& output_matrix, 
                                 const Matrix2d& label);

    // 得到两3d矩阵不相等的值个数
    static int32_t NotEqualTotal(const Matrix3d& output_matrix, 
                                 const Matrix3d& label);

    // 得到两4d矩阵不相等的值个数
    static int32_t NotEqualTotal(const Matrix4d& output_matrix, 
                                 const Matrix4d& label);
    
    
    // 补0填充
    static int8_t ZeroPadding(const Matrix3d& source_matrix, 
                              int32_t zero_padding, 
                              Matrix3d& result_matrix);

    // 补0填充
    static int8_t ZeroPadding(const Matrix4d& source_matrix, 
                              int32_t zero_padding, 
                              Matrix4d& result_matrix);
    
    //将3d源矩阵copy到结果矩阵的roi区域 (x, y)起始行列 x+height y+width是结尾行列
    static int8_t CopyTo(const Matrix3d& source_matrix, 
                         int32_t x, int32_t y, 
                         int32_t height, int32_t width, 
                         Matrix3d& result_matrix);

    //将4d源矩阵copy到结果矩阵的roi区域 (x, y)起始行列 x+height y+width是结尾行列
    static int8_t CopyTo(const Matrix4d& source_matrix, 
                         int32_t x, int32_t y, 
                         int32_t height, int32_t width, 
                         Matrix4d& result_matrix);

    //将源2d矩阵copy到结果矩阵
    static int8_t CopyTo(const Matrix2d& source_matrix, 
                         int32_t& index, 
                         std::shared_ptr<double> result_matrix);

    //将源3d矩阵copy到结果矩阵
    static int8_t CopyTo(const Matrix3d& source_matrix, 
                         int32_t& index, 
                         std::shared_ptr<double> result_matrix);

    //将源矩阵copy到结果矩阵
    static int8_t CopyTo(std::shared_ptr<double> source_matrix, 
                         int32_t start_index, int32_t fc_layers_number, 
                         int32_t fc5_input_node, int32_t fc5_output_node, 
                         int32_t fc6_output_node, 
                         Matrix3d& weights_matrix, 
                         Matrix3d& biases_matrix);

    //将源矩阵copy到结果矩阵
    static int8_t CopyTo(std::shared_ptr<double> source_matrix, 
                         int32_t start_index, int32_t& end_index,  
                         int32_t filter_number, int32_t channel_number, 
                         int32_t filter_height, int32_t filter_width, 
                         Matrix4d& weights_matrix, 
                         Matrix1d& biases_matrix);

    //得到2d矩阵的ROI
    static int8_t GetROI(const std::vector<std::vector<uint8_t>>& source_matrix, 
                         int32_t x, int32_t y, 
                         int32_t height, int32_t width, 
                         int32_t stride, 
                         std::vector<std::vector<uint8_t>>& result_matrix);

    //得到2d矩阵的ROI
    static int8_t GetROI(const std::vector<std::vector<double>>& source_matrix, 
                         int32_t x, int32_t y, 
                         int32_t height, int32_t width, 
                         int32_t stride, 
                         std::vector<std::vector<double>>& result_matrix);

    //得到3d矩阵的ROI
    static int8_t GetROI(const std::vector<std::vector<std::vector<uint8_t>>>& source_matrix, 
                         int32_t x, int32_t y, 
                         int32_t height, int32_t width, 
                         int32_t stride, 
                         std::vector<std::vector<std::vector<uint8_t>>>& result_matrix);

    //得到3d矩阵的ROI
    static int8_t GetROI(const std::vector<std::vector<std::vector<double>>>& source_matrix, 
                         int32_t x, int32_t y, 
                         int32_t height, int32_t width, 
                         int32_t stride, 
                         std::vector<std::vector<std::vector<double>>>& result_matrix);

    //得到4d矩阵的ROI
    static int8_t GetROI(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& source_matrix, 
                         int32_t x, int32_t y, 
                         int32_t height, int32_t width, 
                         int32_t stride, 
                         std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& result_matrix);

    //得到4d矩阵的ROI
    static int8_t GetROI(const std::vector<std::vector<std::vector<std::vector<double>>>>& source_matrix, 
                         int32_t x, int32_t y, 
                         int32_t height, int32_t width, 
                         int32_t stride, 
                         std::vector<std::vector<std::vector<std::vector<double>>>>& result_matrix);

    //2d矩阵卷积运算 height width
    static int8_t Convolution(const Matrix2d& source_matrix, 
                              const Matrix2d& filter_matrix, 
                              Matrix2d& result_matrix,
                              double bias = 0.0, 
                              int32_t stride = 1);

    //3d矩阵卷积运算 channels height width
    static int8_t Convolution(const Matrix3d& source_matrix, 
                              const Matrix3d& filter_matrix, 
                              Matrix2d& result_matrix,
                              double bias = 0.0, 
                              int32_t stride = 1);
    
    //2d矩阵卷积运算 height width
    static int8_t Convolution(const std::vector<std::vector<uint8_t>>& source_matrix, 
                              const std::vector<std::vector<double>>& filter_matrix, 
                              std::vector<std::vector<double>>& result_matrix,
                              double bias = 0.0, 
                              int32_t stride = 1);

    //3d矩阵卷积运算 channels height width
    static int8_t Convolution(const std::vector<std::vector<std::vector<uint8_t>>>& source_matrix, 
                              const std::vector<std::vector<std::vector<double>>>& filter_matrix, 
                              std::vector<std::vector<double>>& result_matrix,
                              double bias = 0.0, 
                              int32_t stride = 1);

    //3d矩阵翻转 180度
    static int8_t Flip(const Matrix3d& source_matrix, 
                       Matrix3d& result_matrix);

    //4d矩阵翻转 180度
    static int8_t Flip(const Matrix4d& source_matrix, 
                       Matrix4d& result_matrix);

    //扩展误差项(敏感图) 还原为步长为1时对应的sensitivity map
    static int8_t ExpandSensitivityMap(const Matrix3d& input_sensitivity_matrix, 
                                       int32_t input_height,  int32_t input_width, 
                                       int32_t filter_number, int32_t filter_height, 
                                       int32_t filter_width,  
                                       int32_t output_height, int32_t output_width, 
                                       int32_t zero_padding,  int32_t stride, 
                                       Matrix3d& output_sensitivity_matrix);

    //4d卷积层前向计算 batch_size channels height width 
    static int8_t ConvolutionalForward(const Matrix4d& source_matrix, 
                                       const Matrix4d& filter_weights_matrix, 
                                       const Matrix1d& biases_matrix, 
                                       Matrix4d& result_matrix,
                                       int32_t stride = 1);

    //4d卷积层前向计算 batch_size channels height width 
    static int8_t ConvolutionalForward(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& source_matrix, 
                                       const std::vector<std::vector<std::vector<std::vector<double>>>>& filter_weights_matrix, 
                                       const std::vector<double>& biases_matrix, 
                                       std::vector<std::vector<std::vector<std::vector<double>>>>& result_matrix, 
                                       int32_t stride = 1, bool aa=false);

    //4d卷积层反向传播 误差传递 batch_size channels height width
    static int8_t ConvolutionalBpSensitivityMap(const Matrix4d& input_sensitivity_matrix, 
                                                const Matrix4d& weights_matrix, 
                                                const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& input_matrix, 
                                                Matrix4d& delta_matrix);

    //4d卷积层反向传播 误差传递 batch_size channels height width
    static int8_t ConvolutionalBpSensitivityMap(const Matrix4d& input_sensitivity_matrix, 
                                                const Matrix4d& weights_matrix, 
                                                const std::vector<std::vector<std::vector<std::vector<double>>>>& input_matrix, 
                                                Matrix4d& delta_matrix);
    
    //4d卷积层反向传播 计算梯度
    static int8_t ConvolutionalBpGradient(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& input_matrix, 
                                          const Matrix4d& sensitivity_matrix, 
                                          Matrix4d& weights_gradient_matrix, 
                                          Matrix1d& biases_gradient_matrix);

    //4d卷积层反向传播 计算梯度
    static int8_t ConvolutionalBpGradient(const std::vector<std::vector<std::vector<std::vector<double>>>>& input_matrix, 
                                          const Matrix4d& sensitivity_matrix, 
                                          Matrix4d& weights_gradient_matrix, 
                                          Matrix1d& biases_gradient_matrix);

    //2d矩阵最大池化前向计算 height width 
    static int8_t MaxPoolingForward(const Matrix2d& source_matrix, 
                                    int32_t filter_height, 
                                    int32_t filter_width, 
                                    int32_t stride, 
                                    Matrix2d& result_matrix); 

    //3d矩阵最大池化前向计算 channels height width 
    static int8_t MaxPoolingForward(const Matrix3d& source_matrix, 
                                    int32_t filter_height, 
                                    int32_t filter_width, 
                                    int32_t stride, 
                                    Matrix3d& result_matrix); 

    //4d矩阵最大池化前向计算 batch_size channels height width 
    static int8_t MaxPoolingForward(const Matrix4d& source_matrix, 
                                    int32_t filter_height, 
                                    int32_t filter_width, 
                                    int32_t stride, 
                                    Matrix4d& result_matrix); 

    //2d矩阵最大池化反向计算 height width 
    static int8_t MaxPoolingBackward(const Matrix2d& source_matrix, 
                                     const Matrix2d& sensitivity_matrix, 
                                     int32_t filter_height, 
                                     int32_t filter_width, 
                                     int32_t stride, 
                                     Matrix2d& result_matrix); 

    //3d矩阵最大池化反向计算 channels height width 
    static int8_t MaxPoolingBackward(const Matrix3d& source_matrix, 
                                     const Matrix3d& sensitivity_matrix, 
                                     int32_t filter_height, 
                                     int32_t filter_width, 
                                     int32_t stride, 
                                     Matrix3d& result_matrix); 

    //4d矩阵最大池化反向计算 batch_size channels height width 
    static int8_t MaxPoolingBackward(const Matrix4d& source_matrix, 
                                     const Matrix4d& sensitivity_matrix, 
                                     int32_t filter_height, 
                                     int32_t filter_width, 
                                     int32_t stride, 
                                     Matrix4d& result_matrix); 

    //2d矩阵全连接层前向计算 height width 
    static int8_t FullConnectedForward(const Matrix2d& source_matrix, 
                                       const Matrix2d& weights_matrix, 
                                       const Matrix2d& biases_matrix, 
                                       Matrix2d& result_matrix); 
    
    //3d矩阵全连接层前向计算 batch_size height width
    static int8_t FullConnectedForward(const Matrix3d& source_matrix, 
                                       const Matrix2d& weights_matrix, 
                                       const Matrix2d& biases_matrix, 
                                       Matrix3d& result_matrix, 
                                       bool is_output_layer); 

    //2d矩阵全连接层反向计算 height width
    static int8_t FullConnectedBackward(const Matrix2d& source_matrix, 
                                        const Matrix2d& weights_matrix, 
                                        const Matrix2d& output_delta_matrix,
                                        Matrix2d& delta_matrix, 
                                        Matrix2d& weights_gradient_matrix, 
                                        Matrix2d& biases_gradient_matrix); 

    //3d矩阵全连接层反向计算 batch_size height width
    static int8_t FullConnectedBackward(const Matrix3d& source_matrix, 
                                        const Matrix2d& weights_matrix, 
                                        const Matrix3d& output_delta_matrix,
                                        Matrix3d& delta_matrix, 
                                        Matrix2d& weights_gradient_matrix, 
                                        Matrix2d& biases_gradient_matrix); 

    //计算神经网络的输出层误差项
    static int8_t CalcOutputDiff(const Matrix2d& output_array, 
                                 const Matrix2d& label, 
                                 Matrix2d& delta_array);

    //计算神经网络的输出层误差项
    static int8_t CalcOutputDiff(const Matrix3d& output_array, 
                                 const Matrix3d& label, 
                                 Matrix3d& delta_array);

    //2d矩阵全连接层的梯度下降优化算法
    static int8_t GradientDescent(const Matrix2d& weights_gradient_array, 
                                  const Matrix2d& biases_gradient_array,
                                  double learning_rate, 
                                  int32_t batch_size, 
                                  Matrix2d& weights_array, 
                                  Matrix2d& biases_array);

    //3d矩阵卷积层的梯度下降优化算法
    static int8_t GradientDescent(const Matrix3d& weights_gradient_array, 
                                  const double biases_gradient,
                                  double learning_rate, 
                                  int32_t batch_size, 
                                  Matrix3d& weights_array, 
                                  double& biases);

    //4d矩阵卷积层的梯度下降优化算法
    static int8_t GradientDescent(const Matrix4d& weights_gradient_array, 
                                  const Matrix1d& biases_gradient_array,
                                  double learning_rate, 
                                  int32_t batch_size, 
                                  Matrix4d& weights_array, 
                                  Matrix1d& biases_array);
    
    //2d矩阵 全连接层的随机梯度下降优化算法升级版 带有动量
    static int8_t SGDMomentum(const Matrix2d& weights_gradient_array, 
                              const Matrix2d& biases_gradient_array,
                              Matrix2d& last_weights_gradient_array, 
                              Matrix2d& last_biases_gradient_array, 
                              double learning_rate, 
                              int32_t batch_size, 
                              double momentum, 
                              Matrix2d& weights_array, 
                              Matrix2d& biases_array);

    //4d矩阵 卷积层的随机梯度下降优化算法升级版 带动量
    static int8_t SGDMomentum(const Matrix4d& weights_gradient_array, 
                              const Matrix1d& biases_gradient_array,
                              Matrix4d& last_weights_gradient_array, 
                              Matrix1d& last_biases_gradient_array, 
                              double learning_rate, 
                              int32_t batch_size, 
                              double momentum, 
                              Matrix4d& weights_array, 
                              Matrix1d& biases_array);

    //评估 得到精度(准确率)
    static double Evaluate(const Matrix3d& output_array, 
                           const Matrix3d& test_label_data_set); 

    //得到2d最大值
    static int32_t ArgMax(const Matrix2d& output_array);

    //得到3d最大值
    static std::vector<int> ArgMax(const Matrix3d& output_array);






    
    
    //判断2d矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<double>>& matrix) {
        return false;
    }

    //判断2d矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<uint8_t>>& matrix) {
        return true;
    }

    //判断3d矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<std::vector<double>>>& matrix) {
        return false;
    }

    //判断3d矩阵是否是Uint8_t类型
    static bool IsImageMatrix(const std::vector<std::vector<std::vector<uint8_t>>>& matrix) {
        return true;
    }


};    //struct Matrix 




// 检查1d源矩阵或 结果矩阵是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix1d& matrix, 
                                   bool is_write_logging) {
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix is empty";
        return false;
    }

    return true;
}
                                
// 检查2d源矩阵或 结果矩阵行 列是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<uint8_t>>& matrix, 
                                   bool is_write_logging) {
    //先判断 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols is empty";
        return false;
    }
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    //再来判断每行的列是否相同
    for (int i = 0; i < rows; i++) {
        if (cols != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols of each row is not equal";
            return false;
        }
    }

    return true;
}

// 检查2d源矩阵或 结果矩阵行 列是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<double>>& matrix, 
                                   bool is_write_logging) {
    //先判断 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols is empty";
        return false;
    }
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    //再来判断每行的列是否相同
    for (int i = 0; i < rows; i++) {
        if (cols != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols of each row is not equal";
            return false;
        }
    }

    return true;
}

// 检查3d源矩阵或 结果矩阵深度 行 列是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<uint8_t>>>& matrix, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channel is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }
    
    int depth = matrix.size();
    int height = matrix[0].size();
    int width = matrix[0][0].size();
    //再来判断每行的列是否相同
    for (int i = 0; i < depth; i++) {
        if (height != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < height; j++) {
            if (width != matrix[i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查3d源矩阵或 结果矩阵深度 行 列是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<double>>>& matrix, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channel is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }
    
    int depth = matrix.size();
    int height = matrix[0].size();
    int width = matrix[0][0].size();
    //再来判断每行的列是否相同
    for (int i = 0; i < depth; i++) {
        if (height != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < height; j++) {
            if (width != matrix[i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查4d源矩阵或 结果矩阵深度 行 列是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix batch size is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channels is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }
    
    int depth = matrix[0].size();
    int height = matrix[0][0].size();
    int width = matrix[0][0][0].size();
    //再来判断每行的列是否相同
    for (int i = 0; i < depth; i++) {
        if (height != matrix[0][i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < height; j++) {
            if (width != matrix[0][i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查4d源矩阵或 结果矩阵深度 行 列是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<std::vector<double>>>>& matrix, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix batch size is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channels is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }
    
    int depth = matrix[0].size();
    int height = matrix[0][0].size();
    int width = matrix[0][0][0].size();
    //再来判断每行的列是否相同
    for (int i = 0; i < depth; i++) {
        if (height != matrix[0][i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < height; j++) {
            if (width != matrix[0][i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查5d源矩阵或 结果矩阵深度 行 列是否正确  是源矩阵就打印日志 结果矩阵就不打印日志 
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& matrix, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix batch size is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix filter number is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channels is empty";
        return false;
    }
    if (0 == matrix[0][0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0][0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }
    
    return true;
}



// 检查2d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix2d& left_matrix, 
                                   const Matrix2d& right_matrix, 
                                   bool is_write_logging) {
    //先分别检查两个矩阵本身行列对不对
    if (!MatrixCheck(left_matrix, is_write_logging)) {
        return false;
    }
    if (!MatrixCheck(right_matrix, is_write_logging)) {
        return false;
    }

    //再来判断两个矩阵 行 列是否相同
    if (left_matrix.size() != right_matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows is not equal";
        return false;
    }

    for (int i = 0; i < left_matrix.size(); i++) {
        if (left_matrix[i].size() != right_matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
            return false;
        }
    }

    return true;
}

// 检查3d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix3d& left_matrix, 
                                   const Matrix3d& right_matrix, 
                                   bool is_write_logging) {
    //先分别检查两个矩阵本身行列对不对
    if (!MatrixCheck(left_matrix, is_write_logging)) {
        return false;
    }
    if (!MatrixCheck(right_matrix, is_write_logging)) {
        return false;
    }

    //再来判断两个矩阵 深度 行 列是否相同
    if (left_matrix.size() != right_matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    for (int i = 0; i < left_matrix.size(); i++) {
        if (left_matrix[i].size() != right_matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < left_matrix[i].size(); j++) {
            if (left_matrix[i][j].size() != right_matrix[i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查4d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const Matrix4d& left_matrix, 
                                   const Matrix4d& right_matrix, 
                                   bool is_write_logging) {
    //先分别检查两个矩阵本身行列对不对
    if (!MatrixCheck(left_matrix, is_write_logging)) {
        return false;
    }
    if (!MatrixCheck(right_matrix, is_write_logging)) {
        return false;
    }

    //再来判断两个矩阵 深度 行 列是否相同
    if (left_matrix[0].size() != right_matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    for (int i = 0; i < left_matrix[0].size(); i++) {
        if (left_matrix[0][i].size() != right_matrix[0][i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < left_matrix[0][i].size(); j++) {
            if (left_matrix[0][i][j].size() != right_matrix[0][i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查两个2d矩阵行 列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<uint8_t>>& left_matrix, 
                                   const std::vector<std::vector<double>>& right_matrix, 
                                   bool is_write_logging) {
    //先分别检查两个矩阵本身行列对不对
    if (!MatrixCheck(left_matrix, is_write_logging)) {
        return false;
    }
    if (!MatrixCheck(right_matrix, is_write_logging)) {
        return false;
    }

    //再来判断两个矩阵 行 列是否相同
    if (left_matrix.size() != right_matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows is not equal";
        return false;
    }

    for (int i = 0; i < left_matrix.size(); i++) {
        if (left_matrix[i].size() != right_matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
            return false;
        }
    }

    return true;
}

// 检查两个3d矩阵 深度 行 列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<uint8_t>>>& left_matrix, 
                                   const std::vector<std::vector<std::vector<double>>>& right_matrix, 
                                   bool is_write_logging) {
    //先分别检查两个矩阵本身行列对不对
    if (!MatrixCheck(left_matrix, is_write_logging)) {
        return false;
    }
    if (!MatrixCheck(right_matrix, is_write_logging)) {
        return false;
    }

    //再来判断两个矩阵 深度 行 列是否相同
    if (left_matrix.size() != right_matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    for (int i = 0; i < left_matrix.size(); i++) {
        if (left_matrix[i].size() != right_matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < left_matrix[i].size(); j++) {
            if (left_matrix[i][j].size() != right_matrix[i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查4d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& left_matrix, 
                                   const std::vector<std::vector<std::vector<std::vector<double>>>>& right_matrix, 
                                   bool is_write_logging) {
    //先分别检查两个矩阵本身行列对不对
    if (!MatrixCheck(left_matrix, is_write_logging)) {
        return false;
    }
    if (!MatrixCheck(right_matrix, is_write_logging)) {
        return false;
    }

    //再来判断两个矩阵 深度 行 列是否相同
    if (left_matrix[0].size() != right_matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    for (int i = 0; i < left_matrix[0].size(); i++) {
        if (left_matrix[0][i].size() != right_matrix[0][i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < left_matrix[0][i].size(); j++) {
            if (left_matrix[0][i][j].size() != right_matrix[0][i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查2d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<uint8_t>>& matrix, 
                                   int32_t rows, int32_t cols,  
                                   bool is_write_logging) {
    //先判断 行 和 列是否是空
    if (rows <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input rows is empty";
        return false;
    }
    if (cols <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input cols is empty";
        return false;
    }

    //判断矩阵的 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols is empty";
        return false;
    }

    //看看行是否是要求的行
    if (rows != matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows is not equal";
        return false;
    }
    
    //看看列是否是要求的行
    for (int i = 0; i < matrix.size(); i++) {
        if (cols != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
            return false;
        }
    }

    return true;
}

// 检查2d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<double>>& matrix, 
                                   int32_t rows, int32_t cols,  
                                   bool is_write_logging) {
    //先判断 行 和 列是否是空
    if (rows <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input rows is empty";
        return false;
    }
    if (cols <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input cols is empty";
        return false;
    }

    //判断矩阵的 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix rows is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix cols is empty";
        return false;
    }

    //看看行是否是要求的行
    if (rows != matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows is not equal";
        return false;
    }
    
    //看看列是否是要求的行
    for (int i = 0; i < matrix.size(); i++) {
        if (cols != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
            return false;
        }
    }

    return true;
}

// 检查3d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<uint8_t>>>& matrix, 
                                   int32_t depth, int32_t height, int32_t width, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (depth <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input depth is empty";
        return false;
    }
    if (height <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input height is empty";
        return false;
    }
    if (width <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input width is empty";
        return false;
    }

    //判断矩阵的 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channel is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }

    //看看行是否是要求的行
    if (depth != matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    //看看列是否是要求的行
    for (int i = 0; i < matrix.size(); i++) {
        if (height != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < matrix[i].size(); j++) {
            if (width != matrix[i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查3d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<double>>>& matrix, 
                                   int32_t depth, int32_t height, int32_t width, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (depth <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input depth is empty";
        return false;
    }
    if (height <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input height is empty";
        return false;
    }
    if (width <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input width is empty";
        return false;
    }

    //判断矩阵的 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channel is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }

    //看看行是否是要求的行
    if (depth != matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    //看看列是否是要求的行
    for (int i = 0; i < matrix.size(); i++) {
        if (height != matrix[i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < matrix[i].size(); j++) {
            if (width != matrix[i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查4d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix, 
                                   int32_t batch_size, int32_t depth, int32_t height, int32_t width, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (batch_size <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input depth is empty";
        return false;
    }
    if (depth <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input depth is empty";
        return false;
    }
    if (height <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input height is empty";
        return false;
    }
    if (width <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input width is empty";
        return false;
    }

    //判断矩阵的 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix batch size is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channels is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }

    //看看行是否是要求的行
    if (depth != matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    //看看列是否是要求的行
    for (int i = 0; i < matrix[0].size(); i++) {
        if (height != matrix[0][i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < matrix[0][i].size(); j++) {
            if (width != matrix[0][i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查3d矩阵通道行列是否正确
template <typename DataType>
bool Matrix<DataType>::MatrixCheck(const std::vector<std::vector<std::vector<std::vector<double>>>>& matrix, 
                                   int32_t batch_size, int32_t depth, int32_t height, int32_t width, 
                                   bool is_write_logging) {
    //先判断 深度 行 和 列是否是空
    if (batch_size <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input depth is empty";
        return false;
    }
    if (depth <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input depth is empty";
        return false;
    }
    if (height <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input height is empty";
        return false;
    }
    if (width <= 0) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input width is empty";
        return false;
    }

    //判断矩阵的 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix batch size is empty";
        return false;
    }
    if (0 == matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix channels is empty";
        return false;
    }
    if (0 == matrix[0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix height is empty";
        return false;
    }
    if (0 == matrix[0][0][0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, input matrix width is empty";
        return false;
    }

    //看看行是否是要求的行
    if (depth != matrix[0].size()) {
        LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices channel is not equal";
        return false;
    }
    
    //看看列是否是要求的行
    for (int i = 0; i < matrix[0].size(); i++) {
        if (height != matrix[0][i].size()) {
            LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices rows of each channel is not equal";
            return false;
        }
        for (int j = 0; j < matrix[0][i].size(); j++) {
            if (width != matrix[0][i][j].size()) {
                LOG_IF(ERROR, is_write_logging) << "matrix check failed, two matrices cols of each row is not equal";
                return false;
            }
        }
    }

    return true;
}

// 检查4d矩阵是否为空
template <typename DataType>
bool Matrix<DataType>::IsEmptyMatrix(const std::vector<std::vector<std::vector<uint8_t>>>& matrix) {
    auto shape = GetShape(matrix);
    if (shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "check is empty matrix failed";
        return true;
    }
    
    std::vector<std::vector<std::vector<uint8_t>>> _matrix;
    CreateZeros(shape, _matrix);

    if (matrix == _matrix) {
        return true;
    }

    return false;
}

// 检查4d矩阵是否为空
template <typename DataType>
bool Matrix<DataType>::IsEmptyMatrix(const std::vector<std::vector<std::vector<double>>>& matrix) {
    auto shape = GetShape(matrix);
    if (shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "check is empty matrix failed";
        return true;
    }
    
    std::vector<std::vector<std::vector<double>>> _matrix;
    CreateZeros(shape, _matrix);

    if (matrix == _matrix) {
        return true;
    }

    return false;
}

// 检查4d矩阵是否为空
template <typename DataType>
bool Matrix<DataType>::IsEmptyMatrix(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix) {
    auto shape = GetShape(matrix);
    if (shape == std::make_tuple(0, 0, 0, 0)) {
        LOG(ERROR) << "check is empty matrix failed";
        return true;
    }
    
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> _matrix;
    CreateZeros(shape, _matrix);

    if (matrix == _matrix) {
        return true;
    }

    return false;
}

// 检查4d矩阵是否为空
template <typename DataType>
bool Matrix<DataType>::IsEmptyMatrix(const std::vector<std::vector<std::vector<std::vector<double>>>>& matrix) {
    auto shape = GetShape(matrix);
    if (shape == std::make_tuple(0, 0, 0, 0)) {
        LOG(ERROR) << "check is empty matrix failed";
        return true;
    }
    
    std::vector<std::vector<std::vector<std::vector<double>>>> _matrix;
    CreateZeros(shape, _matrix);

    if (matrix == _matrix) {
        return true;
    }

    return false;
}



//得到二维矩阵的形状
template <typename DataType>
std::tuple<int32_t, int32_t> Matrix<DataType>::GetShape(const std::vector<std::vector<uint8_t>>& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return std::make_tuple(0, 0);
    }
    
    int height = source_matrix.size();
    int width = source_matrix[0].size();
    
    return std::make_tuple(height, width);
}

//得到二维矩阵的形状
template <typename DataType>
std::tuple<int32_t, int32_t> Matrix<DataType>::GetShape(const std::vector<std::vector<double>>& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return std::make_tuple(0, 0);
    }
    
    int height = source_matrix.size();
    int width = source_matrix[0].size();
    
    return std::make_tuple(height, width);
}

//得到三维矩阵的形状
template <typename DataType>
std::tuple<int32_t, int32_t, int32_t> Matrix<DataType>::GetShape(const std::vector<std::vector<std::vector<uint8_t>>>& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return std::make_tuple(0, 0, 0);
    }
    
    int channel_number = source_matrix.size();
    int height = source_matrix[0].size();
    int width = source_matrix[0][0].size();
    
    return std::make_tuple(channel_number, height, width);
}

//得到三维矩阵的形状
template <typename DataType>
std::tuple<int32_t, int32_t, int32_t> Matrix<DataType>::GetShape(const std::vector<std::vector<std::vector<double>>>& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return std::make_tuple(0, 0, 0);
    }
    
    int channel_number = source_matrix.size();
    int height = source_matrix[0].size();
    int width = source_matrix[0][0].size();
    
    return std::make_tuple(channel_number, height, width);
}

//得到四维矩阵的形状
template <typename DataType>
std::tuple<int32_t, int32_t, int32_t, int32_t> Matrix<DataType>::GetShape(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return std::make_tuple(0, 0, 0, 0);
    }
    
    int batch_size = source_matrix.size();
    int channel_number = source_matrix[0].size();
    int height = source_matrix[0][0].size();
    int width = source_matrix[0][0][0].size();
    
    return std::make_tuple(batch_size, channel_number, height, width);
}

//得到四维矩阵的形状
template <typename DataType>
std::tuple<int32_t, int32_t, int32_t, int32_t> Matrix<DataType>::GetShape(const std::vector<std::vector<std::vector<std::vector<double>>>>& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return std::make_tuple(0, 0, 0, 0);
    }

    int batch_size = source_matrix.size();
    int channel_number = source_matrix[0].size();
    int height = source_matrix[0][0].size();
    int width = source_matrix[0][0][0].size();
    
    return std::make_tuple(batch_size, channel_number, height, width);
}

//得到五维矩阵的形状
template <typename DataType>
std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t> Matrix<DataType>::GetShape(const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return std::make_tuple(0, 0, 0, 0, 0);
    }

    int batch_size = source_matrix.size();
    int filter_number = source_matrix[0].size();
    int channel_number = source_matrix[0][0].size();
    int height = source_matrix[0][0][0].size();
    int width = source_matrix[0][0][0][0].size();
    
    return std::make_tuple(batch_size, filter_number, channel_number, height, width);
}


//打印二维矩阵的形状
template <typename DataType>
void Matrix<DataType>::ShapeShow(const Matrix2d& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return ;
    }
    
    int height = source_matrix.size();
    int width = source_matrix[0].size();
    
    LOG(INFO) << "源矩阵的shape(高 宽): " << height << "*" << width;
}

//打印三维矩阵的形状
template <typename DataType>
void Matrix<DataType>::ShapeShow(const Matrix3d& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return ;
    }
    
    int channel_number = source_matrix.size();
    int height = source_matrix[0].size();
    int width = source_matrix[0][0].size();
    
    LOG(INFO) << "源矩阵的shape(深度 高 宽): " << channel_number
              << "*" << height << "*" << width;
}

//打印三维矩阵的形状
template <typename DataType>
void Matrix<DataType>::ShapeShow(const Matrix4d& source_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix shape failed";
        return ;
    }
    
    int batch_size = source_matrix.size();
    int channel_number = source_matrix[0].size();
    int height = source_matrix[0][0].size();
    int width = source_matrix[0][0][0].size();
    
    LOG(INFO) << "源矩阵的shape(批大小 深度 高 宽): " << batch_size << "*"
              << channel_number << "*" << height << "*" << width;
}


//返回一个浮点型的2d矩阵
template <typename DataType>
std::vector<std::vector<double>> Matrix<DataType>::ToDouble(const std::vector<std::vector<uint8_t>>& matrix) {
    //先判断 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG(ERROR) << "matrix check failed, input matrix rows is empty";
        return std::vector<std::vector<double>>(0, std::vector<double>(0));
    }
    if (0 == matrix[0].size()) {
        LOG(ERROR) << "matrix check failed, input matrix cols is empty";
        return std::vector<std::vector<double>>(0, std::vector<double>(0));
    }

    int rows = matrix.size();
    int cols = matrix[0].size();
    
    std::vector<std::vector<double>> double_array(rows, std::vector<double>(cols));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double_array[i][j] = matrix[i][j];           
            }
        }
    }

    return double_array;
}


//返回一个浮点型的3d矩阵
template <typename DataType>
std::vector<std::vector<std::vector<double>>> Matrix<DataType>::ToDouble(const std::vector<std::vector<std::vector<uint8_t>>>& matrix) {
    //先判断 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG(ERROR) << "matrix check failed, input matrix channel is empty";
        return std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0)));
    }
    if (0 == matrix[0].size()) {
        LOG(ERROR) << "matrix check failed, input matrix height is empty";
        return std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0)));
    }
    if (0 == matrix[0][0].size()) {
        LOG(ERROR) << "matrix check failed, input matrix width is empty";
        return std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0)));
    }

    int depth = matrix.size();
    int height = matrix[0].size();
    int width = matrix[0][0].size();
    
    std::vector<std::vector<std::vector<double>>> double_array(depth, std::vector<std::vector<double>>(height, 
                                                               std::vector<double>(width)));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    double_array[i][j][k] = matrix[i][j][k];           
                }
            }
        }
    }

    return double_array;
}

//返回一个浮点型的4d矩阵
template <typename DataType>
std::vector<std::vector<std::vector<std::vector<double>>>> Matrix<DataType>::ToDouble(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix) {
    //先判断 深度 行 和 列是否是空
    if (0 == matrix.size()) {
        LOG(ERROR) << "matrix check failed, input matrix batch size is empty";
        return  std::vector<std::vector<std::vector<std::vector<double>>>>(0, 
                            std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0))));
    }
    if (0 == matrix[0].size()) {
        LOG(ERROR) << "matrix check failed, input matrix channel is empty";
        return  std::vector<std::vector<std::vector<std::vector<double>>>>(0, 
                            std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0))));
    }
    if (0 == matrix[0][0].size()) {
        LOG(ERROR) << "matrix check failed, input matrix height is empty";
        return  std::vector<std::vector<std::vector<std::vector<double>>>>(0, 
                            std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0))));
    }
    if (0 == matrix[0][0][0].size()) {
        LOG(ERROR) << "matrix check failed, input matrix width is empty";
        return  std::vector<std::vector<std::vector<std::vector<double>>>>(0, 
                            std::vector<std::vector<std::vector<double>>>(0, 
                                    std::vector<std::vector<double>>(0, std::vector<double>(0))));
    }

    int batch_size = matrix.size();
    int depth = matrix[0].size();
    int height = matrix[0][0].size();
    int width = matrix[0][0].size();
    
    std::vector<std::vector<std::vector<std::vector<double>>>> double_array(batch_size, 
                                                               std::vector<std::vector<std::vector<double>>>(depth,
                                                               std::vector<std::vector<double>>(height, 
                                                               std::vector<double>(width))));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int n = 0; n < batch_size; n++) {
            for (int i = 0; i < depth; i++) {
                for (int j = 0; j < height; j++) {
                    for (int k = 0; k < width; k++) {
                        double_array[n][i][j][k] = matrix[n][i][j][k];           
                    }
                }
            }
        }
    }

    return double_array;
}

// 创建2d矩阵 初始值0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(int32_t rows, int32_t cols, 
                                     Matrix2d& matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix2d(rows, Matrix1d(cols, 0));

    return 0;
}

// 创建2d矩阵 初始值0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(const std::tuple<int32_t, int32_t>& shape, 
                                     Matrix2d& matrix) {
    int rows = -1;
    int cols = -1;
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix2d(rows, Matrix1d(cols, 0));

    return 0;
}

// 创建3维矩阵 初始值为0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(int32_t depth, int32_t height, int32_t width, 
                                     Matrix3d& matrix) {
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width, 0)));

    return 0;
}

// 创建3维矩阵 初始值为0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(const std::tuple<int32_t, int32_t, int32_t>& shape, 
                                     Matrix3d& matrix) {
    int depth = -1;
    int height = -1;
    int width = -1;
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width, 0)));

    return 0;
}

// 创建4维矩阵 初始值为0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(int32_t batch_size, int32_t depth, int32_t height, int32_t width, 
                                     Matrix4d& matrix) {
    if (batch_size <= 0) {
        LOG(ERROR) << "create matrix failed, input batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix4d(batch_size, Matrix3d(depth, Matrix2d(height, Matrix1d(width, 0))));

    return 0;
}

// 创建4维矩阵 初始值为0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(const std::tuple<int32_t, int32_t, int32_t, int32_t>& shape, 
                                     std::vector<std::vector<std::vector<std::vector<double>>>>& matrix) {
    int batch_size = -1;
    int depth = -1;
    int height = -1;
    int width = -1;
    std::tie(batch_size, depth, height, width) = shape;
    if (batch_size <= 0) {
        LOG(ERROR) << "create matrix failed, input batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = std::vector<std::vector<std::vector<std::vector<double>>>>(batch_size, 
                                                            std::vector<std::vector<std::vector<double>>>(depth,
                                                            std::vector<std::vector<double>>(height, 
                                                            std::vector<double>(width))));

    return 0;
}

// 创建4维矩阵 初始值为0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(const std::tuple<int32_t, int32_t, int32_t, int32_t>& shape, 
                                     std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& matrix) {
    int batch_size = -1;
    int depth = -1;
    int height = -1;
    int width = -1;
    std::tie(batch_size, depth, height, width) = shape;
    if (batch_size <= 0) {
        LOG(ERROR) << "create matrix failed, input batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = std::vector<std::vector<std::vector<std::vector<uint8_t>>>>(batch_size, 
                                                            std::vector<std::vector<std::vector<uint8_t>>>(depth,
                                                            std::vector<std::vector<uint8_t>>(height, 
                                                            std::vector<uint8_t>(width))));

    return 0;
}

// 创建5维矩阵 初始值为0
template <typename DataType>
int8_t Matrix<DataType>::CreateZeros(int32_t batch_size, int32_t filter_number, int32_t depth, int32_t height, int32_t width, 
                                     Matrix5d& matrix) {
    if (batch_size <= 0) {
        LOG(ERROR) << "create matrix failed, input batch size <= 0";
        return -1;
    }
    if (filter_number <= 0) {
        LOG(ERROR) << "create matrix failed, input filter number <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix5d(batch_size, Matrix4d(filter_number, Matrix3d(depth, Matrix2d(height, Matrix1d(width, 0)))));

    return 0;
}


//创建2维矩阵 初始值为1
template <typename DataType>
int8_t Matrix<DataType>::CreateOnes(int32_t rows, int32_t cols, 
                                    Matrix2d& matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix2d(rows, Matrix1d(cols, 1));

    return 0;
}

//创建2维矩阵 初始值为1
template <typename DataType>
int8_t Matrix<DataType>::CreateOnes(const std::tuple<int32_t, int32_t>& shape, 
                                    Matrix2d& matrix) {
    int rows = -1;
    int cols = -1;
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "create matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "create matrix failed, input cols <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix2d(rows, Matrix1d(cols, 1));

    return 0;
}

// 创建3维矩阵 初始值为1
template <typename DataType>
int8_t Matrix<DataType>::CreateOnes(int32_t depth, int32_t height, int32_t width, 
                                    Matrix3d& matrix) {
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width, 1)));

    return 0;
}

// 创建3维矩阵 初始值为1
template <typename DataType>
int8_t Matrix<DataType>::CreateOnes(const std::tuple<int32_t, int32_t, int32_t>& shape, 
                                    Matrix3d& matrix) {
    int depth = -1;
    int height = -1;
    int width = -1;
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "create matrix failed, input depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "create matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "create matrix failed, input width <= 0";
        return -1;
    }

    //不用check  直接赋值拷贝
    matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width, 1)));

    return 0;
}

//打印一维矩阵
template <typename DataType>
void Matrix<DataType>::MatrixShow(const Matrix1d& matrix) {
    //check 源矩阵
    if (!MatrixCheck(matrix, true)) {
        LOG(ERROR) << "print matrix failed";
        return ;
    }

    for (int i = 0; i < matrix.size(); i++) {
        if (0 == i) {
            std::cout << "  [";
        }

        //如果是负数 则会多占一格 那么是正数 就多加个空格
        //设置浮点的格式 后面6位
        int space_number = 0;
        if (matrix[i] >= 0) {
            if ((matrix[i] / 10.0) < 1.0) {
                space_number = 3;
            } else {
                space_number = 2;
            }
        } else {
            if ((matrix[i] / 10.0) < 1.0) {
                space_number = 2;
            } else {
                space_number = 1;
            }
        }

        std::cout << std::showpoint << std::setiosflags(std::ios::fixed)
                  << std::setprecision(16) << std::string(space_number, ' ')
                      << matrix[i];

        if (0 == (i + 1) % 8) {
            std::cout << std::endl;
        }
            
        if ((i + 1) == matrix.size()) {
            std::cout << "  ]" << std::endl;
        }
    }

    std::cout << "源矩阵size: " << matrix.size(); 
    std::cout << std::endl;
}

//打印二维矩阵
template <typename DataType>
void Matrix<DataType>::MatrixShow(const Matrix2d& matrix) {
    //check 源矩阵
    if (!MatrixCheck(matrix, true)) {
        LOG(ERROR) << "print matrix failed";
        return ;
    }

    bool flag = matrix[0].size() > 8 ? true : false;

    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            if (!flag) {
                if (0 == j) {
                    std::cout << "  [";
                }
            } else {
                if (0 == (i * matrix[0].size() + j) % 8) {
                    std::cout << "  [";
                }
            }

            //如果是负数 则会多占一格 那么是正数 就多加个空格
            //设置浮点的格式 后面6位
            int space_number = 0;
            if (matrix[i][j] >= 0) {
                if ((matrix[i][j] / 10.0) < 1.0) {
                    space_number = 3;
                } else {
                    space_number = 2;
                }
            } else {
                if ((matrix[i][j] / 10.0) < 1.0) {
                    space_number = 2;
                } else {
                    space_number = 1;
                }
            }

            std::cout << std::showpoint << std::setiosflags(std::ios::fixed)
                      << std::setprecision(16) << std::string(space_number, ' ')
                      << matrix[i][j];
            
            if (flag) {
                if (0 == (i * matrix[0].size() + j + 1) % 8) {
                    std::cout << "  ]" << std::endl;
                }
            } else {
                if ((j + 1) == matrix[i].size()) {
                    std::cout << "  ]" << std::endl;
                }
            }
        }
    }

    if (flag) {
        std::cout << std::endl;
    }
    std::cout << "源矩阵shape(行 列): " << matrix.size() 
              << "*" << matrix[0].size() << std::endl;
    std::cout << std::endl;
}

//打印三维矩阵  顺便先打印一句矩阵的shape
template <typename DataType>
void Matrix<DataType>::MatrixShow(const Matrix3d& matrix) {
    //getshape 会check
    auto shape = GetShape(matrix);
    if (shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "print matrix failed";
        return ;
    }
    //元祖解包
    int channel_number;
    int height;
    int width;
    std::tie(channel_number, height, width) = shape;

    bool flag = matrix[0][0].size() > 8 ? true : false;

    for (int i = 0; i < channel_number; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                if (!flag) {
                    if (0 == i
                            && 0 == j
                            && 0 == k) {
                        std::cout << "[[[";
                    } else if (0 == j
                            && 0 == k) {
                        std::cout << " [[";
                    } else if (0 == k) {
                        std::cout << "  [";
                    }
                } else {
                    if (0 == (i * height * width + j * width + k) % 8) {
                        std::cout << "  [";
                    }
                }

                //如果是负数 则会多占一格 那么是正数 就多加个空格
                //设置浮点的格式 后面6位
                int space_number = 0;
                if (matrix[i][j][k] > 0) {
                    if ((matrix[i][j][k] / 10.0) < 1.0) {
                        space_number += 3;
                    } else {
                        space_number += 2;
                    }
                } else if (matrix[i][j][k] < 0) {
                    if ((matrix[i][j][k] / 10.0) < 1.0) {
                        space_number += 2;
                    } else {
                        space_number += 1;
                    }
                } else {
                    space_number += 3;
                }

                //正0 和 负0 打印时都打印成0
                if (0 == matrix[i][j][k]) {
                    std::cout << std::showpoint << std::setiosflags(std::ios::fixed)
                              << std::setprecision(16) << std::string(space_number, ' ')
                              << static_cast<double>(0.0000000000000000);
                } else {
                    std::cout << std::showpoint << std::setiosflags(std::ios::fixed)
                              << std::setprecision(16) << std::string(space_number, ' ')
                              << matrix[i][j][k];
                }
           
                if (flag) {
                    if (0 == (i * height * width + j * width + k + 1) % 8) {
                        std::cout << "  ]" << std::endl;
                    }
                } else {
                    if ((i + 1) == channel_number 
                            && (j + 1) == height
                            && (k + 1) == width) {
                        std::cout << "  ]]]" << std::endl;
                    } else if ((j + 1) == height 
                            && (k + 1) == width) {
                        std::cout << "  ]]\n" << std::endl;
                    } else if ((k + 1) == width) {
                        std::cout << "  ]" << std::endl;
                    }
                }
            }
        }
    }

    if (flag) {
        std::cout << std::endl;
    }
    std::cout << "源矩阵shape(深度 高 宽): " << channel_number << "*"
              << height << "*" << width << std::endl;
    std::cout << std::endl << std::endl;
}

//打印四维矩阵  顺便先打印一句矩阵的shape
template <typename DataType>
void Matrix<DataType>::MatrixShow(const Matrix4d& matrix) {
    //getshape 会check
    auto shape = GetShape(matrix);
    if (shape == std::make_tuple(0, 0, 0, 0)) {
        LOG(ERROR) << "print matrix failed";
        return ;
    }
    //元祖解包
    int batch_size;
    int channel_number;
    int height;
    int width;
    std::tie(batch_size, channel_number, height, width) = shape;
    bool flag = matrix[0][0][0].size() > 8 ? true : false;

    for (int n = 0; n < batch_size; n++) {
        for (int i = 0; i < channel_number; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    if (!flag) {
                        if (0 == i
                                && 0 == j
                                && 0 == k) {
                            std::cout << "[[[";
                        } else if (0 == j
                                && 0 == k) {
                            std::cout << " [[";
                        } else if (0 == k) {
                            std::cout << "  [";
                        }
                    } else {
                        if (0 == (n * channel_number * height * width + i * height * width + j * width + k) % 8) {
                            std::cout << "  [";
                        }
                    }

                    //如果是负数 则会多占一格 那么是正数 就多加个空格
                    //设置浮点的格式 后面6位
                    int space_number = 0;
                    if (matrix[n][i][j][k] > 0) {
                        if ((matrix[n][i][j][k] / 10.0) < 1.0) {
                            space_number += 3;
                        } else {
                            space_number += 2;
                        }
                    } else if (matrix[n][i][j][k] < 0) {
                        if ((matrix[n][i][j][k] / 10.0) < 1.0) {
                            space_number += 2;
                        } else {
                            space_number += 1;
                        }
                    } else {
                        space_number += 3;
                    }

                    //正0 和 负0 打印时都打印成0
                    if (0 == matrix[n][i][j][k]) {
                        std::cout << std::showpoint << std::setiosflags(std::ios::fixed)
                                  << std::setprecision(16) << std::string(space_number, ' ')
                                  << static_cast<double>(0.0000000000000000);
                    } else {
                        std::cout << std::showpoint << std::setiosflags(std::ios::fixed)
                                  << std::setprecision(16) << std::string(space_number, ' ')
                                  << matrix[n][i][j][k];
                    }

                    if (flag) {
                        if (0 == (n * channel_number * height * width + i * height * width + j * width + k + 1) % 8) {
                            std::cout << std::endl;
                        }
                    } else {
                        if ((i + 1) == channel_number 
                                && (j + 1) == height
                                && (k + 1) == width) {
                            std::cout << "  ]]]" << std::endl;
                        } else if ((j + 1) == height 
                                && (k + 1) == width) {
                            std::cout << "  ]]\n" << std::endl;
                        } else if ((k + 1) == width) {
                            std::cout << "  ]" << std::endl;
                        }
                    }
                }
            }
        }
        std::cout << std::endl;
    }

    if (flag) {
        std::cout << std::endl;
    }
    std::cout << "源矩阵shape(批大小 深度 高 宽): " << batch_size << "*"
              << channel_number << "*" << height << "*" << width << std::endl;
    std::cout << std::endl << std::endl;
}



// 打印图像二维矩阵
template <typename DataType>
void Matrix<DataType>::ImageMatrixShow(const std::vector<std::vector<uint8_t>>& matrix) {
    if (!MatrixCheck(matrix, true)) {
        LOG(ERROR) << "print matrix failed, input matrix is wrong";
        return ;
    }

    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            if (0 == j) {
                std::cout << "  [";
            }

            std::cout << std::setw(3) << std::setiosflags(std::ios::right)
                      << static_cast<int>(matrix[i][j]) << " ";
            
            if ((j + 1) == matrix[i].size()) {
                std::cout << " ]" << std::endl;
            }
        }
    }

    std::cout << "源矩阵shape(行 列): " << matrix.size() 
              << "*" << matrix[0].size() << std::endl;
    std::cout << std::endl;
}

// 打印图像三维矩阵
template <typename DataType>
void Matrix<DataType>::ImageMatrixShow(const std::vector<std::vector<std::vector<uint8_t>>>& matrix) {
    //getshape 会check
    auto shape = GetShape(matrix);
    if (shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "print matrix failed";
        return ;
    }
    //元祖解包
    int channel_number;
    int height;
    int width;
    std::tie(channel_number, height, width) = shape;

    for (int i = 0; i < channel_number; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                if (0 == i
                        && 0 == j
                        && 0 == k) {
                    std::cout << "[[[";
                } else if (0 == j
                            && 0 == k) {
                    std::cout << " [[";
                } else if (0 == k) {
                    std::cout << "  [";
                }

                std::cout << std::setw(3) << std::setiosflags(std::ios::right)
                          << static_cast<int>(matrix[i][j][k]) << " ";

                if ((i + 1) == channel_number 
                        && (j + 1) == height
                        && (k + 1) == width) {
                    std::cout << "  ]]]" << std::endl;
                } else if ((j + 1) == height 
                            && (k + 1) == width) {
                    std::cout << "  ]]\n" << std::endl;
                } else if ((k + 1) == width) {
                    std::cout << "  ]" << std::endl;
                }
            }
        }
    }

    std::cout << "源矩阵shape(深度 高 宽): " << channel_number << "*"
              << height << "*" << width << std::endl;
    std::cout << std::endl << std::endl;
}


//矩阵相乘(dot product)点积
template <class DataType>
int8_t Matrix<DataType>::DotProduct(const Matrix2d& left_matrix, 
                                    const Matrix2d& right_matrix, 
                                    Matrix2d& result_matrix) {
    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(left_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    auto right_matrix_shape = GetShape(right_matrix);
    if (right_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    //元组解包
    std::tie(left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix dot product failed, left matrix cols is not equal of right matrix rows";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    result_matrix.clear();
    result_matrix = Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始点积运算  
        for (int i = 0; i < left_matrix_rows; i++) {
            for (int j = 0; j < right_matrix_cols; j++) {
                for (int k = 0; k < left_matrix_cols; k++) {
                    result_matrix[i][j] += left_matrix[i][k] * right_matrix[k][j];
                }
            }
        }
    }

    return 0;
}

//矩阵相乘(dot product)点积
template <class DataType>
int8_t Matrix<DataType>::DotProduct(const Matrix3d& left_matrix, 
                                    const Matrix3d& right_matrix, 
                                    Matrix2d& result_matrix) {
    int batch_size;
    int right_matrix_batch_size;
    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(left_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    auto right_matrix_shape = GetShape(right_matrix);
    if (right_matrix_shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    //元组解包
    std::tie(batch_size, left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(right_matrix_batch_size, right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows
            || batch_size != right_matrix_batch_size) {
        LOG(ERROR) << "matrix dot product failed, left matrix cols is not equal of right matrix rows";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    result_matrix.clear();
    result_matrix = Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始点积运算  
        for (int n = 0; n < batch_size; n++) {
            for (int i = 0; i < left_matrix_rows; i++) {
                for (int j = 0; j < right_matrix_cols; j++) {
                    for (int k = 0; k < left_matrix_cols; k++) {
                        result_matrix[i][j] += left_matrix[n][i][k] * right_matrix[n][k][j];
                    }
                }
            }
        }
    }

    return 0;
}

//矩阵相乘(dot product)点积
template <class DataType>
int8_t Matrix<DataType>::DotProduct(const Matrix3d& left_matrix, 
                                    const Matrix3d& right_matrix, 
                                    Matrix3d& result_matrix) {
    int batch_size;
    int right_matrix_batch_size;
    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(left_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    auto right_matrix_shape = GetShape(right_matrix);
    if (right_matrix_shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    //元组解包
    std::tie(batch_size, left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(right_matrix_batch_size, right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows
            || batch_size != right_matrix_batch_size) {
        LOG(ERROR) << "matrix dot product failed, left matrix cols is not equal of right matrix rows";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    result_matrix.clear();
    result_matrix = Matrix3d(batch_size, Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols)));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始点积运算  
        for (int n = 0; n < batch_size; n++) {
            for (int i = 0; i < left_matrix_rows; i++) {
                for (int j = 0; j < right_matrix_cols; j++) {
                    for (int k = 0; k < left_matrix_cols; k++) {
                        result_matrix[n][i][j] += left_matrix[n][i][k] * right_matrix[n][k][j];
                    }
                }
            }
        }
    }

    return 0;
}

//矩阵相乘函数重载
template <class DataType>
int8_t Matrix<DataType>::DotProduct(const std::vector<std::vector<double>>& left_matrix, 
                                    const std::vector<std::vector<uint8_t>>& right_matrix, 
                                    std::vector<std::vector<double>>& result_matrix) {
    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(left_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    auto right_matrix_shape = GetShape(ToDouble(right_matrix));
    if (right_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix dot product failed";
        return -1;
    }

    //元组解包
    std::tie(left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix dot product failed, left matrix cols is not equal of right matrix rows";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    result_matrix.clear();
    result_matrix = std::vector<std::vector<double>>(left_matrix_rows, std::vector<double>(right_matrix_cols));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始点积运算  
        for (int i = 0; i < left_matrix_rows; i++) {
            for (int j = 0; j < right_matrix_cols; j++) {
                for (int k = 0; k < left_matrix_cols; k++) {
                    result_matrix[i][j] += left_matrix[i][k] * right_matrix[k][j];
                }
            }
        }
    }

    return 0;
}

//hadamark积 2d矩阵相应位置相乘
template <typename DataType>
int8_t Matrix<DataType>::HadamarkProduct(const Matrix2d& left_matrix, 
                                         const Matrix2d& right_matrix, 
                                         Matrix2d& result_matrix) {
    //检查 两个输入矩阵的正确性 内部会先check单独矩阵的行 列 
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix hadamark product failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                result_matrix[i][j] = left_matrix[i][j] * right_matrix[i][j];
            }
        }
    }

    return 0;
}

//hadamark积 3d矩阵相应位置相乘
template <typename DataType>
int8_t Matrix<DataType>::HadamarkProduct(const Matrix3d& left_matrix, 
                                         const Matrix3d& right_matrix, 
                                         Matrix3d& result_matrix) {
    //检查 两个输入矩阵的正确性 内部会先check单独矩阵的行 列 
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix hadamark product failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                for (int k = 0; k < left_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = left_matrix[i][j][k] * right_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
}

//hadamark积 3d矩阵相应位置相乘
template <typename DataType>
int8_t Matrix<DataType>::HadamarkProduct(const Matrix3d& left_matrix, 
                                         const Matrix2d& right_matrix, 
                                         Matrix3d& result_matrix) {
    //检查 两个输入矩阵的正确性 内部会先check单独矩阵的行 列 
    if (!MatrixCheck(left_matrix, true)) {
        LOG(ERROR) << "matrix hadamark product failed";
        return -1;
    }
    //检查 两个输入矩阵的正确性 内部会先check单独矩阵的行 列 
    if (!MatrixCheck(right_matrix, true)) {
        LOG(ERROR) << "matrix hadamark product failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                for (int k = 0; k < left_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = left_matrix[i][j][k] * right_matrix[j][k];
                }
            }
        }
    }

    return 0;
}

//hadamark积 4d矩阵相应位置相乘
template <typename DataType>
int8_t Matrix<DataType>::HadamarkProduct(const Matrix4d& left_matrix, 
                                         const Matrix3d& right_matrix, 
                                         Matrix4d& result_matrix) {
    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int n = 0; n < left_matrix.size(); n++) {
            for (int i = 0; i < left_matrix[n].size(); i++) {
                for (int j = 0; j < left_matrix[n][i].size(); j++) {
                    for (int k = 0; k < left_matrix[n][i][j].size(); k++) {
                        result_matrix[n][i][j][k] = left_matrix[n][i][j][k] * right_matrix[i][j][k];
                    }
                }
            }
        }
    }

    return 0;
}

//hadamark积 4d矩阵相应位置相乘
template <typename DataType>
int8_t Matrix<DataType>::HadamarkProduct(const Matrix4d& left_matrix, 
                                         const Matrix4d& right_matrix, 
                                         Matrix4d& result_matrix) {
    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int n = 0; n < left_matrix.size(); n++) {
            for (int i = 0; i < left_matrix[n].size(); i++) {
                for (int j = 0; j < left_matrix[n][i].size(); j++) {
                    for (int k = 0; k < left_matrix[n][i][j].size(); k++) {
                        result_matrix[n][i][j][k] = left_matrix[n][i][j][k] * right_matrix[n][i][j][k];
                    }
                }
            }
        }
    }

    return 0;
}


// 2d矩阵对应位置相乘 hadamark积 函数重载 输入uint8_t 输出double
template <typename DataType>
int8_t Matrix<DataType>::HadamarkProduct(const std::vector<std::vector<uint8_t>>& left_matrix, 
                                         const std::vector<std::vector<double>>& right_matrix, 
                                         std::vector<std::vector<double>>& result_matrix) {
    //检查 两个输入矩阵的正确性 内部会先check单独矩阵的行 列 
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix hadamark product failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = right_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                result_matrix[i][j] = left_matrix[i][j] * right_matrix[i][j];
            }
        }
    }

    return 0;

}

// 3d矩阵对应位置相乘 hadamark积 函数重载 输入uint8_t 输出double
template <typename DataType>
int8_t Matrix<DataType>::HadamarkProduct(const std::vector<std::vector<std::vector<uint8_t>>>& left_matrix, 
                                         const std::vector<std::vector<std::vector<double>>>& right_matrix, 
                                         std::vector<std::vector<std::vector<double>>>& result_matrix) {
    //检查 两个输入矩阵的正确性 内部会先check单独矩阵的行 列 
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix hadamark product failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = right_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                for (int k = 0; k < left_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = left_matrix[i][j][k] * right_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
}


//hadamark积 4d矩阵相应位置相乘
template <typename DataType>
int8_t Matrix<DataType>::HadamarkProduct(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& left_matrix, 
                                         const std::vector<std::vector<std::vector<double>>>& right_matrix, 
                                         std::vector<std::vector<std::vector<std::vector<double>>>>& result_matrix) {
    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        CreateZeros(GetShape(left_matrix), result_matrix);
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int n = 0; n < left_matrix.size(); n++) {
            for (int i = 0; i < left_matrix[n].size(); i++) {
                for (int j = 0; j < left_matrix[n][i].size(); j++) {
                    for (int k = 0; k < left_matrix[n][i][j].size(); k++) {
                        result_matrix[n][i][j][k] = left_matrix[n][i][j][k] * right_matrix[i][j][k];
                    }
                }
            }
        }
    }

    return 0;
}


//2d矩阵相加
template <typename DataType>
int8_t Matrix<DataType>::Add(const Matrix2d& left_matrix, 
                             const Matrix2d& right_matrix, 
                             Matrix2d& result_matrix) {
    //检查 两个输入矩阵的正确性
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix add failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //矩阵相加
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                result_matrix[i][j] = left_matrix[i][j] + right_matrix[i][j];
            }
        }
    }

    return 0;
}

//3d矩阵相加
template <typename DataType>
int8_t Matrix<DataType>::Add(const Matrix3d& left_matrix, 
                             const Matrix3d& right_matrix, 
                             Matrix3d& result_matrix) {
    //检查 两个输入矩阵的正确性
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix add failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }
        
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //矩阵相加
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                for (int k = 0; k < left_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = left_matrix[i][j][k] + right_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
}

//2d矩阵相减
template <typename DataType>
int8_t Matrix<DataType>::Subtract(const Matrix2d& left_matrix, 
                                  const Matrix2d& right_matrix, 
                                  Matrix2d& result_matrix) {
    //检查 两个输入矩阵的正确性
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix subtract failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }
        
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //矩阵相减
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                result_matrix[i][j] = left_matrix[i][j] - right_matrix[i][j];
            }
        }
    }

    return 0;
}

//3d矩阵相减
template <typename DataType>
int8_t Matrix<DataType>::Subtract(const Matrix3d& left_matrix, 
                                  const Matrix3d& right_matrix, 
                                  Matrix3d& result_matrix) {
    //检查 两个输入矩阵的正确性
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "matrix subtract failed";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!MatrixCheck(left_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = left_matrix;
    }
        
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //矩阵相减
        for (int i = 0; i < left_matrix.size(); i++) {
            for (int j = 0; j < left_matrix[i].size(); j++) {
                for (int k = 0; k < left_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = left_matrix[i][j][k] - right_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
}

//2d reshape 1d
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix2d& source_matrix,
                                 Matrix1d& result_matrix) {
    //check source matrix
    int rows;
    int cols;
    auto shape = GetShape(source_matrix);
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }

    //初始化结果矩阵
    result_matrix = Matrix1d(rows * cols);
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_matrix[i * cols + j] = source_matrix[i][j];    
            }
        }
    }
    
    return 0;
}

//2d reshape 1d
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix2d& source_matrix,
                                 int32_t& rows, int32_t& cols, 
                                 Matrix1d& result_matrix) {
    //check source matrix
    auto shape = GetShape(source_matrix);
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }

    //初始化结果矩阵
    result_matrix = Matrix1d(rows * cols);
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_matrix[i * cols + j] = source_matrix[i][j];    
            }
        }
    }
    
    return 0;
}

// 2个2d矩阵reshape 成2个1d矩阵  
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix2d& left_matrix,
                                 const Matrix2d& right_matrix, 
                                 int32_t& rows, int32_t& cols, 
                                 Matrix1d& result_left_matrix, 
                                 Matrix1d& result_right_matrix) { 
    //check left matrix
    auto shape = GetShape(left_matrix);
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }

    //check left matrix and right matrix 
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "reshape matrix failed, two input matrices is not equal";
        return -1;
    }

    //初始化结果矩阵
    result_left_matrix = Matrix1d(rows * cols);
    result_right_matrix = Matrix1d(rows * cols);
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_left_matrix[i * cols + j] = left_matrix[i][j];    
                result_right_matrix[i * cols + j] = right_matrix[i][j];    
            }
        }
    }

    return 0;
}

// 2个2d矩阵reshape 成2个1d矩阵  
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix2d& left_matrix,
                                 const Matrix2d& right_matrix, 
                                 int32_t& left_rows, int32_t& left_cols,
                                 int32_t& right_cols, 
                                 Matrix1d& result_left_matrix, 
                                 Matrix1d& result_right_matrix) { 
    //check left matrix
    auto shape = GetShape(left_matrix);
    std::tie(left_rows, left_cols) = shape;
    if (left_rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input left rows <= 0";
        return -1;
    }
    if (left_cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input left cols <= 0";
        return -1;
    }

    //check right matrix 
    shape = GetShape(right_matrix);
    int right_rows;
    std::tie(right_rows, right_cols) = shape;
    if (right_rows <= 0
            || right_rows != left_cols) {
        LOG(ERROR) << "reshape matrix failed, input right rows <= 0 or right rows is not equal left cols";
        return -1;
    }
    if (right_cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input right cols <= 0";
        return -1;
    }

    //初始化结果矩阵
    result_left_matrix = Matrix1d(left_rows * left_cols);
    result_right_matrix = Matrix1d(right_rows * right_cols);
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < left_rows; i++) {
            for (int j = 0; j < left_cols; j++) {
                result_left_matrix[i * left_cols + j] = left_matrix[i][j];    
            }
        }

        #pragma omp for schedule(static) 
        for (int i = 0; i < right_rows; i++) {
            for (int j = 0; j < right_cols; j++) {
                result_right_matrix[i * right_cols + j] = right_matrix[i][j];    
            }
        }
    }

    return 0;
}

//3d reshape 1d
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix3d& source_matrix,
                                 Matrix1d& result_matrix) {
    //check source matrix
    int depth;
    int height;
    int width;
    auto shape = GetShape(source_matrix);
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "reshape matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "reshape matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "reshape matrix failed, input width <= 0";
        return -1;
    }

    //初始化结果矩阵
    result_matrix = Matrix1d(depth * height * width);
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result_matrix[i * (height * width) + (j * width) + k] = source_matrix[i][j][k];    
                }
            }
        }
    }
    
    return 0;
}

//3d reshape 1d
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix3d& source_matrix,
                                 int32_t& depth, int32_t& height,
                                 int32_t& width, 
                                 Matrix1d& result_matrix) {
    //check source matrix
    auto shape = GetShape(source_matrix);
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "reshape matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "reshape matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "reshape matrix failed, input width <= 0";
        return -1;
    }

    //初始化结果矩阵
    result_matrix = Matrix1d(depth * height * width);
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result_matrix[i * (height * width) + (j * width) + k] = source_matrix[i][j][k];    
                }
            }
        }
    }
    
    return 0;
}

// 2个3d矩阵reshape 成2个1d矩阵  
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix3d& left_matrix,
                                 const Matrix3d& right_matrix, 
                                 int32_t& depth, int32_t& height,
                                 int32_t& width, 
                                 Matrix1d& result_left_matrix, 
                                 Matrix1d& result_right_matrix) {
    //check left matrix
    auto shape = GetShape(left_matrix);
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "reshape matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "reshape matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "reshape matrix failed, input width <= 0";
        return -1;
    }

    //check left matrix and right matrix 
    if (!MatrixCheck(left_matrix, right_matrix, true)) {
        LOG(ERROR) << "reshape matrix failed, two input matrices is not equal";
        return -1;
    }

    //初始化结果矩阵
    result_left_matrix = Matrix1d(depth * height * width);
    result_right_matrix = Matrix1d(depth * height * width);
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result_left_matrix[i * (height * width) + (j * width) + k] = 
                                                                    left_matrix[i][j][k];    
                    result_right_matrix[i * (height * width) + (j * width) + k] = 
                                                                    right_matrix[i][j][k];    
                }
            }
        }
    }

    return 0;
}

//4d reshape 1d
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix4d& source_matrix,
                                 Matrix1d& result_matrix) {
    //check source matrix
    int batch_size;
    int depth;
    int height;
    int width;
    auto shape = GetShape(source_matrix);
    std::tie(batch_size, depth, height, width) = shape;
    if (batch_size <= 0) {
        LOG(ERROR) << "reshape matrix failed, input batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "reshape matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "reshape matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "reshape matrix failed, input width <= 0";
        return -1;
    }

    //初始化结果矩阵
    result_matrix = Matrix1d(batch_size * depth * height * width);
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int n = 0; n < batch_size; n++) {
            for (int i = 0; i < depth; i++) {
                for (int j = 0; j < height; j++) {
                    for (int k = 0; k < width; k++) {
                        result_matrix[n * (depth * height * width) + (i * height * width) + (j * width) + k] = source_matrix[n][i][j][k];    
                    }
                }
            }
        }
    }
    
    return 0;
}

//1d矩阵 reshape成2d矩阵
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix1d& source_matrix,
                                 int32_t rows, int32_t cols, 
                                 Matrix2d& result_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }

    //check 源矩阵
    if (0 == source_matrix.size()) {
        LOG(ERROR) << "reshape matrix failed, input matrix is empty";
        return -1;
    }

    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = source_matrix.size();
    
    if (matrix_total_size != (rows * cols)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }
    
    if (!MatrixCheck(result_matrix, rows, cols, false)) {
        result_matrix = Matrix2d(rows, Matrix1d(cols));
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //再赋值给新数组
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_matrix[i][j] = source_matrix[i * cols + j];    
            }
        }
    }

    return 0;
}

//1d矩阵 reshape成3d矩阵
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix1d& source_matrix,
                                 int32_t depth, int32_t height,
                                 int32_t width, 
                                 Matrix3d& result_matrix) {
    if (depth <= 0) {
        LOG(ERROR) << "reshape matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "reshape matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "reshape matrix failed, input width <= 0";
        return -1;
    }

    //check 源矩阵
    if (0 == source_matrix.size()) {
        LOG(ERROR) << "reshape matrix failed, input matrix is empty";
        return -1;
    }

    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = source_matrix.size();
    
    if (matrix_total_size != (depth * height * width)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }

    if (!MatrixCheck(result_matrix, depth, height, width, false)) {
        result_matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width)));
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result_matrix[i][j][k] = source_matrix[i * (height * width) + (j * width) + k]; 
                }
            }
        }
    }
    
    return 0;
}

// 2d矩阵reshape 成2d矩阵
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix2d& source_matrix,
                                 int32_t rows, int32_t cols, 
                                 Matrix2d& result_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }
    
    Matrix1d matrix_data;
    Reshape(source_matrix, matrix_data);

    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = matrix_data.size();
    
    if (matrix_total_size != (rows * cols)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }

    //check一下输出数组
    if (!MatrixCheck(result_matrix, rows, cols, false)) {
        result_matrix = Matrix2d(rows, Matrix1d(cols));
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_matrix[i][j] = matrix_data[i * cols + j];    
            }
        }
    }

    return 0;
}

// 3d矩阵reshape 成2d矩阵
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix3d& source_matrix,
                                 int32_t rows, int32_t cols, 
                                 Matrix2d& result_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }
    
    Matrix1d matrix_data;
    Reshape(source_matrix, matrix_data);

    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = matrix_data.size();
    
    if (matrix_total_size != (rows * cols)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }

    //check一下输出数组
    if (!MatrixCheck(result_matrix, rows, cols, false)) {
        result_matrix = Matrix2d(rows, Matrix1d(cols));
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result_matrix[i][j] = matrix_data[i * cols + j];    
            }
        }
    }

    return 0;
}

// 2d矩阵reshape 成3d矩阵
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix2d& source_matrix,
                                 int32_t depth, int32_t height,
                                 int32_t width, Matrix3d& result_matrix) {
    if (depth <= 0) {
        LOG(ERROR) << "reshape matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "reshape matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "reshape matrix failed, input width <= 0";
        return -1;
    }

    Matrix1d matrix_data;
    Reshape(source_matrix, matrix_data);
    
    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = matrix_data.size();
    
    if (matrix_total_size != (depth * height * width)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }

    //check一下输出数组
    if (!MatrixCheck(result_matrix, depth, height, width, false)) {
        result_matrix = Matrix3d(depth, Matrix2d(height, Matrix1d(width)));
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result_matrix[i][j][k] = matrix_data[i * (height * width) + (j * width) + k]; 
                }
            }
        }
    }

    return 0;
}

// 4d矩阵reshape 成3矩阵
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix4d& source_matrix,
                                 int32_t depth, int32_t rows, int32_t cols, 
                                 Matrix3d& result_matrix) {
    if (depth <= 0) {
        LOG(ERROR) << "reshape matrix failed, input depth <= 0";
        return -1;
    }
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }

    Matrix1d matrix_data;
    Reshape(source_matrix, matrix_data);

    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = matrix_data.size();
    
    if (matrix_total_size != (depth * rows * cols)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }

    //check一下输出数组
    if (!MatrixCheck(result_matrix, depth, rows, cols, false)) {
        result_matrix = Matrix3d(depth, Matrix2d(rows, Matrix1d(cols)));
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result_matrix[d][i][j] = matrix_data[d * rows * cols + i * cols + j];    
                }
            }
        }
    }

    return 0;
}

// 3d矩阵reshape 成4矩阵
template <typename DataType>
int8_t Matrix<DataType>::Reshape(const Matrix3d& source_matrix,
                                 int32_t batch_size, int32_t depth,
                                 int32_t rows, int32_t cols, 
                                 Matrix4d& result_matrix) {
    if (batch_size <= 0) {
        LOG(ERROR) << "reshape matrix failed, input batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "reshape matrix failed, input depth <= 0";
        return -1;
    }
    if (rows <= 0) {
        LOG(ERROR) << "reshape matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "reshape matrix failed, input cols <= 0";
        return -1;
    }

    Matrix1d matrix_data;
    Reshape(source_matrix, matrix_data);

    //检查输入矩阵的总数量 是否等于 要reshape的总量
    int matrix_total_size = matrix_data.size();
    
    if (matrix_total_size != (batch_size * depth * rows * cols)) {
        LOG(ERROR) << "matrix reshape failed, input matrix couldn't reshape become that shape";
        return -1;
    }

    //check一下输出数组
    if (!MatrixCheck(result_matrix, batch_size, depth, rows, cols, false)) {
        result_matrix = Matrix4d(batch_size, Matrix3d(depth, Matrix2d(rows, Matrix1d(cols))));
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int n = 0; n < batch_size; n++) {
            for (int d = 0; d < depth; d++) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        result_matrix[n][d][i][j] = matrix_data[n * depth * rows * cols + d * rows * cols + i * cols + j];    
                    }
                }
            }
        }
    }

    return 0;
}




//2d转置矩阵
template <typename DataType>
int8_t Matrix<DataType>::Transpose(const Matrix2d& source_matrix, 
                                   Matrix2d& result_matrix) {
    int source_matrix_rows = 0;
    int source_matrix_cols = 0;
    //reshape 会check
    auto shape = GetShape(source_matrix);
    if (shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "transpose matrix failed";
        return -1;
    }
    //元组解包
    std::tie(source_matrix_rows, source_matrix_cols) = shape;

    //如果数组数据没有初始化 就用移动赋值函数初始化 
    //行为原矩阵的列 列为原矩阵的行 比如2 * 4  变成4 * 2
    if (!MatrixCheck(result_matrix, source_matrix_cols, source_matrix_rows, false)) {
        result_matrix = Matrix2d(source_matrix_cols, Matrix1d(source_matrix_rows));
    } 

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //转置矩阵
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                result_matrix[j][i] = source_matrix[i][j];
            }
        }
    }

    return 0;
}

//3d转置矩阵
template <typename DataType>
int8_t Matrix<DataType>::Transpose(const Matrix3d& source_matrix, 
                                   Matrix3d& result_matrix) {
    int source_matrix_depth = 0;
    int source_matrix_rows = 0;
    int source_matrix_cols = 0;
    //reshape 会check
    auto shape = GetShape(source_matrix);
    if (shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "transpose matrix failed";
        return -1;
    }
    //元组解包
    std::tie(source_matrix_depth, source_matrix_rows, source_matrix_cols) = shape;

    //如果数组数据没有初始化 就用移动赋值函数初始化 
    //行为原矩阵的列 列为原矩阵的行 比如2 * 4  变成4 * 2
    if (!MatrixCheck(result_matrix, source_matrix_depth, source_matrix_cols, source_matrix_rows, false)) {
        result_matrix = Matrix3d(source_matrix_depth, Matrix2d(source_matrix_cols, Matrix1d(source_matrix_rows)));
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //转置矩阵
        for (int n = 0; n < source_matrix_depth; n++) {
            for (int i = 0; i < source_matrix_rows; i++) {
                for (int j = 0; j < source_matrix_cols; j++) {
                    result_matrix[n][j][i] = source_matrix[n][i][j];
                }
            }
        }
    }

    return 0;
}

//2d矩阵都乘以一个值
template <typename DataType>
int8_t Matrix<DataType>::ValueMulMatrix(DataType value,  
                                        const Matrix2d& source_matrix, 
                                        Matrix2d& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "value mul matrix failed";
        return -1;
    }

    //check结果矩阵
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                result_matrix[i][j] = source_matrix[i][j] * value;
            }
        }
    }

    return 0;
}

//3d矩阵都乘以一个值
template <typename DataType>
int8_t Matrix<DataType>::ValueMulMatrix(DataType value,  
                                        const Matrix3d& source_matrix, 
                                        Matrix3d& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "value mul matrix failed";
        return -1;
    }

    //check结果矩阵
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    } 

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                for (int k = 0; k < source_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = source_matrix[i][j][k] * value;
                }
            }
        }
    }

    return 0;
}

//一个值减去矩阵每个值
template <typename DataType>
int8_t Matrix<DataType>::ValueSubMatrix(DataType value,  
                                        const Matrix2d& source_matrix, 
                                        Matrix2d& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "value sub matrix failed";
        return -1;
    }

    //check结果矩阵
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    } 

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                result_matrix[i][j] = value - source_matrix[i][j];
            }
        }
    }

    return 0;
}

//一个值减去矩阵每个值
template <typename DataType>
int8_t Matrix<DataType>::ValueSubMatrix(DataType value,  
                                        const Matrix3d& source_matrix, 
                                        Matrix3d& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "value sub matrix failed";
        return -1;
    }

    //check结果矩阵
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    } 

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                for (int k = 0; k < source_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = value - source_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
}

//2d矩阵都除以一个值
template <typename DataType>
int8_t Matrix<DataType>::MatrixDivValue(const Matrix2d& source_matrix, 
                                        DataType value, 
                                        Matrix2d& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "value mul matrix failed";
        return -1;
    }

    //check结果矩阵
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                result_matrix[i][j] = source_matrix[i][j] / value;
            }
        }
    }

    return 0;
}

//3d矩阵都除以一个值
template <typename DataType>
int8_t Matrix<DataType>::MatrixDivValue(const Matrix3d& source_matrix, 
                                        DataType value, 
                                        Matrix3d& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "value mul matrix failed";
        return -1;
    }

    //check结果矩阵
    if (!MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                for (int k = 0; k < source_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = source_matrix[i][j][k] / value;
                }
            }
        }
    }

    return 0;
}


//计算2d矩阵的和
template <typename DataType>
double Matrix<DataType>::Sum(const std::vector<std::vector<uint8_t>>& source_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix sum failed";
        return -1;
    }

    double sum = 0.0;

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : sum) 
        //多线程数据竞争  加锁保护
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                sum += source_matrix[i][j];
            }
        }
    }

    return sum;
}

//计算2d矩阵的和
template <typename DataType>
double Matrix<DataType>::Sum(const std::vector<std::vector<double>>& source_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix sum failed";
        return -1;
    }

    double sum = 0.0;

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : sum) 
        //多线程数据竞争  加锁保护
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                sum += source_matrix[i][j];
            }
        }
    }

    return sum;
}

//计算3d矩阵的和
template <typename DataType>
double Matrix<DataType>::Sum(const std::vector<std::vector<std::vector<uint8_t>>>& source_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix sum failed";
        return -1;
    }

    double sum = 0.0;

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : sum) 
        //多线程数据竞争  加锁保护
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                for (int k = 0; k < source_matrix[i][j].size(); k++) {
                    sum += source_matrix[i][j][k];
                }
            }
        }
    }

    return sum;
}

//计算3d矩阵的和
template <typename DataType>
double Matrix<DataType>::Sum(const std::vector<std::vector<std::vector<double>>>& source_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix sum failed";
        return -1;
    }

    double sum = 0.0;

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : sum) 
        //多线程数据竞争  加锁保护
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                for (int k = 0; k < source_matrix[i][j].size(); k++) {
                    sum += source_matrix[i][j][k];
                }
            }
        }
    }

    return sum;
}

//得到2d矩阵的最大值
template <typename DataType>
double Matrix<DataType>::Max(const Matrix2d& source_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix max failed";
        return -1;
    }

    double max = 0.0;

    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            if (source_matrix[i][j] > max) {
                max = source_matrix[i][j];
            }
        }
    }

    return max;
}

//得到2d矩阵最大值所在的行 列索引
template <typename DataType>
std::tuple<int32_t, int32_t> Matrix<DataType>::GetMaxIndex(const Matrix2d& source_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix max index failed";
        return std::make_tuple(-1, -1);
    }

    double max = 0.0;
    int rows = 0;
    int cols = 0;

    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            if (source_matrix[i][j] > max) {
                max = source_matrix[i][j];
                rows = i;
                cols = j;
            }
        }
    }

    return std::make_tuple(rows, cols);
}

//得到3d矩阵最大值所在的行 列索引
template <typename DataType>
std::vector<std::tuple<int32_t, int32_t>> Matrix<DataType>::GetMaxIndex(const Matrix3d& source_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix max index failed";
        return std::vector<std::tuple<int32_t, int32_t>>(0);
    }

    std::vector<std::tuple<int32_t, int32_t>> max_array;
    max_array.reserve(source_matrix.size());

    for (int i = 0; i < source_matrix.size(); i++) {
        double max = 0.0;
        int rows = 0;
        int cols = 0;
        for (int j = 0; j < source_matrix[i].size(); j++) {
            for (int k = 0; k < source_matrix[i][j].size(); k++) {
                if (source_matrix[i][j][k] > max) {
                    max = source_matrix[i][j][k];
                    rows = j;
                    cols = k;
                }
            }
        }
        max_array.emplace_back(rows, cols);
    }

    return max_array;
}

// 得到2d矩阵中值为0的所有索引
template <typename DataType>
int8_t Matrix<DataType>::GetZeroIndex(const Matrix2d& source_matrix, 
                                      std::vector<std::tuple<int32_t, int32_t>>& result_matrix) {
    //check源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix zero index failed";
        return -1;
    }
    
    if (0 != result_matrix.size()) {
        result_matrix.clear();
    }
    
    result_matrix.reserve(static_cast<int>(source_matrix.size() * source_matrix[0].size() / 2));
    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            if (0 == source_matrix[i][j]) {
                //emplace 用参数构造元素 直接调用构造函数生成
                result_matrix.emplace_back(i, j);
            }
        }
    }

    return 0;
}

//归一化到0, 1 计算概率分布
template <typename DataType>
int8_t Matrix<DataType>::SoftMax(const Matrix2d& source_matrix, 
                                 Matrix2d& result_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "matrix calculate soft max failed";
        return -1;
    }

    if (!MatrixCheck(result_matrix, source_matrix, false)) {
        result_matrix = source_matrix;
    }
    
    double max = Max(source_matrix);
    Matrix2d exp_array = source_matrix;
    double sum = 0.0;

    for (int i = 0; i < result_matrix.size(); i++) {
        for (int j = 0; j < result_matrix[i].size(); j++) {
            LOG(WARNING) << exp(source_matrix[i][j]);
            //先全部减去最大值 避免求指数得到nan
            double c = source_matrix[i][j] - max;
            exp_array[i][j] = exp(c);
            sum += exp_array[i][j];
        }
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < result_matrix.size(); i++) {
            for (int j = 0; j < result_matrix[i].size(); j++) {
                result_matrix[i][j] = exp_array[i][j] / sum;
            }
        }
    }

    return 0;
}

//归一化到0, 1 计算概率分布
template <typename DataType>
int8_t Matrix<DataType>::SoftMax(const Matrix3d& source_matrix, 
                                 Matrix3d& result_matrix) {
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "matrix calculate soft max failed";
        return -1;
    }

    if (!MatrixCheck(result_matrix, source_matrix, false)) {
        result_matrix = source_matrix;
    }
    
    Matrix3d exp_array = source_matrix;
    Matrix1d sum_array(source_matrix.size());
    for (int i = 0; i < result_matrix.size(); i++) {
        double max = Max(source_matrix[i]);
        double sum = 0.0;

        for (int j = 0; j < result_matrix[i].size(); j++) {
            for (int k = 0; k < result_matrix[i][j].size(); k++) {
                //先全部减去最大值 避免求指数得到nan
                double c = source_matrix[i][j][k] - max;
                exp_array[i][j][k] = exp(c);
                sum += exp_array[i][j][k];
            }
        }
        sum_array[i] = sum;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < result_matrix.size(); i++) {
            for (int j = 0; j < result_matrix[i].size(); j++) {
                for (int k = 0; k < result_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = exp_array[i][j][k] / sum_array[i];
                }
            }
        }
    }

    return 0;
}

//归一化到0, 1 计算概率分布 交叉熵损失函数
template <typename DataType>
double Matrix<DataType>::CrossEntropyLoss(const Matrix3d& output_array, 
                                          const Matrix3d& label, 
                                          Matrix3d& output_delta_array) {
    if (!MatrixCheck(output_array, label, true)) {
        LOG(ERROR) << "matrix calculate cross entropy loss failed";
        return -1.0;
    }

    output_delta_array = output_array;
    int batch_size = output_array.size();
    double loss = 0.0;
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : loss)
        for (int i = 0; i < output_array.size(); i++) {
            for (int j = 0; j < output_array[i].size(); j++) {
                for (int k = 0; k < output_array[i][j].size(); k++) {
                    if (1 == label[i][j][k]) {
                        //输出层误差项
                        output_delta_array[i][j][k] -= 1;
                        //log以e为底 自然对数
                        loss += log(output_array[i][j][k]);
                    }
                    //得到一个batch的平均误差值
                    output_delta_array[i][j][k] /= batch_size;
                }
            }
        }
    }

    loss = -loss / batch_size;
    return loss;
}

//归一化到0, 1 计算概率分布 交叉熵损失函数
template <typename DataType>
double Matrix<DataType>::CrossEntropyLoss(const Matrix3d& output_array, 
                                          const Matrix3d& label) { 
    if (!MatrixCheck(output_array, label, true)) {
        LOG(ERROR) << "matrix calculate cross entropy loss failed";
        return -1.0;
    }

    int batch_size = output_array.size();
    double loss = 0.0;
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : loss)
        for (int i = 0; i < output_array.size(); i++) {
            for (int j = 0; j < output_array[i].size(); j++) {
                for (int k = 0; k < output_array[i][j].size(); k++) {
                    if (1 == label[i][j][k]) {
                        //log以e为底 自然对数
                        loss += log(output_array[i][j][k]);
                    }
                }
            }
        }
    }

    loss = -loss / batch_size;
    return loss;
}


//均方误差 ((y - predict)**2.sum()) / 2 
template <typename DataType>
double Matrix<DataType>::MeanSquareError(const Matrix2d& output_matrix, 
                                         const Matrix2d& label) {
    if (output_matrix.size() != label.size()) {
        LOG(ERROR) << "calculate mean square error failed, two matrices rows is not equal";
        return -1;
    }

    for (int i = 0; i < output_matrix.size(); i++) {
        if (output_matrix[i].size() != label[i].size()) {
            LOG(ERROR) << "calculate mean square error failed, two matrices cols is not equal";
            return -1;
        }
    }

    //计算均方误差
    double sum = 0.0;
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : sum)
        //多线程数据竞争  加锁保护
        for (int i = 0; i < output_matrix.size(); i++) {
            for (int j = 0; j < output_matrix[i].size(); j++) {
                 sum += pow((label[i][j] - output_matrix[i][j]), 2);
            }
        }
    }

    return sum / 2;
}

//均方误差 ((y - predict)**2.sum()) / 2 
template <typename DataType>
double Matrix<DataType>::MeanSquareError(const Matrix3d& output_matrix, 
                                         const Matrix3d& label) {
    if (output_matrix.size() != label.size()) {
        LOG(ERROR) << "calculate mean square error failed, two matrices channels is not equal";
        return -1;
    }

    for (int i = 0; i < output_matrix.size(); i++) {
        if (output_matrix[i].size() != label[i].size()) {
            LOG(ERROR) << "calculate mean square error failed, two matrices rows is not equal";
            return -1;
        }
    }

    //计算均方误差
    double sum = 0.0;
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : sum)
        //多线程数据竞争  加锁保护
        for (int i = 0; i < output_matrix.size(); i++) {
            for (int j = 0; j < output_matrix[i].size(); j++) {
                for (int k = 0; k < output_matrix[i][j].size(); k++) {
                    sum += pow((label[i][j][k] - output_matrix[i][j][k]), 2);
                }
            }
        }
    }

    return sum / 2;
}

//均方误差 ((y - predict)**2.sum()) / 2 
template <typename DataType>
double Matrix<DataType>::MeanSquareError(const Matrix4d& output_matrix, 
                                         const Matrix4d& label) {
    if (output_matrix.size() != label.size()) {
        LOG(ERROR) << "calculate mean square error failed, two matrices batch size is not equal";
        return -1;
    }

    for (int i = 0; i < output_matrix.size(); i++) {
        if (output_matrix[i].size() != label[i].size()) {
            LOG(ERROR) << "calculate mean square error failed, two matrices channels is not equal";
            return -1;
        }
    }

    //计算均方误差
    double sum = 0.0;
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : sum)
        //多线程数据竞争  加锁保护
        for (int i = 0; i < output_matrix.size(); i++) {
            for (int j = 0; j < output_matrix[i].size(); j++) {
                for (int k = 0; k < output_matrix[i][j].size(); k++) {
                    for (int z = 0; z < output_matrix[i][j][k].size(); z++) {
                        sum += pow((label[i][j][k][z] - output_matrix[i][j][k][z]), 2);
                    }
                }
            }
        }
    }

    return sum / 2;
}

//2d不相等个数
template <typename DataType>
int32_t Matrix<DataType>::NotEqualTotal(const Matrix2d& output_matrix, 
                                        const Matrix2d& label) {
    if (output_matrix.size() != label.size()) {
        LOG(ERROR) << "get two matrices not equal total failed, two matrices rows is not equal";
        return -1;
    }

    for (int i = 0; i < output_matrix.size(); i++) {
        if (output_matrix[i].size() != label[i].size()) {
        LOG(ERROR) << "get two matrices not equal total failed, two matrices cols is not equal";
            return -1;
        }
    }

    int total = 0;
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : total)
        //多线程数据竞争  加锁保护
        for (int i = 0; i < output_matrix.size(); i++) {
            for (int j = 0; j < output_matrix[i].size(); j++) {
                if (label[i][j] != output_matrix[i][j]) {
                    total += 1;
                }
            }
        }
    }

    return total;
}

//3d不相等个数
template <typename DataType>
int32_t Matrix<DataType>::NotEqualTotal(const Matrix3d& output_matrix, 
                                        const Matrix3d& label) {
    if (output_matrix.size() != label.size()) {
        LOG(ERROR) << "get two matrices not equal total failed, two matrices channels is not equal";
        return -1;
    }

    for (int i = 0; i < output_matrix.size(); i++) {
        if (output_matrix[i].size() != label[i].size()) {
        LOG(ERROR) << "get two matrices not equal total failed, two matrices rows is not equal";
            return -1;
        }
    }

    int total = 0;
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : total)
        //多线程数据竞争  加锁保护
        for (int i = 0; i < output_matrix.size(); i++) {
            for (int j = 0; j < output_matrix[i].size(); j++) {
                for (int k = 0; k < output_matrix[i][j].size(); k++) {
                    if (label[i][j][k] != output_matrix[i][j][k]) {
                        total += 1;
                    }
                }
            }
        }
    }

    return total;
}

//4d不相等个数
template <typename DataType>
int32_t Matrix<DataType>::NotEqualTotal(const Matrix4d& output_matrix, 
                                        const Matrix4d& label) {
    if (output_matrix.size() != label.size()) {
        LOG(ERROR) << "get two matrices not equal total failed, two matrices channels is not equal";
        return -1;
    }

    for (int i = 0; i < output_matrix.size(); i++) {
        if (output_matrix[i].size() != label[i].size()) {
        LOG(ERROR) << "get two matrices not equal total failed, two matrices rows is not equal";
            return -1;
        }
    }

    int total = 0;
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) reduction(+ : total)
        //多线程数据竞争  加锁保护
        for (int i = 0; i < output_matrix.size(); i++) {
            for (int j = 0; j < output_matrix[i].size(); j++) {
                for (int k = 0; k < output_matrix[i][j].size(); k++) {
                    for (int z = 0; z < output_matrix[i][j][k].size(); z++) {
                        if (label[i][j][k][z] != output_matrix[i][j][k][z]) {
                            total += 1;
                        }
                    }
                }
            }
        }
    }

    return total;
}

// 补0填充
template <typename DataType>
int8_t Matrix<DataType>::ZeroPadding(const Matrix3d& source_matrix, 
                                     int32_t zero_padding, 
                                     Matrix3d& result_matrix) {
    //如果外圈补0是0的话 就不用补
    if (0 == zero_padding) {
        result_matrix = source_matrix;
        return 0;
    }
    //getshape 会check一下源矩阵
    auto shape = GetShape(source_matrix);
    int depth;
    int height;
    int width;
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix depth is empty";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix height is empty";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix width is empty";
        return -1;
    }

    //check一下结果矩阵 深度应该不变 行 列 应该加上zero_padding
    if (!MatrixCheck(result_matrix, depth, height + 2 * zero_padding, width + 2 * zero_padding, false)) {
        result_matrix.clear();
        result_matrix = Matrix3d(depth, Matrix2d(height + 2 * zero_padding, Matrix1d(width + 2 * zero_padding, 0)));
    }

    //直接调用getroi 来把中间那部分矩阵赋值了
    if (-1 == CopyTo(source_matrix, zero_padding, zero_padding, height, width, result_matrix)) {
        LOG(ERROR) << "matrix zero padding failed";
        return -1;
    }
     
    return 0;
}

// 补0填充
template <typename DataType>
int8_t Matrix<DataType>::ZeroPadding(const Matrix4d& source_matrix, 
                                     int32_t zero_padding, 
                                     Matrix4d& result_matrix) {
    //如果外圈补0是0的话 就不用补
    if (0 == zero_padding) {
        result_matrix = source_matrix;
        return 0;
    }
    //getshape 会check一下源矩阵
    auto shape = GetShape(source_matrix);
    int batch_size;
    int depth;
    int height;
    int width;
    std::tie(batch_size, depth, height, width) = shape;
    if (batch_size <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix batch size is empty";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix depth is empty";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix height is empty";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix zero padding failed, matrix width is empty";
        return -1;
    }

    //check一下结果矩阵 深度应该不变 行 列 应该加上zero_padding
    if (!MatrixCheck(result_matrix, batch_size, depth, height + 2 * zero_padding, width + 2 * zero_padding, false)) {
        result_matrix.clear();
        result_matrix = Matrix4d(batch_size, Matrix3d(depth, Matrix2d(height + 2 * zero_padding, Matrix1d(width + 2 * zero_padding, 0))));
    }

    //直接调用getroi 来把中间那部分矩阵赋值了
    if (-1 == CopyTo(source_matrix, zero_padding, zero_padding, height, width, result_matrix)) {
        LOG(ERROR) << "matrix zero padding failed";
        return -1;
    }
     
    return 0;
}

//将3d源矩阵copy到结果矩阵的roi区域 (x, y)起始行列 x+height y+width是结尾行列
template <typename DataType>
int8_t Matrix<DataType>::CopyTo(const Matrix3d& source_matrix, 
                                int32_t x, int32_t y, 
                                int32_t height, int32_t width,
                                Matrix3d& result_matrix) {
    if (x < 0) {
        LOG(ERROR) << "matrix copy to failed, input x < 0";
        return -1;
    }
    if (y < 0) {
        LOG(ERROR) << "matrix copy to failed, input y < 0";
        return -1;
    }
    if (height < x) {
        LOG(ERROR) << "matrix copy to failed, input height < x";
        return -1;
    }
    if (width < y) {
        LOG(ERROR) << "matrix copy to failed, input width < y";
        return -1;
    }

    //getshape 会check源矩阵
    auto shape = GetShape(source_matrix);
    int source_matrix_depth;
    int source_matrix_height;
    int source_matrix_width;
    std::tie(source_matrix_depth, source_matrix_height, source_matrix_width) = shape;
    if (source_matrix_depth <= 0) {
        LOG(ERROR) << "matrix copy to failed, input matrix depth is wrong";
        return -1;
    }
    if (source_matrix_height <= 0
            || source_matrix_height != height) {
        LOG(ERROR) << "matrix copy to failed, input matrix height is wrong";
        return -1;
    }
    if (source_matrix_width <= 0
            || source_matrix_width != width) {
        LOG(ERROR) << "matrix copy to failed, input matrix width is wrong";
        return -1;
    }

    //getshape 会check结果矩阵
    auto result_shape = GetShape(result_matrix);
    int result_matrix_depth;
    int result_matrix_height;
    int result_matrix_width;
    std::tie(result_matrix_depth, result_matrix_height, result_matrix_width) = result_shape;
    if (result_matrix_depth <= 0
            || result_matrix_depth != source_matrix_depth
            || result_matrix_height <= 0 
            || x + height > result_matrix_height 
            || result_matrix_width <= 0 
            || y + width > result_matrix_width) {
        LOG(ERROR) << "matrix copy to failed, result matrix is wrong";
        return -1;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //遍历数组 深度不变 行从x开始 列从y开始 一直到x+height y+width赋值给新矩阵
        for (int i = 0; i < source_matrix_depth; i++) {
            for (int j = 0, q = x; j < source_matrix_height; j++, q++) {
                for (int k = 0, w = y; k < source_matrix_width; k++, w++) {
                    result_matrix[i][q][w] = source_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
}    

//将4d源矩阵copy到结果矩阵的roi区域 (x, y)起始行列 x+height y+width是结尾行列
template <typename DataType>
int8_t Matrix<DataType>::CopyTo(const Matrix4d& source_matrix, 
                                int32_t x, int32_t y, 
                                int32_t height, int32_t width,
                                Matrix4d& result_matrix) {
    if (x < 0) {
        LOG(ERROR) << "matrix copy to failed, input x < 0";
        return -1;
    }
    if (y < 0) {
        LOG(ERROR) << "matrix copy to failed, input y < 0";
        return -1;
    }
    if (height < x) {
        LOG(ERROR) << "matrix copy to failed, input height < x";
        return -1;
    }
    if (width < y) {
        LOG(ERROR) << "matrix copy to failed, input width < y";
        return -1;
    }

    //getshape 会check源矩阵
    auto shape = GetShape(source_matrix);
    int source_matrix_batch_size;
    int source_matrix_depth;
    int source_matrix_height;
    int source_matrix_width;
    std::tie(source_matrix_batch_size, source_matrix_depth, source_matrix_height, source_matrix_width) = shape;
    if (source_matrix_batch_size <= 0) {
        LOG(ERROR) << "matrix copy to failed, input matrix batch size is wrong";
        return -1;
    }
    if (source_matrix_depth <= 0) {
        LOG(ERROR) << "matrix copy to failed, input matrix depth is wrong";
        return -1;
    }
    if (source_matrix_height <= 0
            || source_matrix_height != height) {
        LOG(ERROR) << "matrix copy to failed, input matrix height is wrong";
        return -1;
    }
    if (source_matrix_width <= 0
            || source_matrix_width != width) {
        LOG(ERROR) << "matrix copy to failed, input matrix width is wrong";
        return -1;
    }

    //getshape 会check结果矩阵
    auto result_shape = GetShape(result_matrix);
    int result_matrix_batch_size;
    int result_matrix_depth;
    int result_matrix_height;
    int result_matrix_width;
    std::tie(result_matrix_batch_size,  result_matrix_depth, result_matrix_height, result_matrix_width) = result_shape;
    if (result_matrix_depth <= 0
            || result_matrix_batch_size != source_matrix_batch_size
            || result_matrix_depth != source_matrix_depth
            || result_matrix_height <= 0 
            || x + height > result_matrix_height 
            || result_matrix_width <= 0 
            || y + width > result_matrix_width) {
        LOG(ERROR) << "matrix copy to failed, result matrix is wrong";
        return -1;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //遍历数组 深度不变 行从x开始 列从y开始 一直到x+height y+width赋值给新矩阵
        for (int n = 0; n < source_matrix_batch_size; n++) {
            for (int i = 0; i < source_matrix_depth; i++) {
                for (int j = 0, q = x; j < source_matrix_height; j++, q++) {
                    for (int k = 0, w = y; k < source_matrix_width; k++, w++) {
                        result_matrix[n][i][q][w] = source_matrix[n][i][j][k];
                    }
                }
            }
        }
    }

    return 0;
}    

//将源矩阵copy到结果矩阵
template <typename DataType>
int8_t Matrix<DataType>::CopyTo(const Matrix2d& source_matrix, 
                                int32_t& index, 
                                std::shared_ptr<double> result_matrix) {
    //check source matrix 
    if (!MatrixCheck(source_matrix, true)) {
       LOG(ERROR) << "matrix copy failed, input source matrix is wrong";
       return -1;
    }
    
    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            result_matrix.get()[index++] = source_matrix[i][j];
        }
    }

    return 0;
}

//将源矩阵copy到结果矩阵
template <typename DataType>
int8_t Matrix<DataType>::CopyTo(const Matrix3d& source_matrix, 
                                int32_t& index, 
                                std::shared_ptr<double> result_matrix) {
    //check source matrix 
    if (!MatrixCheck(source_matrix, true)) {
       LOG(ERROR) << "matrix copy failed, input source matrix is wrong";
       return -1;
    }
    
    for (int i = 0; i < source_matrix.size(); i++) {
        for (int j = 0; j < source_matrix[i].size(); j++) {
            for (int k = 0; k < source_matrix[i][j].size(); k++) {
                result_matrix.get()[index++] = source_matrix[i][j][k];
            }
        }
    }

    return 0;
}

//将源矩阵copy到结果矩阵
template <typename DataType>
int8_t Matrix<DataType>::CopyTo(std::shared_ptr<double> source_matrix, 
                                int32_t start_index, int32_t fc_layers_number, 
                                int32_t fc5_input_node, int32_t fc5_output_node, 
                                int32_t fc6_output_node, 
                                Matrix3d& weights_matrix,
                                Matrix3d& biases_matrix) {
    if (fc_layers_number <= 0) {
        LOG(ERROR) << "matrix copy failed, input fc layers number <= 0";
        return -1;
    }
    if (fc5_input_node <= 0) {
        LOG(ERROR) << "matrix copy failed, fc input node <= 0";
        return -1;
    }
    if (fc5_output_node <= 0) {
        LOG(ERROR) << "matrix copy failed, fc output node <= 0";
        return -1;
    }
    if (fc6_output_node <= 0) {
        LOG(ERROR) << "matrix copy failed, fc output node <= 0";
        return -1;
    }
    
    weights_matrix.clear();
    weights_matrix.reserve(fc_layers_number);
    biases_matrix.clear();
    biases_matrix.reserve(fc_layers_number);
    Matrix2d data1(fc5_output_node, Matrix1d(fc5_input_node));
    Matrix2d data2(fc5_output_node, Matrix1d(1));
    Matrix2d data3(fc6_output_node, Matrix1d(fc5_output_node));
    Matrix2d data4(fc6_output_node, Matrix1d(1));
    for (int i = 0; i < fc5_output_node; i++) {
        for (int j = 0; j < fc5_input_node; j++) {
            data1[i][j] = source_matrix.get()[start_index++];
        }
    }
    for (int i = 0; i < fc5_output_node; i++) {
        for (int j = 0; j < 1; j++) {
            data2[i][j] = source_matrix.get()[start_index++];
        }
    }
    for (int i = 0; i < fc6_output_node; i++) {
        for (int j = 0; j < fc5_output_node; j++) {
            data3[i][j] = source_matrix.get()[start_index++];
        }
    }
    for (int i = 0; i < fc6_output_node; i++) {
        for (int j = 0; j < 1; j++) {
            data4[i][j] = source_matrix.get()[start_index++];
        }
    }
    weights_matrix.push_back(data1);
    biases_matrix.push_back(data2);
    weights_matrix.push_back(data3);
    biases_matrix.push_back(data4);

    return 0;
}

//将源矩阵copy到结果矩阵
template <typename DataType>
int8_t Matrix<DataType>::CopyTo(std::shared_ptr<double> source_matrix, 
                                int32_t start_index, int32_t& end_index,  
                                int32_t filter_number, int32_t channel_number, 
                                int32_t filter_height, int32_t filter_width, 
                                Matrix4d& weights_matrix, 
                                Matrix1d& biases_matrix) {
    if (filter_number <= 0) {
        LOG(ERROR) << "matrix copy failed, input filter number <= 0";
        return -1;
    }
    if (channel_number <= 0) {
        LOG(ERROR) << "matrix copy failed, input channel number <= 0";
        return -1;
    }
    if (filter_height <= 0) {
        LOG(ERROR) << "matrix copy failed, input filter height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "matrix copy failed, input filter width <= 0";
        return -1;
    }

    weights_matrix = Matrix4d(filter_number, Matrix3d(channel_number, 
                              Matrix2d(filter_height, Matrix1d(filter_width))));
    biases_matrix = Matrix1d(filter_number);
    for (int i = 0; i < filter_number; i++) {
        for (int j = 0; j < channel_number; j++) {
            for (int k = 0; k < filter_height; k++) {
                for (int z = 0; z < filter_width; z++) {
                    weights_matrix[i][j][k][z] = source_matrix.get()[start_index++];
                }
            }
        }
        biases_matrix[i] = source_matrix.get()[start_index++];
    }
    
    end_index = start_index;
    return 0;
}

//得到2d矩阵的ROI   源矩阵是大矩阵  结果矩阵是取其roi小矩阵
template <typename DataType>
int8_t Matrix<DataType>::GetROI(const std::vector<std::vector<uint8_t>>& source_matrix, 
                                int32_t x, int32_t y, 
                                int32_t height, int32_t width,
                                int32_t stride, 
                                std::vector<std::vector<uint8_t>>& result_matrix) {
    if (x < 0) {
        LOG(ERROR) << "get matrix roi failed, input x < 0";
        return -1;
    }
    if (y < 0) {
        LOG(ERROR) << "get matrix roi failed, input y < 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix roi failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix roi failed, input width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix roi failed, input stride <= 0";
        return -1;
    }

    //getshape 会check源矩阵
    auto shape = GetShape(source_matrix);
    int source_matrix_height;
    int source_matrix_width;
    std::tie(source_matrix_height, source_matrix_width) = shape;
    if (source_matrix_height <= 0
            || x + height > source_matrix_height) {
        LOG(ERROR) << "get matrix roi failed, input matrix height is wrong";
        return -1;
    }
    if (source_matrix_width <= 0
            || y + width > source_matrix_width) {
        LOG(ERROR) << "get matrix roi failed, input matrix width is wrong";
        return -1;
    }

    //check结果矩阵
    int result_matrix_width;
    if (!MatrixCheck(result_matrix, height, width, false)) {
        result_matrix.clear();
        result_matrix = std::vector<std::vector<uint8_t>>(height, std::vector<uint8_t>(width));
    }

    //开始的行列 是步长*行列
    int start_row = x * stride;
    int start_col = y * stride;
 
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        //遍历数组 深度不变 行从x开始 列从y开始 一直到x+height y+width赋值给新矩阵
        //#pragma omp for schedule(static) 
        for (int k = 0; k < 1; k++) {
            for (int i = 0, q = start_row; i < height; i++, q++) {
                for (int j = 0, w = start_col; j < width; j++, w++) {
                    result_matrix[i][j] = source_matrix[q][w];
                }
            }
        }
    }

    return 0;
}

//得到2d矩阵的ROI   源矩阵是大矩阵  结果矩阵是取其roi小矩阵
template <typename DataType>
int8_t Matrix<DataType>::GetROI(const std::vector<std::vector<double>>& source_matrix, 
                                int32_t x, int32_t y, 
                                int32_t height, int32_t width,
                                int32_t stride, 
                                std::vector<std::vector<double>>& result_matrix) {
    if (x < 0) {
        LOG(ERROR) << "get matrix roi failed, input x < 0";
        return -1;
    }
    if (y < 0) {
        LOG(ERROR) << "get matrix roi failed, input y < 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix roi failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix roi failed, input width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix roi failed, input stride <= 0";
        return -1;
    }

    //getshape 会check源矩阵
    auto shape = GetShape(source_matrix);
    int source_matrix_height;
    int source_matrix_width;
    std::tie(source_matrix_height, source_matrix_width) = shape;
    if (source_matrix_height <= 0
            || x + height > source_matrix_height) {
        LOG(ERROR) << "get matrix roi failed, input matrix height is wrong";
        return -1;
    }
    if (source_matrix_width <= 0
            || y + width > source_matrix_width) {
        LOG(ERROR) << "get matrix roi failed, input matrix width is wrong";
        return -1;
    }

    //check结果矩阵
    int result_matrix_width;
    if (!MatrixCheck(result_matrix, height, width, false)) {
        result_matrix.clear();
        result_matrix = std::vector<std::vector<double>>(height, std::vector<double>(width));
    }

    //开始的行列 是步长*行列
    int start_row = x * stride;
    int start_col = y * stride;
 
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        //遍历数组 深度不变 行从x开始 列从y开始 一直到x+height y+width赋值给新矩阵
        //#pragma omp for schedule(static) 
        for (int k = 0; k < 1; k++) {
            for (int i = 0, q = start_row; i < height; i++, q++) {
                for (int j = 0, w = start_col; j < width; j++, w++) {
                    result_matrix[i][j] = source_matrix[q][w];
                }
            }
        }
    }

    return 0;
}


//得到3d矩阵的ROI   源矩阵是大矩阵  结果矩阵是取其roi小矩阵
template <typename DataType>
int8_t Matrix<DataType>::GetROI(const std::vector<std::vector<std::vector<uint8_t>>>& source_matrix, 
                                int32_t x, int32_t y, 
                                int32_t height, int32_t width,
                                int32_t stride, 
                                std::vector<std::vector<std::vector<uint8_t>>>& result_matrix) {
    if (x < 0) {
        LOG(ERROR) << "get matrix roi failed, input x < 0";
        return -1;
    }
    if (y < 0) {
        LOG(ERROR) << "get matrix roi failed, input y < 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix roi failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix roi failed, input width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix roi failed, input stride <= 0";
        return -1;
    }

    //getshape 会check源矩阵
    auto shape = GetShape(source_matrix);
    int source_matrix_depth;
    int source_matrix_height;
    int source_matrix_width;
    std::tie(source_matrix_depth, source_matrix_height, source_matrix_width) = shape;
    if (source_matrix_depth <= 0) {
        LOG(ERROR) << "get matrix roi failed, input matrix depth is wrong";
        return -1;
    }
    if (source_matrix_height <= 0
            || x + height > source_matrix_height) {
        LOG(ERROR) << "get matrix roi failed, input matrix height is wrong";
        return -1;
    }
    if (source_matrix_width <= 0
            || y + width > source_matrix_width) {
        LOG(ERROR) << "get matrix roi failed, input matrix width is wrong";
        return -1;
    }

    //check结果矩阵
    int result_matrix_width;
    if (!MatrixCheck(result_matrix, source_matrix_depth, height, width, false)) {
        result_matrix.clear();
        result_matrix = std::vector<std::vector<std::vector<uint8_t>>>(source_matrix_depth,
                                                                       std::vector<std::vector<uint8_t>>(height, 
                                                                       std::vector<uint8_t>(width)));
    }

    //开始的行列 是步长*行列
    int start_row = x * stride;
    int start_col = y * stride;
 
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //遍历数组 深度不变 行从x开始 列从y开始 一直到x+height y+width赋值给新矩阵
        for (int i = 0; i < source_matrix_depth; i++) {
            for (int j = 0, q = start_row; j < height; j++, q++) {
                for (int k = 0, w = start_col; k < width; k++, w++) {
                    result_matrix[i][j][k] = source_matrix[i][q][w];
                }
            }
        }
    }

    return 0;
}

//得到3d矩阵的ROI   源矩阵是大矩阵  结果矩阵是取其roi小矩阵
template <typename DataType>
int8_t Matrix<DataType>::GetROI(const std::vector<std::vector<std::vector<double>>>& source_matrix, 
                                int32_t x, int32_t y, 
                                int32_t height, int32_t width,
                                int32_t stride, 
                                std::vector<std::vector<std::vector<double>>>& result_matrix) {
    if (x < 0) {
        LOG(ERROR) << "get matrix roi failed, input x < 0";
        return -1;
    }
    if (y < 0) {
        LOG(ERROR) << "get matrix roi failed, input y < 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix roi failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix roi failed, input width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix roi failed, input stride <= 0";
        return -1;
    }

    //getshape 会check源矩阵
    auto shape = GetShape(source_matrix);
    int source_matrix_depth;
    int source_matrix_height;
    int source_matrix_width;
    std::tie(source_matrix_depth, source_matrix_height, source_matrix_width) = shape;
    if (source_matrix_depth <= 0) {
        LOG(ERROR) << "get matrix roi failed, input matrix depth is wrong";
        return -1;
    }
    if (source_matrix_height <= 0
            || x + height > source_matrix_height) {
        LOG(ERROR) << "get matrix roi failed, input matrix height is wrong";
        return -1;
    }
    if (source_matrix_width <= 0
            || y + width > source_matrix_width) {
        LOG(ERROR) << "get matrix roi failed, input matrix width is wrong";
        return -1;
    }

    //check结果矩阵
    int result_matrix_width;
    if (!MatrixCheck(result_matrix, source_matrix_depth, height, width, false)) {
        result_matrix.clear();
        result_matrix = std::vector<std::vector<std::vector<double>>>(source_matrix_depth,
                                                                      std::vector<std::vector<double>>(height,  
                                                                      std::vector<double>(width)));
    }

    //开始的行列 是步长*行列
    int start_row = x * stride;
    int start_col = y * stride;
 
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //遍历数组 深度不变 行从x开始 列从y开始 一直到x+height y+width赋值给新矩阵
        for (int i = 0; i < source_matrix_depth; i++) {
            for (int j = 0, q = start_row; j < height; j++, q++) {
                for (int k = 0, w = start_col; k < width; k++, w++) {
                    result_matrix[i][j][k] = source_matrix[i][q][w];
                }
            }
        }
    }

    return 0;
}

//得到4d矩阵的ROI   源矩阵是大矩阵  结果矩阵是取其roi小矩阵
template <typename DataType>
int8_t Matrix<DataType>::GetROI(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& source_matrix, 
                                int32_t x, int32_t y, 
                                int32_t height, int32_t width,
                                int32_t stride, 
                                std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& result_matrix) {
    if (x < 0) {
        LOG(ERROR) << "get matrix roi failed, input x < 0";
        return -1;
    }
    if (y < 0) {
        LOG(ERROR) << "get matrix roi failed, input y < 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix roi failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix roi failed, input width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix roi failed, input stride <= 0";
        return -1;
    }

    //getshape 会check源矩阵
    auto shape = GetShape(source_matrix);
    int source_matrix_batch_size;
    int source_matrix_depth;
    int source_matrix_height;
    int source_matrix_width;
    std::tie(source_matrix_batch_size, source_matrix_depth, source_matrix_height, source_matrix_width) = shape;
    if (source_matrix_batch_size <= 0) {
        LOG(ERROR) << "get matrix roi failed, input matrix batch size is wrong";
        return -1;
    }
    if (source_matrix_depth <= 0) {
        LOG(ERROR) << "get matrix roi failed, input matrix depth is wrong";
        return -1;
    }
    if (source_matrix_height <= 0
            || x + height > source_matrix_height) {
        LOG(ERROR) << "get matrix roi failed, input matrix height is wrong";
        return -1;
    }
    if (source_matrix_width <= 0
            || y + width > source_matrix_width) {
        LOG(ERROR) << "get matrix roi failed, input matrix width is wrong";
        return -1;
    }

    //check结果矩阵
    int result_matrix_width;
    if (!MatrixCheck(result_matrix, source_matrix_batch_size, source_matrix_depth, height, width, false)) {
        result_matrix.clear();
        result_matrix = std::vector<std::vector<std::vector<std::vector<uint8_t>>>>(source_matrix_batch_size, 
                                                                    std::vector<std::vector<std::vector<uint8_t>>>(source_matrix_depth,
                                                                    std::vector<std::vector<uint8_t>>(height, 
                                                                    std::vector<uint8_t>(width))));
    }

    //开始的行列 是步长*行列
    int start_row = x * stride;
    int start_col = y * stride;
 
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //遍历数组 深度不变 行从x开始 列从y开始 一直到x+height y+width赋值给新矩阵
        for (int n = 0; n < source_matrix_batch_size; n++) {
            for (int i = 0; i < source_matrix_depth; i++) {
                for (int j = 0, q = start_row; j < height; j++, q++) {
                    for (int k = 0, w = start_col; k < width; k++, w++) {
                        result_matrix[n][i][j][k] = source_matrix[n][i][q][w];
                    }
                }
            }
        }
    }

    return 0;
}

//得到4d矩阵的ROI   源矩阵是大矩阵  结果矩阵是取其roi小矩阵
template <typename DataType>
int8_t Matrix<DataType>::GetROI(const std::vector<std::vector<std::vector<std::vector<double>>>>& source_matrix, 
                                int32_t x, int32_t y, 
                                int32_t height, int32_t width,
                                int32_t stride, 
                                std::vector<std::vector<std::vector<std::vector<double>>>>& result_matrix) {
    if (x < 0) {
        LOG(ERROR) << "get matrix roi failed, input x < 0";
        return -1;
    }
    if (y < 0) {
        LOG(ERROR) << "get matrix roi failed, input y < 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix roi failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix roi failed, input width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix roi failed, input stride <= 0";
        return -1;
    }

    //getshape 会check源矩阵
    auto shape = GetShape(source_matrix);
    int source_matrix_batch_size;
    int source_matrix_depth;
    int source_matrix_height;
    int source_matrix_width;
    std::tie(source_matrix_batch_size, source_matrix_depth, source_matrix_height, source_matrix_width) = shape;
    if (source_matrix_batch_size <= 0) {
        LOG(ERROR) << "get matrix roi failed, input matrix batch size is wrong";
        return -1;
    }
    if (source_matrix_depth <= 0) {
        LOG(ERROR) << "get matrix roi failed, input matrix depth is wrong";
        return -1;
    }
    if (source_matrix_height <= 0
            || x + height > source_matrix_height) {
        LOG(ERROR) << "get matrix roi failed, input matrix height is wrong";
        return -1;
    }
    if (source_matrix_width <= 0
            || y + width > source_matrix_width) {
        LOG(ERROR) << "get matrix roi failed, input matrix width is wrong";
        return -1;
    }

    //check结果矩阵
    int result_matrix_width;
    if (!MatrixCheck(result_matrix, source_matrix_batch_size, source_matrix_depth, height, width, false)) {
        result_matrix.clear();
        result_matrix = std::vector<std::vector<std::vector<std::vector<double>>>>(source_matrix_batch_size, 
                                                                    std::vector<std::vector<std::vector<double>>>(source_matrix_depth,
                                                                    std::vector<std::vector<double>>(height, 
                                                                    std::vector<double>(width))));
    }

    //开始的行列 是步长*行列
    int start_row = x * stride;
    int start_col = y * stride;
 
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //遍历数组 深度不变 行从x开始 列从y开始 一直到x+height y+width赋值给新矩阵
        for (int n = 0; n < source_matrix_batch_size; n++) {
            for (int i = 0; i < source_matrix_depth; i++) {
                for (int j = 0, q = start_row; j < height; j++, q++) {
                    for (int k = 0, w = start_col; k < width; k++, w++) {
                        result_matrix[n][i][j][k] = source_matrix[n][i][q][w];
                    }
                }
            }
        }
    }

    return 0;
}


//2d卷积运算
template <typename DataType>
int8_t Matrix<DataType>::Convolution(const Matrix2d& source_matrix, 
                                     const Matrix2d& filter_matrix, 
                                     Matrix2d& result_matrix,
                                     double bias, int32_t stride) {
    //check一下输入矩阵 和 卷积核矩阵
    if (!MatrixCheck(source_matrix, true)) {
       LOG(ERROR) << "matrix convolution operator failed, input source matrix is wrong";
       return -1;
    }
    //得到卷积核shape 顺便check了矩阵
    auto filter_shape = GetShape(filter_matrix);
    int filter_height;
    int filter_width;
    std::tie(filter_height, filter_width) = filter_shape;
    if (filter_height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix width <= 0";
        return -1;
    }
    
    //得到特征图shape 顺便check了结果矩阵
    auto shape = GetShape(result_matrix);
    int height;
    int width;
    std::tie(height, width) = shape;
    if (height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map width <= 0";
        return -1;
    }

    //遍历特征图 计算每一个值
    Matrix2d source_matrix_roi;

    //这里不用多线程 因为里面都是调用的函数 使用的openmp 这样性能更好
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, stride, 
                             source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }
            //roi和 filter(相乘相加) 进行hadamark积(相乘) 后累加(相加) 
            if (-1 == HadamarkProduct(source_matrix_roi, 
                                      filter_matrix, 
                                      source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }

            //最后对roi求和 加上 该filter的偏置 就是特征图对应位置的卷积结果 
            result_matrix[i][j] = Sum(source_matrix_roi) + bias;
        }
    }

    return 0;
}

//2d卷积运算
template <typename DataType>
int8_t Matrix<DataType>::Convolution(const std::vector<std::vector<uint8_t>>& source_matrix, 
                                     const std::vector<std::vector<double>>& filter_matrix, 
                                     std::vector<std::vector<double>>& result_matrix,
                                     double bias, int32_t stride) {
    //check一下输入矩阵 和 卷积核矩阵
    if (!MatrixCheck(source_matrix, true)) {
       LOG(ERROR) << "matrix convolution operator failed, input source matrix is wrong";
       return -1;
    }
    //得到卷积核shape 顺便check了矩阵
    auto filter_shape = GetShape(filter_matrix);
    int filter_height;
    int filter_width;
    std::tie(filter_height, filter_width) = filter_shape;
    if (filter_height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix width <= 0";
        return -1;
    }
    
    //得到特征图shape 顺便check了结果矩阵
    auto shape = GetShape(result_matrix);
    int height;
    int width;
    std::tie(height, width) = shape;
    if (height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map width <= 0";
        return -1;
    }

    //遍历特征图 计算每一个值
    std::vector<std::vector<uint8_t>> source_matrix_roi;
    std::vector<std::vector<double>> prod_result_matrix;

    //这里不用多线程 因为里面都是调用的函数 使用的openmp 这样性能更好
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, stride, 
                             source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }
                
            //roi和 filter(相乘相加) 进行hadamark积(相乘) 后累加(相加) 
            if (-1 == HadamarkProduct(source_matrix_roi, 
                                      filter_matrix, 
                                      prod_result_matrix)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }
            //最后对roi求和 加上 该filter的偏置 就是特征图对应位置的卷积结果 
            result_matrix[i][j] = Sum(prod_result_matrix) + bias;
        }
    }

    return 0;
}

//3d卷积运算
template <typename DataType>
int8_t Matrix<DataType>::Convolution(const Matrix3d& source_matrix, 
                                     const Matrix3d& filter_matrix, 
                                     Matrix2d& result_matrix,
                                     double bias, int32_t stride) {
    //check一下输入矩阵 和 卷积核矩阵
    if (!MatrixCheck(source_matrix, true)) {
       LOG(ERROR) << "matrix convolution operator failed, input source matrix is wrong";
       return -1;
    }
    //得到卷积核shape 顺便check了矩阵
    auto filter_shape = GetShape(filter_matrix);
    int filter_depth;
    int filter_height;
    int filter_width;
    std::tie(filter_depth, filter_height, filter_width) = filter_shape;
    if (filter_depth <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix depth <= 0";
        return -1;
    }
    if (filter_height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix width <= 0";
        return -1;
    }
    
    //得到特征图shape 顺便check了结果矩阵
    auto shape = GetShape(result_matrix);
    int height;
    int width;
    std::tie(height, width) = shape;
    if (height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map width <= 0";
        return -1;
    }

    //遍历特征图 计算每一个值
    Matrix3d source_matrix_roi;

    //这里不用多线程 因为里面都是调用的函数 使用的openmp 这样性能更好
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, stride, 
                             source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }
            //roi和 filter(相乘相加) 进行hadamark积(相乘) 后累加(相加) 
            if (-1 == HadamarkProduct(source_matrix_roi, 
                                      filter_matrix, 
                                      source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }
            //最后对roi求和 加上 该filter的偏置 就是特征图对应位置的卷积结果 
            result_matrix[i][j] = Sum(source_matrix_roi) + bias;
        }
    }

    return 0;
}

//3d卷积运算
template <typename DataType>
int8_t Matrix<DataType>::Convolution(const std::vector<std::vector<std::vector<uint8_t>>>& source_matrix, 
                                     const std::vector<std::vector<std::vector<double>>>& filter_matrix, 
                                     std::vector<std::vector<double>>& result_matrix,
                                     double bias, int32_t stride) {
    //check一下输入矩阵 和 卷积核矩阵
    if (!MatrixCheck(source_matrix, true)) {
       LOG(ERROR) << "matrix convolution operator failed, input source matrix is wrong";
       return -1;
    }
    //得到卷积核shape 顺便check了矩阵
    auto filter_shape = GetShape(filter_matrix);
    int filter_depth;
    int filter_height;
    int filter_width;
    std::tie(filter_depth, filter_height, filter_width) = filter_shape;
    if (filter_depth <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix depth <= 0";
        return -1;
    }
    if (filter_height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix width <= 0";
        return -1;
    }
    
    //得到特征图shape 顺便check了结果矩阵
    auto shape = GetShape(result_matrix);
    int height;
    int width;
    std::tie(height, width) = shape;
    if (height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map width <= 0";
        return -1;
    }

    //遍历特征图 计算每一个值
    std::vector<std::vector<std::vector<uint8_t>>> source_matrix_roi;
    std::vector<std::vector<std::vector<double>>> prod_result_matrix;

    //这里不用多线程 因为里面都是调用的函数 使用的openmp 这样性能更好
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, stride, 
                             source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }
                
            //roi和 filter(相乘相加) 进行hadamark积(相乘) 后累加(相加) 
            if (-1 == HadamarkProduct(source_matrix_roi, 
                                      filter_matrix, 
                                      prod_result_matrix)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }
            //最后对roi求和 加上 该filter的偏置 就是特征图对应位置的卷积结果 
            result_matrix[i][j] = Sum(prod_result_matrix) + bias;
        }
    }

    return 0;
}

//3d矩阵翻转 180度
template <typename DataType>
int8_t Matrix<DataType>::Flip(const Matrix3d& source_matrix, 
                              Matrix3d& result_matrix) {
    //getshape check一下源矩阵
    auto shape = GetShape(source_matrix);
    int depth;
    int height;
    int width;
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "matrix flip failed, source matrix depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "matrix flip failed, source matrix height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix flip failed, source matrix width <= 0";
        return -1;
    }
    
    //check一下结果矩阵
    if (!MatrixCheck(result_matrix, depth, height, width, false)) {
        result_matrix = source_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //矩阵翻转180度 遍历每个深度 每个行和列都翻转了180度
        for (int i = 0; i < depth; i++) {
            for (int j = 0, x = height - 1; j < height && x >= 0; j++, x--) {
                for (int k = 0, y = width - 1; k < width && y >= 0; k++, y--) {
                    result_matrix[i][x][y] = source_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
}

//4d矩阵翻转 180度
template <typename DataType>
int8_t Matrix<DataType>::Flip(const Matrix4d& source_matrix, 
                              Matrix4d& result_matrix) {
    //getshape check一下源矩阵
    auto shape = GetShape(source_matrix);
    int batch_size;
    int depth;
    int height;
    int width;
    std::tie(batch_size, depth, height, width) = shape;
    if (batch_size <= 0) {
        LOG(ERROR) << "matrix flip failed, source matrix batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "matrix flip failed, source matrix depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "matrix flip failed, source matrix height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix flip failed, source matrix width <= 0";
        return -1;
    }
    
    //check一下结果矩阵
    if (!MatrixCheck(result_matrix, batch_size, depth, height, width, false)) {
        result_matrix = source_matrix;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //矩阵翻转180度 遍历每个深度 每个行和列都翻转了180度
        for (int n = 0; n < batch_size; n++) {
            for (int i = 0; i < depth; i++) {
                for (int j = 0, x = height - 1; j < height && x >= 0; j++, x--) {
                    for (int k = 0, y = width - 1; k < width && y >= 0; k++, y--) {
                        result_matrix[n][i][x][y] = source_matrix[n][i][j][k];
                    }
                }
            }
        }
    }

    return 0;
}

//扩展误差项(敏感图) 还原为步长为1时对应的sensitivity map
template <typename DataType>
int8_t Matrix<DataType>::ExpandSensitivityMap(const Matrix3d& input_sensitivity_matrix, 
                                              int32_t input_height,  int32_t input_width, 
                                              int32_t filter_number, int32_t filter_height, 
                                              int32_t filter_width,  
                                              int32_t output_height, int32_t output_width, 
                                              int32_t zero_padding,  int32_t stride, 
                                              Matrix3d& output_sensitivity_matrix) {
    if (input_height <= 0
            || input_width <= 0
            || filter_height <= 0
            || filter_width <= 0
            || filter_number <= 0
            || output_height <= 0
            || output_width <= 0
            || zero_padding < 0
            || stride <= 0) {
        LOG(ERROR) << "expand sensitivity map failed, input parameters is wrong";
        return -1;
    }
    if (!MatrixCheck(input_sensitivity_matrix, filter_number, 
                     output_height, output_width, true)) {
        LOG(ERROR) << "expand sensitivity map failed, input sensitivity map shape is wrong";
        return -1;
    }
    
    //得到步长为1时的sensitivity map的深度 高 宽
    int expanded_depth = filter_number;
    int expanded_height = input_height - filter_height + 2 * zero_padding + 1;
    int expanded_width = input_width - filter_width + 2 * zero_padding + 1;
    //构造新的sensitivity map 原来值赋值到相应位置 其余地方补0
    if (!MatrixCheck(output_sensitivity_matrix, expanded_depth, 
                     expanded_height, expanded_width, false)) {
        CreateZeros(expanded_depth, expanded_height, expanded_width, output_sensitivity_matrix);
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //遍历output的shape 也就是输入敏感图的shape
        for (int i = 0; i < filter_number; i++) {
            for (int j = 0; j < output_height; j++) {
                for (int k = 0; k < output_width; k++) {
                    int row_pos = j * stride;
                    int col_pos = k * stride;
                    //步长为S时sensitivity map跳过了步长为1时的那些值 那些值赋值为0
                    output_sensitivity_matrix[i][row_pos][col_pos] = input_sensitivity_matrix[i][j][k];
                }
            }
        }
    }

    return 0;
}



/*
 * 卷积层的前向计算
 * 先给输入数组补0填充 
 * 遍历每一个filter 在输入图像上卷积 得到每一个filter提取的特征图
 * 最后经过ReLu激活函数 得到前向计算的输出结果特征图
 */
template <typename DataType>
int8_t Matrix<DataType>::ConvolutionalForward(const Matrix4d& source_matrix, 
                                              const Matrix4d& filter_weights_matrix, 
                                              const Matrix1d& biases_matrix,  
                                              Matrix4d& result_matrix,
                                              int32_t stride) {
    //check一下输入矩阵 和 卷积核矩阵
    if (!MatrixCheck(source_matrix, true)) {
       LOG(ERROR) << "matrix convolution operator failed, input source matrix is wrong";
       return -1;
    }
    //得到卷积核shape 顺便check了矩阵
    auto filter_shape = GetShape(filter_weights_matrix);
    int filter_number;
    int filter_depth;
    int filter_height;
    int filter_width;
    std::tie(filter_number, filter_depth, filter_height, filter_width) = filter_shape;
    if (filter_number <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix number <= 0";
        return -1;
    }
    if (filter_depth <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix depth <= 0";
        return -1;
    }
    if (filter_height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix width <= 0";
        return -1;
    }
    
    //得到特征图shape 顺便check了结果矩阵
    auto shape = GetShape(result_matrix);
    int batch_size;
    int depth;
    int height;
    int width;
    std::tie(batch_size, depth, height, width) = shape;
    if (batch_size <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map width <= 0";
        return -1;
    }

    //遍历特征图 计算每一个值
    Matrix4d source_matrix_roi;
    Matrix4d prod_matrix;
    
    //这里不用多线程 因为里面都是调用的函数 使用的openmp 这样性能更好
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, stride, 
                             source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }
            for (int f = 0; f < filter_number; f++) {
                //roi和 filter(相乘相加) 进行hadamark积(相乘) 后累加(相加) 
                if (-1 == HadamarkProduct(source_matrix_roi, 
                                          filter_weights_matrix[f], 
                                          prod_matrix)) {
                    LOG(ERROR) << "matrix convolution operator failed";
                    return -1;
                }
                for (int n = 0; n < source_matrix_roi.size(); n++) {
                    //最后对roi求和 加上 该filter的偏置 就是特征图对应位置的卷积结果 
                    result_matrix[n][f][i][j] = std::max<double>(0.0, Sum(prod_matrix[n]) + biases_matrix[f]);
                }
            }
        }
    }

    return 0;
}

/*
 * 卷积层的前向计算
 * 先给输入数组补0填充 
 * 遍历每一个filter 在输入图像上卷积 得到每一个filter提取的特征图
 * 最后经过ReLu激活函数 得到前向计算的输出结果特征图
 */
template <typename DataType>
int8_t Matrix<DataType>::ConvolutionalForward(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& source_matrix, 
                                              const std::vector<std::vector<std::vector<std::vector<double>>>>& filter_weights_matrix, 
                                              const std::vector<double>& biases_matrix, 
                                              std::vector<std::vector<std::vector<std::vector<double>>>>& result_matrix, 
                                              int32_t stride, bool aa) {
    //check一下输入矩阵 和 卷积核矩阵
    if (!MatrixCheck(source_matrix, true)) {
       LOG(ERROR) << "matrix convolution operator failed, input source matrix is wrong";
       return -1;
    }
    //得到卷积核shape 顺便check了矩阵
    auto filter_shape = GetShape(filter_weights_matrix);
    int filter_number;
    int filter_depth;
    int filter_height;
    int filter_width;
    std::tie(filter_number, filter_depth, filter_height, filter_width) = filter_shape;
    if (filter_number <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix number <= 0";
        return -1;
    }
    if (filter_depth <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix depth <= 0";
        return -1;
    }
    if (filter_height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, filter matrix width <= 0";
        return -1;
    }
    
    //得到特征图shape 顺便check了结果矩阵
    auto shape = GetShape(result_matrix);
    int batch_size;
    int depth;
    int height;
    int width;
    std::tie(batch_size, depth, height, width) = shape;
    if (batch_size <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "matrix convolution operator failed, feature map width <= 0";
        return -1;
    }

    //遍历特征图 计算每一个值
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> source_matrix_roi; 
    std::vector<std::vector<std::vector<std::vector<double>>>> prod_matrix; 
    
    //这里不用多线程 因为里面都是调用的函数 使用的openmp 这样性能更好
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, stride, 
                             source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }
            
            for (int f = 0; f < filter_number; f++) {
                //roi和 filter(相乘相加) 进行hadamark积(相乘) 后累加(相加) 
                if (-1 == HadamarkProduct(source_matrix_roi, 
                                          filter_weights_matrix[f], 
                                          prod_matrix)) {
                    LOG(ERROR) << "matrix convolution operator failed";
                    return -1;
                }

                for (int n = 0; n < source_matrix_roi.size(); n++) {
                    //最后对roi求和 加上 该filter的偏置 就是特征图对应位置的卷积结果 
                    result_matrix[n][f][i][j] = std::max<double>(0.0, Sum(prod_matrix[n]) + biases_matrix[f]);
                }
            }
        }
    }

    return 0;
}

//4d卷积层反向传播 误差传递 batch_size channels height width
template <typename DataType>
int8_t Matrix<DataType>::ConvolutionalBpSensitivityMap(const Matrix4d& input_sensitivity_matrix, 
                                                       const Matrix4d& weights_matrix, 
                                                       const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& input_matrix, 
                                                       Matrix4d& delta_matrix) {
    int batch_size;
    int depth;
    int height;
    int width;
    int filter_number;
    int filter_depth;
    int filter_height;
    int filter_width;
    
    auto shape = GetShape(delta_matrix);
    std::tie(batch_size, depth, height, width) = shape;
    shape = GetShape(weights_matrix);
    std::tie(filter_number, filter_depth, filter_height, filter_width) = shape;
    
    if (batch_size <= 0
            || depth <= 0
            || height <= 0
            || width <= 0
            || filter_number <= 0
            || filter_depth <= 0
            || filter_height <= 0
            || filter_width <= 0) {
        LOG(ERROR) << "convolution calculate bp sensitivity map failed, input parameters is empty";
        return -1;
    }

    Matrix4d flipped_weights_matrix;
    Flip(weights_matrix, flipped_weights_matrix);
    Matrix4d source_matrix_roi;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(input_sensitivity_matrix, i, j, 
                             filter_height, filter_width, 1, 
                             source_matrix_roi)) {
                LOG(ERROR) << "convolution calculate bp sensitivity map failed";
                return -1;
            }
            
            for (int f = 0; f < filter_number; f++) {
                for (int n = 0; n < batch_size; n++) {
                    for (int d = 0; d < depth; d++) {
                        if (input_matrix[n][d][i][j] <= 0) {
                            delta_matrix[n][d][i][j] = 0.0;
                            continue;
                        }
                        for (int x = 0; x < filter_height; x++) {
                            for (int y = 0; y < filter_width; y++) {
                                delta_matrix[n][d][i][j] += source_matrix_roi[n][f][x][y] * 
                                                            flipped_weights_matrix[f][d][x][y];
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

//4d卷积层反向传播 误差传递 batch_size channels height width
template <typename DataType>
int8_t Matrix<DataType>::ConvolutionalBpSensitivityMap(const Matrix4d& input_sensitivity_matrix, 
                                                       const Matrix4d& weights_matrix, 
                                                       const std::vector<std::vector<std::vector<std::vector<double>>>>& input_matrix, 
                                                       Matrix4d& delta_matrix) {
    int batch_size;
    int depth;
    int height;
    int width;
    int filter_number;
    int filter_depth;
    int filter_height;
    int filter_width;
    
    auto shape = GetShape(delta_matrix);
    std::tie(batch_size, depth, height, width) = shape;
    shape = GetShape(weights_matrix);
    std::tie(filter_number, filter_depth, filter_height, filter_width) = shape;
    
    if (batch_size <= 0
            || depth <= 0
            || height <= 0
            || width <= 0
            || filter_number <= 0
            || filter_depth <= 0
            || filter_height <= 0
            || filter_width <= 0) {
        LOG(ERROR) << "convolution calculate bp sensitivity map failed, input parameters is empty";
        return -1;
    }

    Matrix4d flipped_weights_matrix;
    Flip(weights_matrix, flipped_weights_matrix);
    Matrix4d source_matrix_roi;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(input_sensitivity_matrix, i, j, 
                             filter_height, filter_width, 1, 
                             source_matrix_roi)) {
                LOG(ERROR) << "convolution calculate bp sensitivity map failed";
                return -1;
            }
            
            for (int f = 0; f < filter_number; f++) {
                for (int n = 0; n < batch_size; n++) {
                    for (int d = 0; d < depth; d++) {
                        if (input_matrix[n][d][i][j] <= 0) {
                            delta_matrix[n][d][i][j] = 0.0;
                            continue;
                        }
                        for (int x = 0; x < filter_height; x++) {
                            for (int y = 0; y < filter_width; y++) {
                                delta_matrix[n][d][i][j] += source_matrix_roi[n][f][x][y] * 
                                                            flipped_weights_matrix[f][d][x][y];
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

//4d卷积层反向传播 计算梯度 batch_size channels height width
template <typename DataType>
int8_t Matrix<DataType>::ConvolutionalBpGradient(const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& input_matrix, 
                                                 const Matrix4d& sensitivity_matrix, 
                                                 Matrix4d& weights_gradient_matrix, 
                                                 Matrix1d& biases_gradient_matrix) {
    int filter_number;
    int depth;
    int height;
    int width;
    int batch_size;
    int _filter_number;
    int filter_height;
    int filter_width;
    
    auto shape = GetShape(weights_gradient_matrix);
    std::tie(filter_number, depth, height, width) = shape;
    auto _shape = GetShape(sensitivity_matrix);
    std::tie(batch_size, _filter_number, filter_height, filter_width) = _shape;
    
    if (filter_number <= 0
            || depth <= 0
            || height <= 0
            || width <= 0
            || batch_size <= 0
            || filter_number != _filter_number
            || filter_height <= 0
            || filter_width <= 0) {
        LOG(ERROR) << "convolution calculate bp gradient failed, input parameters is empty";
        return -1;
    }

    //遍历特征图 计算每一个值
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> source_matrix_roi; 

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(input_matrix, i, j, 
                             filter_height, filter_width, 1, 
                             source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }

            for (int n = 0; n < batch_size; n++) {
                for (int f = 0; f < filter_number; f++) {
                    for (int d = 0; d < depth; d++) {
                        for (int x = 0; x < filter_height; x++) {
                            for (int y = 0; y < filter_width; y++) {
                                weights_gradient_matrix[f][d][i][j] += source_matrix_roi[n][d][x][y] * 
                                                                       sensitivity_matrix[n][f][x][y]; 
                            }
                        }
                    }
                }
            }
        }
    }

    for (int q = 0; q < sensitivity_matrix.size(); q++) {
        for (int w = 0; w < sensitivity_matrix[q].size(); w++) {
            for (int e = 0; e < sensitivity_matrix[q][w].size(); e++) {
                for (int r = 0; r < sensitivity_matrix[q][w][e].size(); r++) {
                    biases_gradient_matrix[w] += sensitivity_matrix[q][w][e][r];
                }
            }
        }
    }

    return 0;
}

//4d卷积层反向传播 计算梯度 batch_size channels height width
template <typename DataType>
int8_t Matrix<DataType>::ConvolutionalBpGradient(const std::vector<std::vector<std::vector<std::vector<double>>>>& input_matrix, 
                                                 const Matrix4d& sensitivity_matrix, 
                                                 Matrix4d& weights_gradient_matrix, 
                                                 Matrix1d& biases_gradient_matrix) {
    int filter_number;
    int depth;
    int height;
    int width;
    int batch_size;
    int _filter_number;
    int filter_height;
    int filter_width;
    
    auto shape = GetShape(weights_gradient_matrix);
    std::tie(filter_number, depth, height, width) = shape;
    auto _shape = GetShape(sensitivity_matrix);
    std::tie(batch_size, _filter_number, filter_height, filter_width) = _shape;
    
    if (filter_number <= 0
            || depth <= 0
            || height <= 0
            || width <= 0
            || batch_size <= 0
            || filter_number != _filter_number
            || filter_height <= 0
            || filter_width <= 0) {
        LOG(ERROR) << "convolution calculate bp gradient failed, input parameters is empty";
        return -1;
    }

    //遍历特征图 计算每一个值
    std::vector<std::vector<std::vector<std::vector<double>>>> source_matrix_roi; 
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            //得到roi区域
            if (-1 == GetROI(input_matrix, i, j, 
                             filter_height, filter_width, 1, 
                             source_matrix_roi)) {
                LOG(ERROR) << "matrix convolution operator failed";
                return -1;
            }

            for (int n = 0; n < batch_size; n++) {
                for (int f = 0; f < filter_number; f++) {
                    for (int d = 0; d < depth; d++) {
                        for (int x = 0; x < filter_height; x++) {
                            for (int y = 0; y < filter_width; y++) {
                                weights_gradient_matrix[f][d][i][j] += source_matrix_roi[n][d][x][y] * 
                                                                       sensitivity_matrix[n][f][x][y]; 
                            }
                        }
                    }
                }
            }
        }
    }

    for (int q = 0; q < sensitivity_matrix.size(); q++) {
        for (int w = 0; w < sensitivity_matrix[q].size(); w++) {
            for (int e = 0; e < sensitivity_matrix[q][w].size(); e++) {
                for (int r = 0; r < sensitivity_matrix[q][w][e].size(); r++) {
                    biases_gradient_matrix[w] += sensitivity_matrix[q][w][e][r];
                }
            }
        }
    }

    return 0;
}

//2d矩阵最大池化前向运算 源矩阵是大矩阵 结果矩阵是小矩阵
template <typename DataType>
int8_t Matrix<DataType>::MaxPoolingForward(const Matrix2d& source_matrix, 
                                           int32_t filter_height, 
                                           int32_t filter_width, 
                                           int32_t stride, 
                                           Matrix2d& result_matrix) {
    //check一下源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix max pooling forward failed, source matrix is wrong";
        return -1;
    }

    if (filter_height <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, input filter height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, input filter width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, input stride <= 0";
        return -1;
    }

    //getshape check一下结果矩阵
    auto shape = GetShape(result_matrix);
    int height;
    int width;
    std::tie(height, width) = shape;
    if (height <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, result matrix height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, result matrix width <= 0";
        return -1;
    }

    Matrix2d source_matrix_roi;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, 
                             stride, source_matrix_roi)) {
                LOG(ERROR) << "get matrix max pooling forward failed";
                return -1;
            }
            result_matrix[i][j] = Max(source_matrix_roi);
        }
    }

    return 0;
}

//3d矩阵最大池化前向运算 源矩阵是大矩阵 结果矩阵是小矩阵
template <typename DataType>
int8_t Matrix<DataType>::MaxPoolingForward(const Matrix3d& source_matrix, 
                                           int32_t filter_height, 
                                           int32_t filter_width, 
                                           int32_t stride, 
                                           Matrix3d& result_matrix) {
    //check一下源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix max pooling forward failed, source matrix is wrong";
        return -1;
    }

    if (filter_height <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, input filter height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, input filter width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, input stride <= 0";
        return -1;
    }

    //getshape check一下结果矩阵
    auto shape = GetShape(result_matrix);
    int depth;
    int height;
    int width;
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, result matrix depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, result matrix height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, result matrix width <= 0";
        return -1;
    }

    Matrix3d source_matrix_roi;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, 
                             stride, source_matrix_roi)) {
                LOG(ERROR) << "get matrix max pooling forward failed";
                return -1;
            }
            for (int d = 0; d < depth; d++) {
                result_matrix[d][i][j] = Max(source_matrix_roi[d]);
            }
        }
    }

    return 0;
}

//4d矩阵最大池化前向运算 源矩阵是大矩阵 结果矩阵是小矩阵
template <typename DataType>
int8_t Matrix<DataType>::MaxPoolingForward(const Matrix4d& source_matrix, 
                                           int32_t filter_height, 
                                           int32_t filter_width, 
                                           int32_t stride, 
                                           Matrix4d& result_matrix) {
    //check一下源矩阵
    if (!MatrixCheck(source_matrix, true)) {
        LOG(ERROR) << "get matrix max pooling forward failed, source matrix is wrong";
        return -1;
    }

    if (filter_height <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, input filter height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, input filter width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, input stride <= 0";
        return -1;
    }

    //getshape check一下结果矩阵
    auto shape = GetShape(result_matrix);
    int batch_size;
    int depth;
    int height;
    int width;
    std::tie(batch_size, depth, height, width) = shape;
    if (batch_size <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, result matrix batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, result matrix depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, result matrix height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix max pooling forward failed, result matrix width <= 0";
        return -1;
    }

    Matrix4d source_matrix_roi;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, 
                             stride, source_matrix_roi)) {
                LOG(ERROR) << "get matrix max pooling forward failed";
                return -1;
            }
            for (int n = 0; n < batch_size; n++) {
                for (int d = 0; d < depth; d++) {
                    result_matrix[n][d][i][j] = Max(source_matrix_roi[n][d]);
                }
            }
        }
    }

    return 0;
}


//2d矩阵最大池化反向运算 源矩阵和结果矩阵是大矩阵 sensitivity map是小矩阵
template <typename DataType>
int8_t Matrix<DataType>::MaxPoolingBackward(const Matrix2d& source_matrix, 
                                            const Matrix2d& sensitivity_matrix, 
                                            int32_t filter_height, 
                                            int32_t filter_width, 
                                            int32_t stride, 
                                            Matrix2d& result_matrix) {
    //check一下源矩阵
    if (!MatrixCheck(source_matrix, result_matrix, true)) {
        LOG(ERROR) << "get matrix max pooling backward failed, source matrix or result matrix is wrong";
        return -1;
    }

    if (filter_height <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, input filter height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, input filter width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, input stride <= 0";
        return -1;
    }

    //getshape check一下敏感图矩阵
    auto shape = GetShape(sensitivity_matrix);
    int height;
    int width;
    std::tie(height, width) = shape;
    if (height <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, result matrix height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, result matrix width <= 0";
        return -1;
    }

    Matrix2d source_matrix_roi;
    int rows = 0;
    int cols = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, 
                             stride, source_matrix_roi)) {
                LOG(ERROR) << "get matrix max pooling backward failed";
                return -1;
            }
            auto shape = GetMaxIndex(source_matrix_roi);
            if (shape == std::make_tuple(-1, -1)) {
                LOG(ERROR) << "get matrix max pooling backward failed";
                return -1;
            }
            //每个得到的索引都是固定在x * x里的i j
            //比如 2 * 2只有0 0, 0 1, 1 0, 1 1 然后加上当前行和列*步长就是对应位置
            std::tie(rows, cols) = shape;
            rows = i * stride + rows;
            cols = j * stride + cols;
            //本层sensitivity map的值 应该赋值给对应上一层sensitivity map的最大值那个索引处
            result_matrix[rows][cols] = sensitivity_matrix[i][j];
        }
    }

    return 0;
}

//3d矩阵最大池化反向运算 源矩阵和结果矩阵是大矩阵 sensitivity map是小矩阵
template <typename DataType>
int8_t Matrix<DataType>::MaxPoolingBackward(const Matrix3d& source_matrix, 
                                            const Matrix3d& sensitivity_matrix, 
                                            int32_t filter_height, 
                                            int32_t filter_width, 
                                            int32_t stride, 
                                            Matrix3d& result_matrix) {
    //check一下源矩阵
    if (!MatrixCheck(source_matrix, result_matrix, true)) {
        LOG(ERROR) << "get matrix max pooling backward failed, source matrix or result matrix is wrong";
        return -1;
    }

    if (filter_height <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, input filter height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, input filter width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, input stride <= 0";
        return -1;
    }

    //getshape check一下敏感图矩阵
    auto shape = GetShape(sensitivity_matrix);
    int depth;
    int height;
    int width;
    std::tie(depth, height, width) = shape;
    if (depth <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, result matrix depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, result matrix height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, result matrix width <= 0";
        return -1;
    }

    Matrix3d source_matrix_roi;
    int rows = 0;
    int cols = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, 
                             stride, source_matrix_roi)) {
                LOG(ERROR) << "get matrix max pooling backward failed";
                return -1;
            }
            auto max_array = GetMaxIndex(source_matrix_roi);
            //每个得到的索引都是固定在x * x里的i j
            //比如 2 * 2只有0 0, 0 1, 1 0, 1 1 然后加上当前行和列*步长就是对应位置
            for (int n = 0; n < depth; n++) {
                std::tie(rows, cols) = max_array[n];
                rows = i * stride + rows;
                cols = j * stride + cols;
                //本层sensitivity map的值 应该赋值给对应上一层sensitivity map的最大值那个索引处
                result_matrix[n][rows][cols] = sensitivity_matrix[n][i][j];
            }
        }
    }

    return 0;
}

//4d矩阵最大池化反向运算 源矩阵和结果矩阵是大矩阵 sensitivity map是小矩阵
template <typename DataType>
int8_t Matrix<DataType>::MaxPoolingBackward(const Matrix4d& source_matrix, 
                                            const Matrix4d& sensitivity_matrix, 
                                            int32_t filter_height, 
                                            int32_t filter_width, 
                                            int32_t stride, 
                                            Matrix4d& result_matrix) {
    //check一下源矩阵
    if (!MatrixCheck(source_matrix, result_matrix, true)) {
        LOG(ERROR) << "get matrix max pooling backward failed, source matrix or result matrix is wrong";
        return -1;
    }

    if (filter_height <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, input filter height <= 0";
        return -1;
    }
    if (filter_width <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, input filter width <= 0";
        return -1;
    }
    if (stride <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, input stride <= 0";
        return -1;
    }

    //getshape check一下敏感图矩阵
    auto shape = GetShape(sensitivity_matrix);
    int batch_size;
    int depth;
    int height;
    int width;
    std::tie(batch_size, depth, height, width) = shape;
    if (batch_size <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, result matrix batch size <= 0";
        return -1;
    }
    if (depth <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, result matrix depth <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, result matrix height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get matrix max pooling backward failed, result matrix width <= 0";
        return -1;
    }

    Matrix4d source_matrix_roi;
    int rows = 0;
    int cols = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (-1 == GetROI(source_matrix, i, j, 
                             filter_height, filter_width, 
                             stride, source_matrix_roi)) {
                LOG(ERROR) << "get matrix max pooling backward failed";
                return -1;
            }
            for (int n = 0; n < batch_size; n++) {
                auto max_array = GetMaxIndex(source_matrix_roi[n]);
                //每个得到的索引都是固定在x * x里的i j
                //比如 2 * 2只有0 0, 0 1, 1 0, 1 1 然后加上当前行和列*步长就是对应位置
                for (int d = 0; d < depth; d++) {
                    std::tie(rows, cols) = max_array[d];
                    rows = i * stride + rows;
                    cols = j * stride + cols;
                    //本层sensitivity map的值 应该赋值给对应上一层sensitivity map的最大值那个索引处
                    result_matrix[n][d][rows][cols] = sensitivity_matrix[n][d][i][j];
                }
            }
        }
    }

    return 0;
}



//全连接层前向计算 a = f(w .* x + b)
template <typename DataType>
int8_t Matrix<DataType>::FullConnectedForward(const Matrix2d& source_matrix, 
                                              const Matrix2d& weights_matrix, 
                                              const Matrix2d& biases_matrix, 
                                              Matrix2d& result_matrix) { 
    if (!MatrixCheck(weights_matrix, true)) {
        LOG(ERROR) << "matrix full connected forward";
    }
    if (!MatrixCheck(biases_matrix, true)) {
        LOG(ERROR) << "matrix full connected forward";
    }

    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(weights_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix full connected forward";
        return -1;
    }

    auto right_matrix_shape = GetShape(source_matrix);
    if (right_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix full connected forward";
        return -1;
    }

    //元组解包
    std::tie(left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix full connected forward";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    result_matrix.clear();
    result_matrix = Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始点积运算  
        for (int i = 0; i < left_matrix_rows; i++) {
            for (int j = 0; j < right_matrix_cols; j++) {
                for (int k = 0; k < left_matrix_cols; k++) {
                    result_matrix[i][j] += weights_matrix[i][k] * source_matrix[k][j];
                }
                result_matrix[i][j] += biases_matrix[i][j];
                result_matrix[i][j] = 1.0 / (1.0 + exp(-result_matrix[i][j]));   //sigmoid
                //result_matrix[i][j] = result_matrix[i][j] > 0.0 ? result_matrix[i][j] : 0.0;  //relu
            }
        }
    }

    return 0;
    
}

//全连接层前向计算 a = f(w .* x + b)
template <typename DataType>
int8_t Matrix<DataType>::FullConnectedForward(const Matrix3d& source_matrix, 
                                              const Matrix2d& weights_matrix, 
                                              const Matrix2d& biases_matrix, 
                                              Matrix3d& result_matrix, 
                                              bool is_output_layer) { 
    if (!MatrixCheck(weights_matrix, true)) {
        LOG(ERROR) << "matrix full connected forward";
    }
    if (!MatrixCheck(biases_matrix, true)) {
        LOG(ERROR) << "matrix full connected forward";
    }

    int batch_size = 0;
    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(weights_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix full connected forward";
        return -1;
    }

    auto right_matrix_shape = GetShape(source_matrix);
    if (right_matrix_shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "matrix full connected forward";
        return -1;
    }

    //元组解包
    std::tie(left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(batch_size, right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix full connected forward";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    result_matrix.clear();
    result_matrix = Matrix3d(batch_size, Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols)));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始点积运算  
        for (int n = 0; n < batch_size; n++) {
            for (int i = 0; i < left_matrix_rows; i++) {
                for (int j = 0; j < right_matrix_cols; j++) {
                    for (int k = 0; k < left_matrix_cols; k++) {
                        result_matrix[n][i][j] += weights_matrix[i][k] * source_matrix[n][k][j];
                    }
                    result_matrix[n][i][j] += biases_matrix[i][j];
                    //如果不是输出层 就经过激活函数 是输出层就不经过relu 后面用softmax得到概率
                    if (!is_output_layer) {
                        result_matrix[n][i][j] = result_matrix[n][i][j] > 0.0 ? result_matrix[n][i][j] : 0.0;  //relu
                        //result_matrix[n][i][j] = 1.0 / (1.0 + exp(-result_matrix[n][i][j])); //sigmoid
                    }
                }
            }
        }
    }

    return 0;
    
}

/* 全连接层反向计算
 * 本层的误差项 = x * (1 - x) * WT .* delta_array
 * w权重的梯度 就是 delta_array .* xT  下一层的误差项 点积 本层节点值的转置矩阵
 * b偏置的梯度 就是 delta_array
 */
template <typename DataType>
int8_t Matrix<DataType>::FullConnectedBackward(const Matrix2d& source_matrix, 
                                               const Matrix2d& weights_matrix, 
                                               const Matrix2d& output_delta_matrix,
                                               Matrix2d& delta_matrix, 
                                               Matrix2d& weights_gradient_matrix, 
                                               Matrix2d& biases_gradient_matrix) { 
    if (0 == source_matrix.size()
            || 0 == weights_matrix.size()
            || 0 == output_delta_matrix.size()
            || 0 == weights_gradient_matrix.size()) {
        LOG(ERROR) << "matrix full connected backward, input source matrix is empty";
        return -1;
    }

    Matrix2d weights_transpose_matrix;
    Transpose(weights_matrix, weights_transpose_matrix);
    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(weights_transpose_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix full connected backward";
        return -1;
    }

    auto right_matrix_shape = GetShape(output_delta_matrix);
    if (right_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix full connected backward";
        return -1;
    }

    //元组解包
    std::tie(left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix full connected backward";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    delta_matrix.clear();
    delta_matrix = Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始点积运算  
        for (int i = 0; i < left_matrix_rows; i++) {
            for (int j = 0; j < right_matrix_cols; j++) {
                for (int k = 0; k < left_matrix_cols; k++) {
                    delta_matrix[i][j] += weights_transpose_matrix[i][k] * output_delta_matrix[k][j];
                }
                delta_matrix[i][j] *= source_matrix[i][j] * (1.0 - source_matrix[i][j]);
            }
        }
    }

    Matrix2d source_transpose_matrix;
    Transpose(source_matrix, source_transpose_matrix);
    if (-1 == DotProduct(output_delta_matrix, source_transpose_matrix, weights_gradient_matrix)) {
        LOG(ERROR) << "matrix full connected backward";
        return -1;
    }
    biases_gradient_matrix = output_delta_matrix;

    return 0;
}

/* 全连接层反向计算
 * 本层的误差项 = WT .* delta_array
 * w权重的梯度 就是 delta_array .* xT  下一层的误差项 点积 本层节点值的转置矩阵
 * b偏置的梯度 就是 delta_array
 */
template <typename DataType>
int8_t Matrix<DataType>::FullConnectedBackward(const Matrix3d& source_matrix, 
                                               const Matrix2d& weights_matrix, 
                                               const Matrix3d& output_delta_matrix,
                                               Matrix3d& delta_matrix, 
                                               Matrix2d& weights_gradient_matrix, 
                                               Matrix2d& biases_gradient_matrix) { 
    if (0 == source_matrix.size()
            || 0 == weights_matrix.size()
            || 0 == output_delta_matrix.size()) {
        LOG(ERROR) << "matrix full connected backward, input source matrix is empty";
        return -1;
    }

    Matrix2d weights_transpose_matrix;
    Transpose(weights_matrix, weights_transpose_matrix);
    int batch_size;
    int left_matrix_rows = 0;    //左矩阵的行
    int left_matrix_cols = 0;    //左矩阵的列
    int right_matrix_rows = 0;   //右矩阵的行
    int right_matrix_cols = 0;   //右矩阵的列

    //getshape里已经check单个矩阵的正确性 这里不用check 
    auto left_matrix_shape = GetShape(weights_transpose_matrix);
    if (left_matrix_shape == std::make_tuple(0, 0)) {
        LOG(ERROR) << "matrix full connected backward";
        return -1;
    }

    auto right_matrix_shape = GetShape(output_delta_matrix);
    if (right_matrix_shape == std::make_tuple(0, 0, 0)) {
        LOG(ERROR) << "matrix full connected backward";
        return -1;
    }

    //元组解包
    std::tie(left_matrix_rows, left_matrix_cols) = left_matrix_shape;
    std::tie(batch_size, right_matrix_rows, right_matrix_cols) = right_matrix_shape;

    //判断左矩阵的列和右矩阵的行是否相等
    if (left_matrix_cols != right_matrix_rows) {
        LOG(ERROR) << "matrix full connected backward";
        return -1;
    }
    
    //这里不做判断直接清零 和赋值 因为下面是+= 要是之前有值会出错 
    delta_matrix.clear();
    delta_matrix = Matrix3d(batch_size, Matrix2d(left_matrix_rows, Matrix1d(right_matrix_cols)));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //误差传递  
        for (int n = 0; n < batch_size; n++) {
            for (int i = 0; i < left_matrix_rows; i++) {
                for (int j = 0; j < right_matrix_cols; j++) {
                    for (int k = 0; k < left_matrix_cols; k++) {
                        delta_matrix[n][i][j] += weights_transpose_matrix[i][k] * output_delta_matrix[n][k][j];
                    }
                    //delta_matrix[n][i][j] *= source_matrix[n][i][j] * (1.0 - source_matrix[n][i][j]);
                }
            }
        }
    }

    //计算梯度
    Matrix3d source_transpose_matrix;
    Transpose(source_matrix, source_transpose_matrix);
    if (-1 == DotProduct(output_delta_matrix, source_transpose_matrix, weights_gradient_matrix)) {
        LOG(ERROR) << "matrix full connected backward";
        return -1;
    }
    
    CreateZeros(GetShape(output_delta_matrix[0]), biases_gradient_matrix); 
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int n = 0; n < output_delta_matrix.size(); n++) {
            for (int i = 0; i < output_delta_matrix[n].size(); i++) {
                for (int j = 0; j < output_delta_matrix[n][i].size(); j++) {
                    biases_gradient_matrix[i][j] += output_delta_matrix[n][i][j];
                }
            }
        }
    }

    return 0;
}

    

//计算神经网络的输出层误差项
template <typename DataType>
int8_t Matrix<DataType>::CalcOutputDiff(const Matrix3d& output_array, 
                                        const Matrix3d& label, 
                                        Matrix3d& delta_array) {
    //check source matrix
    if (!MatrixCheck(output_array, label, true)) {
        LOG(ERROR) << "calc diff failed";
        return -1;
    }
    
    if (!MatrixCheck(output_array, delta_array, false)) {
        delta_array = output_array;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < output_array.size(); i++) {
            for (int j = 0; j < output_array[i].size(); j++) {
                for (int k = 0; k < output_array[i][j].size(); k++) {
                    delta_array[i][j][k] = output_array[i][j][k] * 
                                           (1.0 - output_array[i][j][k]) *           //sigmoid导数
                                           (label[i][j][k] - output_array[i][j][k]);  
                    //delta_array[i][j][k] = output_array[i][j][k] > 0 ?                 //relu导数
                    //                       (label[i][j][k] - output_array[i][j][k]) : 0.0;
                }
            }
        }
    }
    
    return 0;
}

//计算神经网络的输出层误差项
template <typename DataType>
int8_t Matrix<DataType>::CalcOutputDiff(const Matrix2d& output_array, 
                                        const Matrix2d& label, 
                                        Matrix2d& delta_array) {
    //check source matrix
    if (!MatrixCheck(output_array, label, true)) {
        LOG(ERROR) << "calc diff failed";
        return -1;
    }
    
    if (!MatrixCheck(output_array, delta_array, false)) {
        delta_array = output_array;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < output_array.size(); i++) {
            for (int j = 0; j < output_array[i].size(); j++) {
                delta_array[i][j] = output_array[i][j] * 
                                    (1.0 - output_array[i][j]) *
                                    (label[i][j] - output_array[i][j]);
            }
        }
    }
    
    return 0;
}


//2d全连接层的梯度下降优化算法
template <typename DataType>
int8_t Matrix<DataType>::GradientDescent(const Matrix2d& weights_gradient_array, 
                                         const Matrix2d& biases_gradient_array,
                                         double learning_rate, 
                                         int32_t batch_size, 
                                         Matrix2d& weights_array, 
                                         Matrix2d& biases_array) {
    //check source matrix and result
    if (0 == weights_gradient_array.size() 
            || 0 == biases_gradient_array.size()
            || 0 == weights_array.size()
            || 0 == biases_array.size()
            || learning_rate <= 0.0
            || batch_size <= 0) {
        LOG(ERROR) << "gradient descent failed, input source matrix is empty";
        return -1;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < weights_gradient_array.size(); i++) {
            for (int j = 0; j < weights_gradient_array[i].size(); j++) {
                if (isnan(weights_array[i][j] 
                            || isinf(weights_array[i][j]))) {
                    weights_array[i][j] = 0.0;
                }
                weights_array[i][j] += weights_gradient_array[i][j] / batch_size * learning_rate; 
                if (isnan(weights_array[i][j] 
                            || isinf(weights_array[i][j]))) {
                    weights_array[i][j] = 0.0;
                }
            }
        }

        #pragma omp for schedule(static) 
        for (int i = 0; i < biases_gradient_array.size(); i++) {
            for (int j = 0; j < biases_gradient_array[i].size(); j++) {
                if (isnan(biases_array[i][j] 
                            || isinf(biases_array[i][j]))) {
                    biases_array[i][j] = 0.0;
                }
                biases_array[i][j] += biases_gradient_array[i][j] / batch_size * learning_rate; 
                if (isnan(biases_array[i][j] 
                            || isinf(biases_array[i][j]))) {
                    biases_array[i][j] = 0.0;
                }
            }
        }
    }

    return 0;
}


//3d卷积层的梯度下降优化算法
template <typename DataType>
int8_t Matrix<DataType>::GradientDescent(const Matrix3d& weights_gradient_array, 
                                         const double biases_gradient,
                                         double learning_rate, 
                                         int32_t batch_size, 
                                         Matrix3d& weights_array, 
                                         double& biases) {
    //check source matrix and result
    if (0 == weights_gradient_array.size() 
            || 0 == weights_array.size()
            || learning_rate <= 0.0
            || batch_size <= 0) {
        LOG(ERROR) << "gradient descent failed, input source matrix is empty";
        return -1;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < weights_gradient_array.size(); i++) {
            for (int j = 0; j < weights_gradient_array[i].size(); j++) {
                for (int k = 0; k < weights_gradient_array[i][j].size(); k++) {
                    if (isnan(weights_array[i][j][k] 
                            || isinf(weights_array[i][j][k]))) {
                        weights_array[i][j][k] = 0.0;
                    }
                    weights_array[i][j][k] -= weights_gradient_array[i][j][k] / batch_size * learning_rate; 
                    if (isnan(weights_array[i][j][k] 
                            || isinf(weights_array[i][j][k]))) {
                        weights_array[i][j][k] = 0.0;
                    }
                }
            }
        }
    }

    if (isnan(biases)
            || isinf(biases)) {
        biases = 0.0;
    }
    biases -= biases_gradient / batch_size * learning_rate;
    if (isnan(biases)
            || isinf(biases)) {
        biases = 0.0;
    }

    return 0;
}

//4d卷积层的梯度下降优化算法
template <typename DataType>
int8_t Matrix<DataType>::GradientDescent(const Matrix4d& weights_gradient_array, 
                                         const Matrix1d& biases_gradient_array,
                                         double learning_rate, 
                                         int32_t batch_size, 
                                         Matrix4d& weights_array, 
                                         Matrix1d& biases_array) {
    //check source matrix and result
    if (0 == weights_gradient_array.size() 
            || 0 == weights_array.size()
            || 0 == biases_gradient_array.size()
            || 0 == biases_array.size()
            || learning_rate <= 0.0
            || batch_size <= 0) {
        LOG(ERROR) << "gradient descent failed, input source matrix is empty";
        return -1;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < weights_array.size(); i++) {
            for (int j = 0; j < weights_array[i].size(); j++) {
                for (int k = 0; k < weights_array[i][j].size(); k++) {
                    for (int z = 0; z < weights_array[i][j][k].size(); z++) {
                        if (isnan(weights_array[i][j][k][z] 
                                || isinf(weights_array[i][j][k][z]))) {
                            weights_array[i][j][k][z] = 0.0;
                        }
                        weights_array[i][j][k][z] -= weights_gradient_array[i][j][k][z] / batch_size * learning_rate; 
                        if (isnan(weights_array[i][j][k][z] 
                                || isinf(weights_array[i][j][k][z]))) {
                            weights_array[i][j][k][z] = 0.0;
                        }
                    }
                }
            }
            if (isnan(biases_array[i])
                    || isinf(biases_array[i])) {
                biases_array[i] = 0.0;
            }
            biases_array[i] -= biases_gradient_array[i] / batch_size * learning_rate;
            if (isnan(biases_array[i])
                    || isinf(biases_array[i])) {
                biases_array[i] = 0.0;
            }
        }
    }

    return 0;
}

//2d全连接层的梯度下降优化算法
template <typename DataType>
int8_t Matrix<DataType>::SGDMomentum(const Matrix2d& weights_gradient_array, 
                                     const Matrix2d& biases_gradient_array,
                                     Matrix2d& last_weights_gradient_array, 
                                     Matrix2d& last_biases_gradient_array, 
                                     double learning_rate, 
                                     int32_t batch_size, 
                                     double momentum, 
                                     Matrix2d& weights_array, 
                                     Matrix2d& biases_array) {
    //check source matrix and result
    if (0 == weights_gradient_array.size() 
            || 0 == biases_gradient_array.size()
            || 0 == weights_array.size()
            || 0 == biases_array.size()
            || learning_rate <= 0.0
            || batch_size <= 0
            || momentum <= 0.0) {
        LOG(ERROR) << "gradient descent failed, input source matrix is empty";
        return -1;
    }

    if (0 == last_weights_gradient_array.size()) {
        last_weights_gradient_array = Matrix2d(weights_array.size(), 
                                               Matrix1d(weights_array[0].size()));
    }
    if (0 == last_biases_gradient_array.size()) {
        last_biases_gradient_array = Matrix2d(biases_array.size(), 
                                              Matrix1d(biases_array[0].size()));
    }

    Matrix2d _last_weights_gradient_array = Matrix2d(weights_array.size(), 
                                                     Matrix1d(weights_array[0].size()));
    Matrix2d _last_biases_gradient_array = Matrix2d(biases_array.size(), 
                                                    Matrix1d(biases_array[0].size()));

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < weights_gradient_array.size(); i++) {
            for (int j = 0; j < weights_gradient_array[i].size(); j++) {
                if (isnan(weights_array[i][j] 
                            || isinf(weights_array[i][j]))) {
                    weights_array[i][j] = 0.0;
                }
                
                //velocity 速度 保持上次梯度的方向 减去这次梯度的方向
                _last_weights_gradient_array[i][j] = (momentum * last_weights_gradient_array[i][j]) -
                                                     (learning_rate * weights_gradient_array[i][j]);
                weights_array[i][j] += _last_weights_gradient_array[i][j];
                //weights_array[i][j] -= learning_rate * weights_gradient_array[i][j];

                if (isnan(weights_array[i][j] 
                            || isinf(weights_array[i][j]))) {
                    weights_array[i][j] = 0.0;
                }
            }
        }

        #pragma omp for schedule(static) 
        for (int i = 0; i < biases_gradient_array.size(); i++) {
            for (int j = 0; j < biases_gradient_array[i].size(); j++) {
                if (isnan(biases_array[i][j] 
                            || isinf(biases_array[i][j]))) {
                    biases_array[i][j] = 0.0;
                }

                //velocity 速度 保持上次梯度的方向 减去这次梯度的方向11
                _last_biases_gradient_array[i][j] = (momentum * last_biases_gradient_array[i][j]) -
                                                    (learning_rate * biases_gradient_array[i][j]);
                biases_array[i][j] += _last_biases_gradient_array[i][j];
                //biases_array[i][j] -= learning_rate * biases_gradient_array[i][j];

                if (isnan(biases_array[i][j] 
                            || isinf(biases_array[i][j]))) {
                    biases_array[i][j] = 0.0;
                }
            }
        }
    }

    //保存本次梯度
    last_weights_gradient_array = _last_weights_gradient_array;
    last_biases_gradient_array = _last_biases_gradient_array;

    return 0;
}

//4d卷积层的梯度下降优化算法
template <typename DataType>
int8_t Matrix<DataType>::SGDMomentum(const Matrix4d& weights_gradient_array, 
                                     const Matrix1d& biases_gradient_array,
                                     Matrix4d& last_weights_gradient_array, 
                                     Matrix1d& last_biases_gradient_array, 
                                     double learning_rate, 
                                     int32_t batch_size, 
                                     double momentum, 
                                     Matrix4d& weights_array, 
                                     Matrix1d& biases_array) {
    //check source matrix and result
    if (0 == weights_gradient_array.size() 
            || 0 == weights_array.size()
            || 0 == biases_gradient_array.size()
            || 0 == biases_array.size()
            || learning_rate <= 0.0
            || batch_size <= 0
            || momentum <= 0.0) {
        LOG(ERROR) << "gradient descent failed, input source matrix is empty";
        return -1;
    }

    if (0 == last_weights_gradient_array.size()) {
        CreateZeros(GetShape(weights_array), last_weights_gradient_array);
    }
    if (0 == last_biases_gradient_array.size()) {
        last_biases_gradient_array = Matrix1d(biases_array.size());
    }

    Matrix4d _last_weights_gradient_array = last_weights_gradient_array; 
    Matrix1d _last_biases_gradient_array = last_biases_gradient_array;

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        for (int i = 0; i < weights_array.size(); i++) {
            for (int j = 0; j < weights_array[i].size(); j++) {
                for (int k = 0; k < weights_array[i][j].size(); k++) {
                    for (int z = 0; z < weights_array[i][j][k].size(); z++) {
                        if (isnan(weights_array[i][j][k][z] 
                                || isinf(weights_array[i][j][k][z]))) {
                            weights_array[i][j][k][z] = 0.0;
                        }

                        //velocity 速度 保持上次梯度的方向 减去这次梯度的方向
                        _last_weights_gradient_array[i][j][k][z] = (momentum * last_weights_gradient_array[i][j][k][z]) -
                                                                   (learning_rate * weights_gradient_array[i][j][k][z]);
                        weights_array[i][j][k][z] += _last_weights_gradient_array[i][j][k][z];
                        //weights_array[i][j][k][z] -= learning_rate * weights_gradient_array[i][j][k][z];

                        if (isnan(weights_array[i][j][k][z] 
                                || isinf(weights_array[i][j][k][z]))) {
                            weights_array[i][j][k][z] = 0.0;
                        }
                    }
                }
            }
            if (isnan(biases_array[i])
                    || isinf(biases_array[i])) {
                biases_array[i] = 0.0;
            }

            //velocity 速度 保持上次梯度的方向 减去这次梯度的方向
            _last_biases_gradient_array[i] = (momentum * last_biases_gradient_array[i]) -
                                             (learning_rate * biases_gradient_array[i]);
            biases_array[i] += _last_biases_gradient_array[i];
            //biases_array[i] -= learning_rate * biases_gradient_array[i];

            if (isnan(biases_array[i])
                    || isinf(biases_array[i])) {
                biases_array[i] = 0.0;
            }
        }
    }

    //保存本次梯度
    last_weights_gradient_array = _last_weights_gradient_array;
    last_biases_gradient_array = _last_biases_gradient_array;

    return 0;
}

//评估 得到精度(准确率)
template <typename DataType>
double Matrix<DataType>::Evaluate(const Matrix3d& output_array, 
                                  const Matrix3d& test_label_data_set) {
    double correct = 0.0;
    std::vector<int> ground_truth;
    std::vector<int> predict_output;

    //得到标签
    ground_truth = ArgMax(test_label_data_set);
    predict_output = ArgMax(output_array);
    if (predict_output.size() != ground_truth.size()) {
        LOG(ERROR) << "evaluate failed";
        return -1;
    }

    //遍历测试数据集 做预测 查看结果 
    for (int i = 0; i < predict_output.size(); i++) {
        //得到预测结果
        if (predict_output[i] == ground_truth[i]) {
            correct += 1.0;
        }
    }
    
    return correct / predict_output.size() * 100.0;
}

//得到最大值
template <typename DataType>
int32_t Matrix<DataType>::ArgMax(const Matrix2d& output_array) {
    double max_value = 0.0;
    int max_value_index = 0;

    for (int i = 0; i < output_array.size(); i++) {
        for (int j = 0; j < output_array[i].size(); j++) {
            if (output_array[i][j] > max_value) {
                max_value = output_array[i][j];
                max_value_index = i;
            }
        }
    }
    
    return max_value_index;
}

//得到最大值
template <typename DataType>
std::vector<int> Matrix<DataType>::ArgMax(const Matrix3d& output_array) {
    std::vector<int> result_array;
    result_array.reserve(output_array.size());
    for (int i = 0; i < output_array.size(); i++) {
        double max_value = 0.0; 
        int max_value_index = 0;

        for (int j = 0; j < output_array[i].size(); j++) {
            for (int k = 0; k < output_array[i][j].size(); k++) {
                if (output_array[i][j][k] > max_value) {
                    max_value = output_array[i][j][k];
                    max_value_index = j;
                }
            }
        }
        result_array.push_back(max_value_index);
    }
    
    return result_array;
}


















}       //namespace matrix




    



namespace random {

//模板类 随机数对象
template <typename DataType=double>
struct Random { 
    //类型别名
    typedef std::vector<DataType> Matrix1d;
    typedef std::vector<std::vector<DataType>> Matrix2d;
    typedef std::vector<std::vector<std::vector<DataType>>> Matrix3d;
    typedef std::vector<std::vector<std::vector<std::vector<DataType>>>> Matrix4d;

    //生成服从正态分布的随机数二维矩阵
    static int8_t Normal(float mean, float stddev, int32_t rows, int32_t cols, 
                         Matrix2d& random_matrix);

    //生成服从均匀分布的随机浮点数二维矩阵
    static int8_t Uniform(float a, float b, int32_t rows, int32_t cols, 
                          Matrix2d& random_matrix);

    //生成服从均匀分布的随机浮点数三维矩阵
    static int8_t Uniform(float a, float b, int32_t channel_number, 
                          int32_t height, int32_t width, 
                          Matrix3d& random_matrix);

    //生成服从均匀分布的随机浮点数四维矩阵
    static int8_t Uniform(float a, float b, int32_t batch_size, 
                          int32_t channel_number, 
                          int32_t height, int32_t width, 
                          Matrix4d& random_matrix);

    //生成服从均匀分布的随机整数二维矩阵
    static int8_t Randint(float a, float b, int32_t rows, int32_t cols, 
                          Matrix2d& random_matrix);

    //生成服从均匀分布的随机整数三维矩阵
    static int8_t Randint(float a, float b, int32_t channel_number,
                          int32_t height, int32_t width, 
                          Matrix3d& random_matrix);

    //生成服从均匀分布的随机整数四维矩阵
    static int8_t Randint(float a, float b, int32_t batch_size, 
                          int32_t channel_number, 
                          int32_t height, int32_t width, 
                          Matrix4d& random_matrix);

    //生成服从二项分布的随机数二维矩阵 
    static int8_t Binomial(int32_t experiment_times, float probability, 
                           int32_t rows, int32_t cols, 
                           Matrix2d& random_matrix);

    //生成服从二项分布的随机数二维矩阵 
    static int8_t Binomial(int32_t experiment_times, float probability, 
                           std::tuple<int32_t, int32_t> shape, 
                           Matrix2d& random_matrix);

    //生成服从二项分布的随机数三维矩阵
    static int8_t Binomial(int32_t experiment_times, float probability,
                           int32_t channel_number, int32_t height, int32_t width, 
                           Matrix3d& random_matrix);

    //生成服从二项分布的随机数三维矩阵
    static int8_t Binomial(int32_t experiment_times, float probability,
                           std::tuple<int32_t, int32_t, int32_t> shape, 
                           Matrix3d& random_matrix);

    //全连接层的隐藏层 训练时使用dropout 防止过拟合
    //dropout过程相当于对很多不同的神经网络取平均 不同网络产生不同的过拟合 相互抵消
    static int8_t DropOut(const Matrix2d& source_matrix, 
                          int32_t experiment_times, float probability, 
                          Matrix2d& binomial_matrix, 
                          Matrix2d& result_matrix);

    //全连接层的隐藏层 训练时使用dropout 防止过拟合
    //dropout过程相当于对很多不同的神经网络取平均 不同网络产生不同的过拟合 相互抵消
    static int8_t DropOut(const Matrix3d& source_matrix, 
                          int32_t experiment_times, float probability, 
                          Matrix2d& binomial_matrix, 
                          Matrix3d& result_matrix);

    //全连接层的隐藏层 训练时使用dropout 防止过拟合
    //dropout过程相当于对很多不同的神经网络取平均 不同网络产生不同的过拟合 相互抵消
    static int8_t DropOut(const Matrix3d& delta_matrix, 
                          const Matrix2d& binomial_matrix,
                          const Matrix3d& input_matrix, 
                          double rescale, 
                          Matrix3d& result_matrix);
    


};   //struct Random


//生成服从正态分布的随机数二维矩阵
template <typename DataType>
int8_t Random<DataType>::Normal(float mean, float stddev, int32_t rows, int32_t cols, 
                                Matrix2d& random_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "get normal distribution matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get normal distribution matrix failed, input cols <= 0";
        return -1;
    }

    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix.reserve(rows);
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::normal_distribution<double> generate_random(mean, stddev);
    for (int i = 0; i < rows; i++) {
        std::vector<DataType> random_array;
        random_array.reserve(cols);
        for (int j = 0; j < cols; j++) {
            random_array.push_back(generate_random(random_engine));
        }
        random_matrix.push_back(random_array);
    }

    return 0;
}


//生成服从均匀分布的随机浮点数二维矩阵
template <typename DataType>
int8_t Random<DataType>::Uniform(float a, float b, int32_t rows, int32_t cols, 
                                 Matrix2d& random_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input cols <= 0";
        return -1;
    }
    if (b < a) {
        LOG(ERROR) << "get uniform distribution matrix failed, max value < min value";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix.reserve(rows);
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::uniform_real_distribution<double> generate_random(a, b);
    for (int i = 0; i < rows; i++) {
        std::vector<DataType> random_array;
        random_array.reserve(cols);
        for (int j = 0; j < cols; j++) {
            random_array.push_back(generate_random(random_engine));
        }
        random_matrix.push_back(random_array);
    }

    return 0;
}

//生成服从均匀分布的随机浮点数三维矩阵
template <typename DataType>
int8_t Random<DataType>::Uniform(float a, float b, int32_t channel_number, 
                                 int32_t height, int32_t width, 
                                 Matrix3d& random_matrix) {
                
    if (channel_number <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input width <= 0";
        return -1;
    }
    if (b < a) {
        LOG(ERROR) << "get uniform distribution matrix failed, max value < min value";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }
    random_matrix = Matrix3d(channel_number, Matrix2d(height, Matrix1d(width)));
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::uniform_real_distribution<double> generate_random(a, b);
    for (int i = 0; i < channel_number; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                random_matrix[i][j][k] = generate_random(random_engine);
            }
        }
    }

    return 0;
}

//生成服从均匀分布的随机浮点数四维矩阵
template <typename DataType>
int8_t Random<DataType>::Uniform(float a, float b, int32_t batch_size, 
                                 int32_t channel_number, 
                                 int32_t height, int32_t width, 
                                 Matrix4d& random_matrix) {
                
    if (batch_size <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input batch size <= 0";
        return -1;
    }
    if (channel_number <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input width <= 0";
        return -1;
    }
    if (b < a) {
        LOG(ERROR) << "get uniform distribution matrix failed, max value < min value";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }
    random_matrix = Matrix4d(batch_size, Matrix3d(channel_number, Matrix2d(height, Matrix1d(width))));
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::uniform_real_distribution<double> generate_random(a, b);
    for (int n = 0; n < batch_size; n++) {
        for (int i = 0; i < channel_number; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    random_matrix[n][i][j][k] = generate_random(random_engine);
                }
            }
        }
    }

    return 0;
}

//生成服从均匀分布的随机整数二维矩阵
template <typename DataType>
int8_t Random<DataType>::Randint(float a, float b, int32_t rows, int32_t cols,  
                                 Matrix2d& random_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input cols <= 0";
        return -1;
    }
    if (b < a) {
        LOG(ERROR) << "get uniform distribution matrix failed, max value < min value";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix.reserve(rows);
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::uniform_int_distribution<int> generate_random(a, b);
    for (int i = 0; i < rows; i++) {
        std::vector<DataType> random_array;
        random_array.reserve(cols);
        for (int j = 0; j < cols; j++) {
            random_array.push_back(generate_random(random_engine));
        }
        random_matrix.push_back(random_array);
    }

    return 0;
}

//生成服从均匀分布的随机整数三维矩阵
template <typename DataType>
int8_t Random<DataType>::Randint(float a, float b, int32_t channel_number, 
                                 int32_t height, int32_t width, 
                                 Matrix3d& random_matrix) {
                
    if (channel_number <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input width <= 0";
        return -1;
    }
    if (b < a) {
        LOG(ERROR) << "get uniform distribution matrix failed, max value < min value";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }
    random_matrix = Matrix3d(channel_number, Matrix2d(height, Matrix1d(width)));
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::uniform_int_distribution<int> generate_random(a, b);
    for (int i = 0; i < channel_number; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                random_matrix[i][j][k] = generate_random(random_engine);
            }
        }
    }

    return 0;
}

//生成服从均匀分布的随机浮点数四维矩阵
template <typename DataType>
int8_t Random<DataType>::Randint(float a, float b, int32_t batch_size, 
                                 int32_t channel_number, 
                                 int32_t height, int32_t width, 
                                 Matrix4d& random_matrix) {
                
    if (batch_size <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input batch size <= 0";
        return -1;
    }
    if (channel_number <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get uniform distribution matrix failed, input width <= 0";
        return -1;
    }
    if (b < a) {
        LOG(ERROR) << "get uniform distribution matrix failed, max value < min value";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }
    random_matrix = Matrix4d(batch_size, Matrix3d(channel_number, Matrix2d(height, Matrix1d(width))));
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::uniform_int_distribution<int> generate_random(a, b);
    for (int n = 0; n < batch_size; n++) {
        for (int i = 0; i < channel_number; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    random_matrix[n][i][j][k] = generate_random(random_engine);
                }
            }
        }
    }

    return 0;
}


//生成服从二项分布的随机数二维矩阵 
template <typename DataType>
int8_t Random<DataType>::Binomial(int32_t experiment_times, float probability, 
                                  int32_t rows, int32_t cols, 
                                  Matrix2d& random_matrix) {
    if (rows <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input cols <= 0";
        return -1;
    }
    if (experiment_times <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input experiment times <= 0";
        return -1;
    }
    if (probability <= 0
            || probability >= 1) {
        LOG(ERROR) << "get binomial distribution matrix failed, input probability <= 0 or >= 1";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix.reserve(rows);
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::binomial_distribution<int> generate_random(experiment_times, probability);
    for (int i = 0; i < rows; i++) {
        std::vector<DataType> random_array;
        random_array.reserve(cols);
        for (int j = 0; j < cols; j++) {
            random_array.push_back(generate_random(random_engine));
        }
        random_matrix.push_back(random_array);
    }

    return 0;
}

//生成服从二项分布的随机数二维矩阵 
template <typename DataType>
int8_t Random<DataType>::Binomial(int32_t experiment_times, float probability, 
                                  std::tuple<int32_t, int32_t> shape,  
                                  Matrix2d& random_matrix) {
    int rows;
    int cols;
    std::tie(rows, cols) = shape;
    if (rows <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input rows <= 0";
        return -1;
    }
    if (cols <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input cols <= 0";
        return -1;
    }
    if (experiment_times <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input experiment times <= 0";
        return -1;
    }
    if (probability <= 0
            || probability >= 1) {
        LOG(ERROR) << "get binomial distribution matrix failed, input probability <= 0 or >= 1";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix.reserve(rows);
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::binomial_distribution<int> generate_random(experiment_times, probability);
    for (int i = 0; i < rows; i++) {
        std::vector<DataType> random_array;
        random_array.reserve(cols);
        for (int j = 0; j < cols; j++) {
            random_array.push_back(generate_random(random_engine));
        }
        random_matrix.push_back(random_array);
    }

    return 0;
}

//生成服从二项分布的随机数三维矩阵 
template <typename DataType>
int8_t Random<DataType>::Binomial(int32_t experiment_times, float probability, 
                                  int32_t channel_number, int32_t height, int32_t width,  
                                  Matrix3d& random_matrix) {
    if (channel_number <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input width <= 0";
        return -1;
    }
    if (experiment_times <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input experiment times <= 0";
        return -1;
    }
    if (probability <= 0
            || probability >= 1) {
        LOG(ERROR) << "get binomial distribution matrix failed, input probability <= 0 or >= 1";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix = Matrix3d(channel_number, Matrix2d(height, Matrix1d(width)));
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::binomial_distribution<int> generate_random(experiment_times, probability);
    for (int i = 0; i < channel_number; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                random_matrix[i][j][k] = generate_random(random_engine);
            }
        }
    }

    return 0;
}

//生成服从二项分布的随机数三维矩阵 
template <typename DataType>
int8_t Random<DataType>::Binomial(int32_t experiment_times, float probability, 
                                  std::tuple<int32_t, int32_t, int32_t> shape,  
                                  Matrix3d& random_matrix) {
    int channel_number;
    int height;
    int width;
    std::tie(channel_number, height, width) = shape;
    if (channel_number <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input channel number <= 0";
        return -1;
    }
    if (height <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input height <= 0";
        return -1;
    }
    if (width <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input width <= 0";
        return -1;
    }
    if (experiment_times <= 0) {
        LOG(ERROR) << "get binomial distribution matrix failed, input experiment times <= 0";
        return -1;
    }
    if (probability <= 0
            || probability >= 1) {
        LOG(ERROR) << "get binomial distribution matrix failed, input probability <= 0 or >= 1";
        return -1;
    }
    
    //判断输出矩阵是否初始化过
    if (0 != random_matrix.size()) {
        random_matrix.clear();
    }

    random_matrix = Matrix3d(channel_number, Matrix2d(height, Matrix1d(width)));
    
    std::random_device rand_device;
    //static std::mt19937 gen(rand_device());
    std::default_random_engine random_engine(rand_device());
    std::binomial_distribution<int> generate_random(experiment_times, probability);
    for (int i = 0; i < channel_number; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                random_matrix[i][j][k] = generate_random(random_engine);
            }
        }
    }

    return 0;
}

//全连接层的隐藏层 训练时使用dropout 防止过拟合
//dropout过程相当于对很多不同的神经网络取平均 不同网络产生不同的过拟合 相互抵消
template <typename DataType>
int8_t Random<DataType>::DropOut(const Matrix2d& source_matrix, 
                                 int32_t experiment_times, float probability, 
                                 Matrix2d& binomial_matrix, 
                                 Matrix2d& result_matrix) {
    //伯努利分布 得到0 1数组 
    if (-1 == Binomial(experiment_times, probability,
                       matrix::Matrix<double>::GetShape(source_matrix), binomial_matrix)) {
        LOG(ERROR) << "Drop Out failed, binomial distribution occur error";
        return -1;
    }
    
    //相乘 矩阵中就有一部分参数置0了
    if (-1 == matrix::Matrix<double>::HadamarkProduct(source_matrix, binomial_matrix, result_matrix)) {
        LOG(ERROR) << "Drop Out failed, matrix hadamark product occur error";
        return -1;
    }
    
    //对所有激活值rescale 乘以1/(1-p) 因为有些值置0了 缩放一下总期望才能是一样的
    float rescale = 1.0 / (1.0 - probability);
    if (-1 == matrix::Matrix<double>::ValueMulMatrix(rescale, result_matrix, result_matrix)) {
        LOG(ERROR) << "Drop Out failed, matrix rescale occur error";
        return -1;
    }

    return 0;
}

//全连接层的隐藏层 训练时使用dropout 防止过拟合
//dropout过程相当于对很多不同的神经网络取平均 不同网络产生不同的过拟合 相互抵消
template <typename DataType>
int8_t Random<DataType>::DropOut(const Matrix3d& source_matrix, 
                                 int32_t experiment_times, float probability, 
                                 Matrix2d& binomial_matrix, 
                                 Matrix3d& result_matrix) {
    auto shape = matrix::Matrix<double>::GetShape(source_matrix);
    int height = std::get<1>(shape);
    int width = std::get<2>(shape);
    //伯努利分布 得到0 1数组 
    if (-1 == Binomial(experiment_times, probability,
                       height, width, binomial_matrix)) {
        LOG(ERROR) << "Drop Out failed, binomial distribution occur error";
        return -1;
    }
    
    //对所有激活值rescale 乘以1/(1-p) 因为有些值置0了 缩放一下总期望才能是一样的
    float rescale = 1.0 / (1.0 - probability);
    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!matrix::Matrix<double>::MatrixCheck(source_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = source_matrix;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int i = 0; i < source_matrix.size(); i++) {
            for (int j = 0; j < source_matrix[i].size(); j++) {
                for (int k = 0; k < source_matrix[i][j].size(); k++) {
                    result_matrix[i][j][k] = source_matrix[i][j][k] * binomial_matrix[j][k] * rescale;
                }
            }
        }
    }

    return 0;
}

//全连接层的隐藏层 训练时使用dropout 防止过拟合
//dropout过程相当于对很多不同的神经网络取平均 不同网络产生不同的过拟合 相互抵消
template <typename DataType>
int8_t Random<DataType>::DropOut(const Matrix3d& delta_matrix, 
                                 const Matrix2d& binomial_matrix,
                                 const Matrix3d& input_matrix, 
                                 double rescale, 
                                 Matrix3d& result_matrix) {
    if (0 == delta_matrix.size()
            || 0 == binomial_matrix.size()
            || 0 == input_matrix.size()
            || 0 == result_matrix.size()) {
        LOG(ERROR) << "Drop Out failed, input matrix is empty";
        return -1;
    }

    //检查结果矩阵是否有 并且格式正确  不正确就用赋值拷贝重新赋值
    if (!matrix::Matrix<double>::MatrixCheck(delta_matrix, result_matrix, false)) {
        result_matrix.clear();
        result_matrix = delta_matrix;
    }
    
#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //开始乘积运算
        for (int i = 0; i < delta_matrix.size(); i++) {
            for (int j = 0; j < delta_matrix[i].size(); j++) {
                for (int k = 0; k < delta_matrix[i][j].size(); k++) {
                    if (input_matrix[i][j][k] > 0) {
                        result_matrix[i][j][k] = delta_matrix[i][j][k] * binomial_matrix[j][k] / rescale;
                    } else {
                        result_matrix[i][j][k] = 0.0;
                    }
                }
            }
        }
    }

    return 0;
}






}       //namespace random



namespace time {
static void GetCurrentTime(char* now_time);


void GetCurrentTime(char* now_time) {
    time_t now = std::chrono::system_clock::to_time_t(
                              std::chrono::system_clock::now());
    
    struct tm* ptime = localtime(&now);
    sprintf(now_time, "%d-%02d-%02d %02d:%02d:%02d",
		   (int)ptime->tm_year + 1900, (int)ptime->tm_mon + 1, 
           (int)ptime->tm_mday,        (int)ptime->tm_hour, 
           (int)ptime->tm_min,         (int)ptime->tm_sec);
}

}         //namespace time



namespace activator {

//模板类  激活函数
template <typename DataType=double>
struct Activator {
    //类型别名
    typedef std::vector<double> Matrix1d;
    typedef std::vector<std::vector<double>> Matrix2d;
    typedef std::vector<std::vector<std::vector<double>>> Matrix3d;
    typedef std::vector<std::vector<std::vector<std::vector<double>>>> Matrix4d;
    typedef std::vector<std::vector<uint8_t>> ImageMatrix2d;
    typedef std::vector<std::vector<std::vector<uint8_t>>> ImageMatrix3d;
    typedef std::vector<std::vector<std::vector<std::vector<uint8_t>>>> ImageMatrix4d;

    //sigmoid激活函数的前向计算
    static void SigmoidForward(const Matrix2d& input_array, 
                               Matrix2d& output_array);

    //sigmoid激活函数的反向计算
    static void SigmoidBackward(const Matrix2d& output_array, 
                                Matrix2d& delta_array);

    //sigmoid激活函数的反向计算
    static void SigmoidImageBackward(const ImageMatrix2d& output_array, 
                                     Matrix2d& delta_array);

    //ReLu激活函数的前向计算
    static void ReLuForward2d(const Matrix2d& input_array, 
                              Matrix2d& output_array);

    //ReLu激活函数的前向计算
    static void ReLuForward3d(const Matrix3d& input_array, 
                              Matrix3d& output_array);

    //ReLu激活函数的前向计算
    static void ReLuForward4d(const Matrix4d& input_array, 
                              Matrix4d& output_array);

    //ReLu激活函数的反向计算
    static void ReLuBackward2d(const Matrix2d& input_array, 
                               Matrix2d& output_array);

    //ReLu激活函数的反向计算
    static void ReLuBackward3d(const Matrix3d& input_array, 
                               Matrix3d& output_array);

    //ReLu激活函数的反向计算
    static void ReLuBackward4d(const Matrix4d& input_array, 
                               Matrix4d& output_array);

    //ReLu激活函数的反向计算
    static void ReLuImageBackward3d(const ImageMatrix3d& input_array, 
                                    Matrix3d& output_array);

    //ReLu激活函数的反向计算
    static void ReLuImageBackward4d(const ImageMatrix4d& input_array, 
                                    Matrix4d& output_array);

};        //struct Activator


//sigmoid激活函数的前向计算
template <typename DataType>
void Activator<DataType>::SigmoidForward(const Matrix2d& input_array, 
                                         Matrix2d& output_array) { 
    //如果输出数组未初始化 
    if (0 == output_array.size()) {
        output_array = input_array;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //计算 1 / (1 + exp(-input_array))
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                //exp返回e的x次方 得到0. 1. 2.值 加上1都大于1了 然后用1除  最后都小于1
                output_array[i][j] = 1.0 / (1.0 + exp(-input_array[i][j])); 
            }
        }
    }
}

//sigmoid激活函数的反向计算
template <typename DataType>
void Activator<DataType>::SigmoidBackward(const Matrix2d& output_array, 
                                          Matrix2d& delta_array) {
    //如果输出数组未初始化 
    if (0 == delta_array.size()) {
        delta_array = output_array;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //计算 output(1 - output)
        for (int i = 0; i < output_array.size(); i++) {
            for (int j = 0; j < output_array[i].size(); j++) {
                delta_array[i][j] = output_array[i][j] * (1.0 - output_array[i][j]);
            }
        }
    }
}

//sigmoid激活函数的反向计算
template <typename DataType>
void Activator<DataType>::SigmoidImageBackward(const ImageMatrix2d& output_array, 
                                               Matrix2d& delta_array) {
    //如果输出数组未初始化 
    if (0 == delta_array.size()) {
        auto shape = matrix::Matrix<uint8_t>::GetShape(output_array);
        matrix::Matrix<double>::CreateZeros(shape, delta_array);
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //计算 output(1 - output)
        for (int i = 0; i < output_array.size(); i++) {
            for (int j = 0; j < output_array[i].size(); j++) {
                delta_array[i][j] = output_array[i][j] * (1.0 - output_array[i][j]);
            }
        }
    }
}

//ReLu激活函数的2d前向计算
template <typename DataType>
void Activator<DataType>::ReLuForward2d(const Matrix2d& input_array, 
                                        Matrix2d& output_array) {
    if (!matrix::Matrix<double>::MatrixCheck(input_array, true)) {
        LOG(ERROR) << "relu forward activator failed, input array is empty";
        return ;
    }

    if (!matrix::Matrix<double>::MatrixCheck(input_array, output_array, false)) {
        output_array.clear();
        output_array = input_array;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //大于0是本身 小于0就是0
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                output_array[i][j] = std::max<double>(0.0, input_array[i][j]);
            }
        }
    }
}

//ReLu激活函数的3d前向计算
template <typename DataType>
void Activator<DataType>::ReLuForward3d(const Matrix3d& input_array, 
                                        Matrix3d& output_array) {
    if (!matrix::Matrix<double>::MatrixCheck(input_array, true)) {
        LOG(ERROR) << "relu forward activator failed, input array is empty";
        return ;
    }

    if (!matrix::Matrix<double>::MatrixCheck(input_array, output_array, false)) {
        output_array.clear();
        output_array = input_array;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //大于0是本身 小于0就是0
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                for (int k = 0; k < input_array[i][j].size(); k++) {
                    output_array[i][j][k] = std::max<double>(0.0, input_array[i][j][k]);
                }
            }
        }
    }
}

//ReLu激活函数的3d前向计算
template <typename DataType>
void Activator<DataType>::ReLuForward4d(const Matrix4d& input_array, 
                                        Matrix4d& output_array) {
    if (!matrix::Matrix<double>::MatrixCheck(input_array, true)) {
        LOG(ERROR) << "relu forward activator failed, input array is empty";
        return ;
    }

    if (!matrix::Matrix<double>::MatrixCheck(input_array, output_array, false)) {
        output_array.clear();
        output_array = input_array;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //大于0是本身 小于0就是0
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                for (int k = 0; k < input_array[i][j].size(); k++) {
                    for (int z = 0; z < input_array[i][j][k].size(); z++) {
                        output_array[i][j][k][z] = std::max<double>(0.0, input_array[i][j][k][z]);
                    }
                }
            }
        }
    }
}

//ReLu激活函数的2d反向计算
template <typename DataType>
void Activator<DataType>::ReLuBackward2d(const Matrix2d& input_array, 
                                         Matrix2d& output_array) {
    if (!matrix::Matrix<double>::MatrixCheck(input_array, true)) {
        LOG(ERROR) << "relu backward activator failed, input array is empty";
        return ;
    }

    if (!matrix::Matrix<double>::MatrixCheck(input_array, output_array, false)) {
        output_array.clear();
        output_array = input_array;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //大于0就是1 其余是0
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                if (input_array[i][j] > 0) {
                    output_array[i][j] = 1;
                } else {
                    output_array[i][j] = 0;
                }
            }
        }
    }
}

//ReLu激活函数的3d反向计算
template <typename DataType>
void Activator<DataType>::ReLuBackward3d(const Matrix3d& input_array, 
                                         Matrix3d& output_array) {
    if (!matrix::Matrix<double>::MatrixCheck(input_array, true)) {
        LOG(ERROR) << "relu backward activator failed, input array is empty";
        return ;
    }

    if (!matrix::Matrix<double>::MatrixCheck(input_array, output_array, false)) {
        output_array.clear();
        output_array = input_array;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //大于0就是1 其余是0
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                for (int k = 0; k < input_array[i][j].size(); k++) {
                    if (input_array[i][j][k] > 0) {
                        output_array[i][j][k] = 1;
                    } else {
                        output_array[i][j][k] = 0;
                    }
                }
            }
        }
    }
}

//ReLu激活函数的4d反向计算
template <typename DataType>
void Activator<DataType>::ReLuBackward4d(const Matrix4d& input_array, 
                                         Matrix4d& output_array) {
    if (!matrix::Matrix<double>::MatrixCheck(input_array, true)) {
        LOG(ERROR) << "relu backward activator failed, input array is empty";
        return ;
    }

    if (!matrix::Matrix<double>::MatrixCheck(input_array, output_array, false)) {
        output_array.clear();
        output_array = input_array;
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //大于0就是1 其余是0
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                for (int k = 0; k < input_array[i][j].size(); k++) {
                    for (int z = 0; z < input_array[i][j][k].size(); z++) {
                        if (input_array[i][j][k][z] > 0) {
                            output_array[i][j][k][z] = 1;
                        } else {
                            output_array[i][j][k][z] = 0;
                        }
                    }
                }
            }
        }
    }
}


//ReLu激活函数的3d反向计算
template <typename DataType>
void Activator<DataType>::ReLuImageBackward3d(const ImageMatrix3d& input_array, 
                                              Matrix3d& output_array) {
    if (!matrix::Matrix<uint8_t>::MatrixCheck(input_array, true)) {
        LOG(ERROR) << "relu backward activator failed, input array is empty";
        return ;
    }

    if (!matrix::Matrix<uint8_t>::MatrixCheck(input_array, output_array, false)) {
        output_array.clear();
        output_array = matrix::Matrix<double>::ToDouble(input_array);
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //大于0就是1 其余是0
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                for (int k = 0; k < input_array[i][j].size(); k++) {
                    if (input_array[i][j][k] > 0) {
                        output_array[i][j][k] = 1;
                    } else {
                        output_array[i][j][k] = 0;
                    }
                }
            }
        }
    }
}

//ReLu激活函数的4d反向计算
template <typename DataType>
void Activator<DataType>::ReLuImageBackward4d(const ImageMatrix4d& input_array, 
                                              Matrix4d& output_array) {
    if (!matrix::Matrix<uint8_t>::MatrixCheck(input_array, true)) {
        LOG(ERROR) << "relu backward activator failed, input array is empty";
        return ;
    }

    if (!matrix::Matrix<uint8_t>::MatrixCheck(input_array, output_array, false)) {
        output_array.clear();
        output_array = matrix::Matrix<double>::ToDouble(input_array);
    }

#pragma omp parallel num_threads(OPENMP_THREADS_NUMBER)
    {
        #pragma omp for schedule(static) 
        //大于0就是1 其余是0
        for (int i = 0; i < input_array.size(); i++) {
            for (int j = 0; j < input_array[i].size(); j++) {
                for (int k = 0; k < input_array[i][j].size(); k++) {
                    for (int z = 0; z < input_array[i][j][k].size(); z++) {
                        if (input_array[i][j][k][z] > 0) {
                            output_array[i][j][k][z] = 1;
                        } else {
                            output_array[i][j][k][z] = 0;
                        }
                    }
                }
            }
        }
    }
}






}         //namespace activator
}         //namespace calculate
}         //namespace moon

//定义别名
typedef moon::calculate::matrix::Matrix<double> Matrix;
typedef moon::calculate::random::Random<double> Random;
typedef moon::calculate::activator::Activator<double> Activator;

//图像矩阵
typedef moon::calculate::matrix::Matrix<uint8_t> ImageMatrix;
typedef moon::calculate::random::Random<uint8_t> ImageRandom;
typedef moon::calculate::activator::Activator<uint8_t> ImageActivator;


#endif    //MOON_CALCULATE_MATRIX_CPU_HPP__
