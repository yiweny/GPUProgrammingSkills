# cublas-cudnn

This guide covers NVIDIA GPU-accelerated math library integration using cuBLAS, cuDNN, and related libraries.

## Overview

Covered GPU library operations include:
- Configure cuBLAS tensor core operations
- Generate cuBLAS GEMM calls with optimal parameters
- Integrate cuDNN convolution and normalization layers
- Handle cuBLAS/cuDNN algorithm selection
- Configure workspace memory requirements
- Benchmark library operations vs custom kernels
- Support mixed-precision operations (FP16, TF32, INT8)
- Integrate with cuSPARSE for sparse operations

## Prerequisites

- CUDA Toolkit 11.0+
- cuBLAS library
- cuDNN 8.0+
- cuSPARSE (optional)

## Capabilities

### 1. cuBLAS GEMM Operations

Matrix multiplication with cuBLAS:

```c
#include <cublas_v2.h>

// Initialize cuBLAS
cublasHandle_t handle;
cublasCreate(&handle);

// Standard SGEMM: C = alpha * A * B + beta * C
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
    M, N, K,                    // Dimensions
    &alpha,
    d_A, M,                     // A matrix and leading dimension
    d_B, K,                     // B matrix and leading dimension
    &beta,
    d_C, M);                    // C matrix and leading dimension

// Batched GEMM for multiple matrices
cublasSgemmBatched(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    d_Aarray, M,
    d_Barray, K,
    &beta,
    d_Carray, M,
    batchCount);

// Strided batched GEMM (contiguous memory)
cublasSgemmStridedBatched(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    d_A, M, strideA,
    d_B, K, strideB,
    &beta,
    d_C, M, strideC,
    batchCount);
```

### 2. Tensor Core Operations

Enable tensor cores for maximum performance:

```c
// Enable tensor cores (requires Volta+)
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// For Ampere+, use TF32
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

// Half precision GEMM with tensor cores
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    d_A, CUDA_R_16F, M,      // FP16 input
    d_B, CUDA_R_16F, K,      // FP16 input
    &beta,
    d_C, CUDA_R_16F, M,      // FP16 output
    CUDA_R_16F,              // Compute type
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);

// Mixed precision: FP16 inputs, FP32 accumulate
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    d_A, CUDA_R_16F, M,
    d_B, CUDA_R_16F, K,
    &beta,
    d_C, CUDA_R_32F, M,      // FP32 output
    CUDA_R_32F,              // FP32 compute
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### 3. cuDNN Convolution

Deep learning convolutions:

```c
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error: %s\n", cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

cudnnHandle_t cudnn;
CUDNN_CHECK(cudnnCreate(&cudnn));

cudnnTensorDescriptor_t inputDesc, outputDesc;
CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW,
    CUDNN_DATA_FLOAT, N, C, H, W));

cudnnFilterDescriptor_t filterDesc;
CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
    CUDNN_TENSOR_NCHW, K, C, R, S));

cudnnConvolutionDescriptor_t convDesc;
CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
    pad_h, pad_w, stride_h, stride_w, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

// Derive and configure the output descriptor before algorithm selection.
int outN, outC, outH, outW;
CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
    convDesc, inputDesc, filterDesc, &outN, &outC, &outH, &outW));
CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW,
    CUDNN_DATA_FLOAT, outN, outC, outH, outW));

float* d_output;
size_t outputBytes = (size_t)outN * outC * outH * outW * sizeof(float);
CUDA_CHECK(cudaMalloc(&d_output, outputBytes));

cudnnConvolutionFwdAlgoPerf_t perfResults[8];
int returnedAlgoCount;
CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn,
    inputDesc, filterDesc, convDesc, outputDesc,
    8, &returnedAlgoCount, perfResults));
if (returnedAlgoCount == 0 || perfResults[0].status != CUDNN_STATUS_SUCCESS) {
    fprintf(stderr, "No successful cuDNN forward convolution algorithm found\n");
    exit(EXIT_FAILURE);
}
cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;

size_t workspaceSize;
CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
    inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize));
void* workspace = NULL;
if (workspaceSize > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
}

float alpha = 1.0f, beta = 0.0f;
CUDNN_CHECK(cudnnConvolutionForward(cudnn,
    &alpha, inputDesc, d_input, filterDesc, d_filter,
    convDesc, algo, workspace, workspaceSize,
    &beta, outputDesc, d_output));
```

### 4. cuDNN Batch Normalization

```c
cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc, inputDesc,
    CUDNN_BATCHNORM_SPATIAL);

// Forward training
cudnnBatchNormalizationForwardTraining(cudnn,
    CUDNN_BATCHNORM_SPATIAL,
    &alpha, &beta,
    inputDesc, d_input,
    outputDesc, d_output,
    bnScaleBiasMeanVarDesc,
    d_scale, d_bias,
    0.1,  // Exponential average factor
    d_runningMean, d_runningVariance,
    1e-5, // Epsilon
    d_savedMean, d_savedInvVariance);

// Forward inference
cudnnBatchNormalizationForwardInference(cudnn,
    CUDNN_BATCHNORM_SPATIAL,
    &alpha, &beta,
    inputDesc, d_input,
    outputDesc, d_output,
    bnScaleBiasMeanVarDesc,
    d_scale, d_bias,
    d_runningMean, d_runningVariance,
    1e-5);
```

### 5. Algorithm Selection and Benchmarking

```c
// Benchmark all algorithms
cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
int returnedCount;

cudnnFindConvolutionForwardAlgorithmEx(cudnn,
    inputDesc, d_input,
    filterDesc, d_filter,
    convDesc,
    outputDesc, d_output,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    &returnedCount,
    perfResults,
    workspace, workspaceSize);

// Print benchmark results
for (int i = 0; i < returnedCount; i++) {
    printf("Algorithm %d: %.3f ms, workspace: %zu bytes\n",
        perfResults[i].algo,
        perfResults[i].time,
        perfResults[i].memory);
}

// Select algorithm by heuristics
cudnnConvolutionFwdAlgo_t algo;
cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
    inputDesc, filterDesc, convDesc, outputDesc,
    8, &returnedCount, perfResults);
```

### 6. Workspace Memory Management

```c
// Query workspace for all operations
size_t convWorkspace, bnWorkspace, poolWorkspace;

cudnnGetConvolutionForwardWorkspaceSize(cudnn, ...);
cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnn, ...);

// Allocate maximum needed
size_t maxWorkspace = max(convWorkspace, max(bnWorkspace, poolWorkspace));
void* workspace;
cudaMalloc(&workspace, maxWorkspace);

// Reuse workspace across operations
```

### 7. Mixed Precision Support

```c
// FP16 convolution
cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NHWC,
    CUDNN_DATA_HALF, N, C, H, W);
cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_HALF,
    CUDNN_TENSOR_NHWC, K, C, R, S);

// INT8 convolution for inference
cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NHWC,
    CUDNN_DATA_INT8, N, C, H, W);
cudnnSetConvolution2dDescriptor(convDesc,
    pad_h, pad_w, stride_h, stride_w, 1, 1,
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_INT32);  // INT32 accumulation
```

### 8. cuSPARSE Integration

```c
#include <cusparse.h>

cusparseHandle_t sparse;
cusparseCreate(&sparse);

// Create sparse matrix in CSR format
cusparseSpMatDescr_t matA;
cusparseCreateCsr(&matA, M, N, nnz,
    d_rowPtr, d_colIdx, d_values,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

// Create dense vectors
cusparseDnVecDescr_t vecX, vecY;
cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F);
cusparseCreateDnVec(&vecY, M, d_y, CUDA_R_32F);

// SpMV: y = alpha * A * x + beta * y
size_t bufferSize;
cusparseSpMV_bufferSize(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
    CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

void* buffer;
cudaMalloc(&buffer, bufferSize);

cusparseSpMV(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
    CUSPARSE_SPMV_ALG_DEFAULT, buffer);
```

## Output Format

```json
{
  "operation": "gemm-benchmark",
  "library": "cuBLAS",
  "configuration": {
    "M": 4096, "N": 4096, "K": 4096,
    "datatype": "FP16",
    "math_mode": "TENSOR_OP_MATH"
  },
  "performance": {
    "time_ms": 0.85,
    "tflops": 16.2,
    "efficiency_pct": 81.0
  },
  "recommendations": [
    "Use CUBLAS_GEMM_DEFAULT_TENSOR_OP for tensor core path",
    "Ensure dimensions are multiples of 8 for optimal tensor core usage"
  ]
}
```

## Dependencies

- CUDA Toolkit 11.0+
- cuBLAS
- cuDNN 8.0+
- cuSPARSE (optional)

## Constraints

- Tensor cores require specific data types and alignments
- Algorithm selection should be cached per configuration
- Workspace memory must be allocated before execution
- Mixed precision may require loss scaling
