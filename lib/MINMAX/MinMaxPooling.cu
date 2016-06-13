#include "MINMAX.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include <cfloat>

__global__ void cuda_MinMaxPooling_updateOutput(
  THCDeviceTensor<float, 4> input,
  THCDeviceTensor<float, 2> thresholds,
  THCDeviceTensor<float, 4> mask,
  THCDeviceTensor<float, 4> indices,
  THCDeviceTensor<float, 4> output,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int padT, int padH, int padW)
{
  int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame  = blockIdx.z % output.getSize(1); // output frame/time
  int slice   = blockIdx.z / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oColumn < output.getSize(3))
  {
    int iColumn = oColumn * dW - padW;
    int iRow    = oRow    * dH - padH;
    int iFrame  = oFrame  * dT - padT;

    int minColumn = 0;
    int minRow = 0;
    int minFrame = 0;

    int miniColumn = 0;
    int miniRow = 0;
    int miniFrame = 0;

    float min = FLT_MAX;

    for (int frame = 0; frame < kT; ++frame)
    {
      if (iFrame + frame < input.getSize(1) && iFrame + frame >= 0)
      {
        for (int row = 0; row < kH; ++row)
        {
          if (iRow + row < input.getSize(2) && iRow + row >= 0)
          {
            for (int column = 0; column < kW; ++column)
            {
              if (iColumn + column < input.getSize(3) && iColumn + column >= 0)
              {
                float val = input[slice][iFrame + frame][iRow + row][iColumn + column];
                float threshold_lower = thresholds[0][frame];
                float threshold_upper = thresholds[1][frame];

                if ( val < threshold_lower || val > threshold_upper || frame == kT-1)
                {
                  min = val;
                  minColumn = column;
                  minRow    = row;
                  minFrame  = frame;

                  miniColumn = iColumn + column;
                  miniRow = iRow + row;
                  miniFrame = iFrame + frame;
                  break;
                }
              }
            }

            if (min < FLT_MAX) 
            {
              break;
            }
          }
        }

        if (min < FLT_MAX)
        {
          break;
        }
      }
    }

    if (min == FLT_MAX)
    {
      min = 0;
    }

    output[slice][oFrame][oRow][oColumn] = min;
    if (min == 0)
    {
      mask[slice][miniFrame][miniRow][miniColumn] = 0.0;
    }
    else
    {
      mask[slice][miniFrame][miniRow][miniColumn] = 1.0;
    }

    float *idx = &indices[slice][oFrame][oRow][oColumn];
    ((unsigned char*)(idx))[0] = minFrame;
    ((unsigned char*)(idx))[1] = minRow;
    ((unsigned char*)(idx))[2] = minColumn;
    ((unsigned char*)(idx))[3] = 0;
  }
}

template <int KERNEL_WIDTH>
__global__ void cuda_MinMaxPooling_updateOutput(
  THCDeviceTensor<float, 4> input, 
  THCDeviceTensor<float, 2> thresholds, 
  THCDeviceTensor<float, 4> mask,
  THCDeviceTensor<float, 4> indices,
  THCDeviceTensor<float, 4> output,
  int kT, int kH,
  int dT, int dH, int dW,
  int padT, int padH, int padW)
{
  int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame  = blockIdx.z % output.getSize(1); // output frame/time
  int slice   = blockIdx.z / output.getSize(1); // output slice/feature

  if (oRow < output.getSize(2) && oColumn < output.getSize(3))
  {
    int iColumn = oColumn * dW - padW;
    int iRow    = oRow    * dH - padH;
    int iFrame  = oFrame  * dT - padT;

    int minColumn = 0;
    int minRow = 0;
    int minFrame;

    int miniColumn = 0;
    int miniRow = 0;
    int miniFrame = 0;

    float min = FLT_MAX;

    for (int frame = 0; frame < kT; ++frame)
    {
      if (iFrame + frame < input.getSize(1) && iFrame + frame >= 0)
      {
        for (int row = 0; row < kH; ++row)
        {
          if (iRow + row < input.getSize(2) && iRow + row >= 0)
          {
            for (int column = 0; column < KERNEL_WIDTH; ++column)
            {
              if (iColumn + column < input.getSize(3) && iColumn + column >= 0)
              {
                float val = input[slice][iFrame + frame][iRow + row][iColumn + column];
                float threshold_lower = thresholds[0][frame];
                float threshold_upper = thresholds[1][frame];

                if ( val < threshold_lower || val > threshold_upper || frame == kT-1)
                {
                  min = val;
                  minColumn = column;
                  minRow    = row;
                  minFrame  = frame;

                  miniColumn = iColumn + column;
                  miniRow = iRow + row;
                  miniFrame = iFrame + frame;
                  break;
                }
              }
            }

            if (min < FLT_MAX) 
            {
              break;
            }
          }
        }

        if (min < FLT_MAX)
        {
          break;
        }
      }
    }

    if (min == FLT_MAX)
    {
      min = 0;
    }

    output[slice][oFrame][oRow][oColumn] = min;
    if (min == 0)
    {
      mask[slice][miniFrame][miniRow][miniColumn] = 0.0;
    }
    else
    {
      mask[slice][miniFrame][miniRow][miniColumn] = 1.0;
    }

    float *idx = &indices[slice][oFrame][oRow][oColumn];
    ((unsigned char*)(idx))[0] = minFrame;
    ((unsigned char*)(idx))[1] = minRow;
    ((unsigned char*)(idx))[2] = minColumn;
    ((unsigned char*)(idx))[3] = 0;
  }
}

#define UPDATE_OUTPUT_KERNEL_WIDTH(KW) case KW:                                   \
  cuda_MinMaxPooling_updateOutput<KW><<<grid, block,                       \
                                0, THCState_getCurrentStream(state)>>>(           \
      cudaInput, cudaThresholds, cudaMask, cudaIndices, cudaOutput, kT, kH, dT, dH, dW, padT, padH, padW);  \
    break


void THNN_CudaMinMaxPooling_updateOutput(
  THCState *state, THCudaTensor *input, THCudaTensor *thresholds, 
  THCudaTensor *mask, THCudaTensor *output, THCudaTensor *indices,
  int kT, int kW, int kH,
  int dT, int dW, int dH,
  int padT, int padW, int padH,
  bool ceilMode)
{
  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;
  int outputTime;
  int outputHeight;
  int outputWidth;

  MINMAX_assertSameGPU(state, 5, input, thresholds, mask, indices, output);

  if (THCudaTensor_nDimension(state, input) == 4)
  {
    THArgCheck(
      THCudaTensor_size(state, input, 1) >= kT &&
      THCudaTensor_size(state, input, 2) >= kH &&
      THCudaTensor_size(state, input, 3) >= kW, 2,
      "input image smaller than kernel size"
    );

    /* sizes */
    batchSize   = 1;
    inputSlices = THCudaTensor_size(state, input, 0);
    inputTime   = THCudaTensor_size(state, input, 1);
    inputHeight = THCudaTensor_size(state, input, 2);
    inputWidth  = THCudaTensor_size(state, input, 3);
  }
  else if (THCudaTensor_nDimension(state, input) == 5)
  {
    THArgCheck(
      THCudaTensor_size(state, input, 4) >= kW &&
      THCudaTensor_size(state, input, 3) >= kH &&
      THCudaTensor_size(state, input, 2) >= kT, 2,
      "input image smaller than kernel size"
    );

    /* sizes */
    batchSize   = THCudaTensor_size(state, input, 0);
    inputSlices = THCudaTensor_size(state, input, 1);
    inputTime   = THCudaTensor_size(state, input, 2);
    inputHeight = THCudaTensor_size(state, input, 3);
    inputWidth  = THCudaTensor_size(state, input, 4);
  }
  else
  {
    THArgCheck(false, 2, "4D or 5D tensor expected");
  }

  THArgCheck(kT/2 >= padT && kW/2 >= padW && kH/2 >= padH, 2,
    "pad should be smaller than half of kernel size"
  );

  if (ceilMode)
  {
    outputTime   = (int)(ceil((float)(inputTime   - kT + 2*padT) / dT)) + 1;
    outputHeight = (int)(ceil((float)(inputHeight - kH + 2*padH) / dH)) + 1;
    outputWidth  = (int)(ceil((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
  }
  else
  {
    outputTime   = (int)(floor((float)(inputTime   - kT + 2*padT) / dT)) + 1;
    outputHeight = (int)(floor((float)(inputHeight - kH + 2*padH) / dH)) + 1;
    outputWidth  = (int)(floor((float)(inputWidth  - kW + 2*padW) / dW)) + 1;
  }

  if (padT || padW || padH)
  {
    if ((outputTime - 1)*dT >= inputTime + padT)
      --outputTime;
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  if (input->nDimension == 4) /* 4D */
  {
    /* resize mask */
    THCudaTensor_resize4d(state, mask, inputSlices,
                          inputTime, inputHeight, inputWidth);
    /* resize output */
    THCudaTensor_resize4d(state, output, inputSlices,
                          outputTime, outputHeight, outputWidth);
    /* indices pack ti,i,j locations for each output point as uchar into
     each float of the tensor */
    THCudaTensor_resize4d(state, indices, inputSlices,
                          outputTime, outputHeight, outputWidth);
  }
  else
  { /* 5D */
    THCudaTensor_resize5d(state, mask, batchSize, inputSlices,
                          inputTime, inputHeight, inputWidth);
    THCudaTensor_resize5d(state, output, batchSize, inputSlices,
                          outputTime, outputHeight, outputWidth);
    // Index tensor packs index offsets as uchars into floats
    THCudaTensor_resize5d(state, indices, batchSize, inputSlices,
                          outputTime, outputHeight, outputWidth);
  }

  input = THCudaTensor_newContiguous(state, input);

  // Collapse batch and feature dimensions
  THCDeviceTensor<float, 4> cudaInput;
  THCDeviceTensor<float, 2> cudaThresholds;
  // THCDeviceTensor<float, 4> cudaMask;
  THCDeviceTensor<float, 4> cudaOutput;
  if (THCudaTensor_nDimension(state, input) == 4)
  {
    cudaInput  = toDeviceTensor<float, 4>(state, input);
    cudaThresholds = toDeviceTensor<float, 2>(state, thresholds);
    // cudaMask   = toDeviceTensor<float, 4>(state, mask);
    cudaOutput = toDeviceTensor<float, 4>(state, output);
  }
  else
  {
    cudaInput  = toDeviceTensor<float, 5>(state, input).downcastOuter<4>();
    cudaThresholds = toDeviceTensor<float, 2>(state, thresholds);
    // cudaMask   = toDeviceTensor<float, 5>(state, mask).downcastOuter<4>();
    cudaOutput = toDeviceTensor<float, 5>(state, output).downcastOuter<4>();
  }

  // copy indices tensor
  THLongStorage *indicesSize = THLongStorage_newWithSize(4);
  long indicesSizeRaw[4] = { batchSize * inputSlices,
                            outputTime, outputHeight, outputWidth };
  THLongStorage_rawCopy(indicesSize, indicesSizeRaw);

  THCudaTensor *indices1 = THCudaTensor_newWithStorage(
    state, THCudaTensor_storage(state, indices),
    THCudaTensor_storageOffset(state, indices),
    indicesSize, NULL);

  THLongStorage_free(indicesSize);

  THCDeviceTensor<float, 4> cudaIndices =
    toDeviceTensor<float, 4>(state, indices1);

  // copy mask tensor
  THLongStorage *maskSize = THLongStorage_newWithSize(4);
  long maskSizeRaw[4] = { batchSize * inputSlices,
                           inputTime, inputHeight, inputWidth };
  THLongStorage_rawCopy(maskSize, maskSizeRaw);
  THCudaTensor *mask1 = THCudaTensor_newWithStorage(
    state, THCudaTensor_storage(state, mask),
    THCudaTensor_storageOffset(state, mask), maskSize, NULL);
  THLongStorage_free(maskSize);

  // Clear mask
  THCudaTensor_zero(state, mask1);

  THCDeviceTensor<float, 4> cudaMask =
    toDeviceTensor<float, 4>(state, mask1);

  dim3 block(32, 8);
  dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
            THCCeilDiv(outputHeight, static_cast<int>(block.y)),
            outputTime * inputSlices * batchSize);

  switch (kW)
  {
    UPDATE_OUTPUT_KERNEL_WIDTH(1);
    UPDATE_OUTPUT_KERNEL_WIDTH(2);
    UPDATE_OUTPUT_KERNEL_WIDTH(3);
    UPDATE_OUTPUT_KERNEL_WIDTH(4);
    UPDATE_OUTPUT_KERNEL_WIDTH(5);
    UPDATE_OUTPUT_KERNEL_WIDTH(6);
    UPDATE_OUTPUT_KERNEL_WIDTH(7);
    default:
      cuda_MinMaxPooling_updateOutput<<<grid, block,
        0, THCState_getCurrentStream(state)>>>(
        cudaInput, cudaThresholds, cudaMask, cudaIndices, cudaOutput, kT, kH, kW, dT, dH, dW, padT, padH, padW);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, indices1);
  THCudaTensor_free(state, mask1);
}

#undef UPDATE_OUTPUT_KERNEL_WIDTH

__global__ void cuda_MinMaxPooling_updateGradInput(
  THCDeviceTensor<float, 4> gradOutput,
  THCDeviceTensor<float, 4> mask,
  THCDeviceTensor<float, 4> indices,
  THCDeviceTensor<float, 4> gradInput,
  int dT, int dH, int dW,
  int padT, int padH, int padW)
{
  int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame  = blockIdx.z % gradOutput.getSize(1); // output frame/time
  int slice   = blockIdx.z / gradOutput.getSize(1); // output slice/feature

  if (oRow < gradOutput.getSize(2) && oColumn < gradOutput.getSize(3))
  {
    float *idx = &indices[slice][oFrame][oRow][oColumn];
    int iFrame  = ((unsigned char*)(idx))[0] + oFrame  * dT - padT;
    int iRow    = ((unsigned char*)(idx))[1] + oRow    * dH - padH;
    int iColumn = ((unsigned char*)(idx))[2] + oColumn * dW - padW;
    float maskval = mask[slice][iFrame][iRow][iColumn];
    float maskedGradOutput = gradOutput[slice][oFrame][oRow][oColumn]*maskval;
    atomicAdd(&gradInput[slice][iFrame][iRow][iColumn],
              maskedGradOutput);
              //gradOutput[slice][oFrame][oRow][oColumn]);
  }
}

void THNN_CudaMinMaxPooling_updateGradInput(
  THCState *state, 
  THCudaTensor *input, THCudaTensor *mask, 
  THCudaTensor *gradOutput, THCudaTensor *gradInput,
  THCudaTensor *indices,
  int dT, int dW, int dH,
  int padT, int padW, int padH)
{
  // Resize and initialize result tensor.
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;

  int outputTime;
  int outputHeight;
  int outputWidth;

  MINMAX_assertSameGPU(state, 5, input, mask, indices, gradOutput, gradInput);

  if (THCudaTensor_nDimension(state, input) == 4) /* 4D */
  {
    batchSize = 1;
    inputSlices  = THCudaTensor_size(state, input, 0);
    inputTime   = THCudaTensor_size(state, input, 1);
    inputHeight = THCudaTensor_size(state, input, 2);
    inputWidth  = THCudaTensor_size(state, input, 3);

    outputTime   = THCudaTensor_size(state, gradOutput, 1);
    outputHeight = THCudaTensor_size(state, gradOutput, 2);
    outputWidth  = THCudaTensor_size(state, gradOutput, 3);
  }
  else
  {
    batchSize    = THCudaTensor_size(state, input, 0);
    inputSlices  = THCudaTensor_size(state, input, 1);
    inputTime   = THCudaTensor_size(state, input, 2);
    inputHeight = THCudaTensor_size(state, input, 3);
    inputWidth  = THCudaTensor_size(state, input, 4);

    outputTime   = THCudaTensor_size(state, gradOutput, 2);
    outputHeight = THCudaTensor_size(state, gradOutput, 3);
    outputWidth  = THCudaTensor_size(state, gradOutput, 4);
  }

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  // Collapse batch and feature dimensions
  THCDeviceTensor<float, 4> cudaGradInput;
  // THCDeviceTensor<float, 4> cudaMask;
  THCDeviceTensor<float, 4> cudaGradOutput;
  if (THCudaTensor_nDimension(state, input) == 4)
  {
    cudaGradInput  = toDeviceTensor<float, 4>(state, gradInput);
    // cudaMask       = toDeviceTensor<float, 4>(state, mask);
    cudaGradOutput = toDeviceTensor<float, 4>(state, gradOutput);
  }
  else
  {
    cudaGradInput =
      toDeviceTensor<float, 5>(state, gradInput).downcastOuter<4>();
    // cudaMask =
    //   toDeviceTensor<float, 5>(state, mask).downcastOuter<4>();
    cudaGradOutput =
      toDeviceTensor<float, 5>(state, gradOutput).downcastOuter<4>();
  }

  // copy indices tensor
  THLongStorage *indicesSize = THLongStorage_newWithSize(4);
  long indicesSizeRaw[4] = { batchSize * inputSlices,
                           outputTime, outputHeight, outputWidth };
  THLongStorage_rawCopy(indicesSize, indicesSizeRaw);
  THCudaTensor *indices1 = THCudaTensor_newWithStorage(
    state, THCudaTensor_storage(state, indices),
    THCudaTensor_storageOffset(state, indices), indicesSize, NULL);
  THLongStorage_free(indicesSize);

  THCDeviceTensor<float, 4> cudaIndices =
    toDeviceTensor<float, 4>(state, indices1);

  // copy mask tensor
  THLongStorage *maskSize = THLongStorage_newWithSize(4);
  long maskSizeRaw[4] = { batchSize * inputSlices,
                           inputTime, inputHeight, inputWidth };
  THLongStorage_rawCopy(maskSize, maskSizeRaw);
  THCudaTensor *mask1 = THCudaTensor_newWithStorage(
    state, THCudaTensor_storage(state, mask),
    THCudaTensor_storageOffset(state, mask), maskSize, NULL);
  THLongStorage_free(maskSize);

  THCDeviceTensor<float, 4> cudaMask =
    toDeviceTensor<float, 4>(state, mask1);

  dim3 block(32, 8);
  dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
            THCCeilDiv(outputHeight, static_cast<int>(block.y)),
            outputTime * inputSlices * batchSize);

  cuda_MinMaxPooling_updateGradInput<<<grid, block,
    0, THCState_getCurrentStream(state)>>>(
    cudaGradOutput,
    cudaMask,
    cudaIndices,
    cudaGradInput,
    dT, dH, dW,
    padT, padH, padW);

  // cleanup
  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, indices1);
  THCudaTensor_free(state, mask1);
}
