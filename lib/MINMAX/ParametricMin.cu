#include "MINMAX.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include <cfloat>

#define SCALE_LOOP(idx, nScales) for (int idx = 0; idx < (nScales); ++idx)

struct sigmoidupdateOutput_functor
{
  __device__ __forceinline__ void operator()(float *output, const float *input) const
  {
    *output = 1./(1.+ exp(-*input));
  }
};

struct sigmoidupdateGradInput_functor
{
  __device__ __forceinline__ void operator()(float *gradInput, const float *output, const float *gradOutput) const
  {
    *gradInput = *gradOutput * (1.-*output) * (*output);
  }
};

struct add_functor
{
  __device__ __forceinline__ void operator()(float *output, const float *input, const float value) const
  {
    *output = *input + value;
  }
};

struct mul_functor
{
  __device__ __forceinline__ void operator()(float *output, const float *input, const float value) const
  {
    *output = *input * value;
  }
};

struct add_mul_functor
{
  __device__ __forceinline__ void operator()(float *output, const float *input,
    const float addend, const float factor) const
  {
    *output = ((*input) + addend) * factor;
  }
};

__global__ void cuda_ParametricMin_updateOutput(
  const THCDeviceTensor<float, 3> input,
  const THCDeviceTensor<float, 1> thresholds,
  const THCDeviceTensor<float, 2> weights,
  THCDeviceTensor<float, 3> output,
  THCDeviceTensor<float, 3> reliability,
  THCDeviceTensor<float, 3> selection,
  const float kSlope,
  const int totalElements)
{
  const int nthreads = totalElements;

  const int kF = input.getSize(0); // number of features
  const int kS = input.getSize(1); // number of scales per feature

  //instantiate functors
  add_mul_functor addmul;
  mul_functor mul;
  sigmoidupdateOutput_functor sigmoid;

  CUDA_KERNEL_LOOP(index, nthreads) {
    // int N = (totalElements / kF);
    // int feature = index % N;
    // int elem = (index / N) % kF;
    int N = (totalElements / kF);
    int elem = index % N;
    int feature = index / N;

    if (feature < 0 || feature >= input.getSize(0)) continue;
    if (elem < 0 || elem >= input.getSize(2)) continue;

    //calculate reliability
    SCALE_LOOP(scale, kS) {
      const float *iv = &input[feature][scale][elem];
      float *rv = &reliability[feature][scale][elem];

      //subtract thresholds and multiply with slope factor
      addmul(rv, iv, -thresholds[scale], kSlope);

      //apply sigmoid
      sigmoid(rv, rv);
    }

    // multiply with weights
    SCALE_LOOP(scale, kS) {
      float sum = 0;

      // iterate over columns of weight matrix
      for (int i = 0; i < kS; ++i) {
        float factor;
        if ( i == scale ) factor = 1.0;
        else factor = weights[scale][i];
        float value = reliability[feature][i][elem];

        sum += factor * value;
      }

      output[feature][scale][elem] = sum;
    }

    // calculate selection
    SCALE_LOOP(scale, kS) {
      float *ov = &output[feature][scale][elem];
      const float iv = input[feature][scale][elem];
      float *sv = &selection[feature][scale][elem];

      addmul(sv, ov, -0.5, kSlope);
      sigmoid(sv, sv);

      //multiply with input value
      mul(ov, sv, iv);
    }
  }

}

template <int Dim>
static THCDeviceTensor<float, Dim> devicetensor(THCState *state, THCudaTensor *t) {
  if (!t) {
    return THCDeviceTensor<float, Dim>();
  }

  int inDim = THCudaTensor_nDimension(state, t);
  if (inDim == Dim) {
    return toDeviceTensor<float, Dim>(state, t);
  }

  // View in which the last dimensions are collapsed or expanded as needed
  THAssert(THCudaTensor_isContiguous(state, t));
  int size[Dim];
  for (int i = 0; i < Dim || i < inDim; ++i) {
    if (i < Dim && i < inDim) {
      size[i] = t->size[i];
    } else if (i < Dim) {
      size[i] = 1;
    } else {
      size[Dim - 1] *= t->size[i];
    }
  }
  return THCDeviceTensor<float, Dim>(THCudaTensor_data(state, t), size);
}

void THNN_CudaParametricMin_updateOutput(
  THCState *state,
  THCudaTensor *input, THCudaTensor *thresholds,
  THCudaTensor *weights, THCudaTensor *output,
  THCudaTensor *reliability, THCudaTensor *selection,
  float slope)
{
  //  int nFeatures;
   int nScales;
   int nBatches;
   int inputHeight;
   int inputWidth;

  MINMAX_assertSameGPU(state, 6, input, thresholds, weights, output, reliability, selection);

  if (THCudaTensor_nDimension(state, input) == 4)
  {
    /* sizes */
    // nFeatures   = 1;
    nScales     = THCudaTensor_size(state, input, 0);
    nBatches    = THCudaTensor_size(state, input, 1);
    inputHeight = THCudaTensor_size(state, input, 2);
    inputWidth  = THCudaTensor_size(state, input, 3);
  }
  else if (THCudaTensor_nDimension(state, input) == 5)
  {
    // nFeatures   = THCudaTensor_size(state, input, 0);
    nScales     = THCudaTensor_size(state, input, 1);
    nBatches    = THCudaTensor_size(state, input, 2);
    inputHeight = THCudaTensor_size(state, input, 3);
    inputWidth  = THCudaTensor_size(state, input, 4);
  }
  else
  {
    THArgCheck(false, 2, "4D or 5D tensor expected");
  }

  input = THCudaTensor_newContiguous(state, input);

  // View 4d input as 5d with leading single dimension
  if (THCudaTensor_nDimension(state, input) == 4)
  {
    THCudaTensor_resize5d(state, input, 1, nScales, nBatches, inputHeight, inputWidth);
  }
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_resizeAs(state, reliability, input);
  THCudaTensor_resizeAs(state, selection, input);


  // Collapse batch, height and width dimensions
  THCDeviceTensor<float, 3> cudaInput  = devicetensor<3>(state, input);
  THCDeviceTensor<float, 3> cudaOutput = devicetensor<3>(state, output);
  THCDeviceTensor<float, 1> cudaThresholds = devicetensor<1>(state, thresholds);
  THCDeviceTensor<float, 2> cudaWeights = devicetensor<2>(state, weights);
  THCDeviceTensor<float, 3> cudaReliability = devicetensor<3>(state, reliability);
  THCDeviceTensor<float, 3> cudaSelection = devicetensor<3>(state, selection);

  long totalElements = THCudaTensor_nElement(state, output) / nScales;

  cuda_ParametricMin_updateOutput<<<
    GET_BLOCKS(totalElements), CUDA_NUM_THREADS,
    0, THCState_getCurrentStream(state)>>>(
    cudaInput, cudaThresholds, cudaWeights, cudaOutput,
    cudaReliability, cudaSelection,
    slope, totalElements);

  THCudaTensor_free(state, input);
}

__global__ void cuda_ParametricMin_updateGradInput(
  const THCDeviceTensor<float, 3> input,
  const THCDeviceTensor<float, 1> thresholds,
  const THCDeviceTensor<float, 2> weights,
  const THCDeviceTensor<float, 3> reliability,
  const THCDeviceTensor<float, 3> selection,
  const THCDeviceTensor<float, 3> gradOutput,
  THCDeviceTensor<float, 3> gradInput,
  THCDeviceTensor<float, 3> gradReliability,
  THCDeviceTensor<float, 3> gradSelection,
  const float kSlope,
  const int totalElements,
  const float scalef)
{
  const int nthreads = totalElements;

  const int kF = input.getSize(0);
  const int kS = input.getSize(1);

  //instantiate functors
  mul_functor mul;
  sigmoidupdateGradInput_functor dsigmoid;

  CUDA_KERNEL_LOOP(index, nthreads) {
    // int N = (totalElements / kF);
    // int feature = index % N;
    // int elem = (index / N) % kF;
    int N = (totalElements / kF);
    int elem = index % N;
    int feature = index / N;

    if (feature < 0 || feature >= input.getSize(0)) continue;
    if (elem < 0 || elem >= input.getSize(2)) continue;

    //multiply gradOutput with input
    SCALE_LOOP(scale, kS) {
      const float *iv = &input[feature][scale][elem];
      const float *gov = &gradOutput[feature][scale][elem];
      float *gsv = &gradSelection[feature][scale][elem];

      *gsv = (*iv) * (*gov);
    }

    //calculate gradient of outer sigmoid
    SCALE_LOOP(scale, kS) {
      float *gsv = &gradSelection[feature][scale][elem];
      const float *sv = &selection[feature][scale][elem];

      dsigmoid(gsv, sv, gsv);
    }

    //multiply gradInput with transposed weights
    SCALE_LOOP(scale, kS) {
      float sum = 0;

      // iterate over columns of transposed weight matrix
      for (int i = 0; i < kS; ++i) {
        float factor;
        if ( i == scale ) factor = 1.0f;
        else factor = weights[i][scale];
        float value = gradSelection[feature][i][elem];

        sum += factor * value;
      }

      gradInput[feature][scale][elem] = sum;
    }

    //gradient of inner sigmoid
    SCALE_LOOP(scale, kS) {
      const float *rv = &reliability[feature][scale][elem];
      float *giv = &gradInput[feature][scale][elem];
      float *grv = &gradReliability[feature][scale][elem];

      dsigmoid(grv, rv, giv);

      mul(giv, grv, kSlope*kSlope*scalef);
    }

    //add gradient of second, masked pathway
    SCALE_LOOP(scale, kS) {
      // add
      const float *sv = &selection[feature][scale][elem];
      const float *gov = &gradOutput[feature][scale][elem];
      float *giv = &gradInput[feature][scale][elem];

      *giv += scalef * ((*sv) * (*gov));
    }

    //TODO: Update weights here
    //TODO: Update bias here
  }
}

void THNN_CudaParametricMin_updateGradInput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *thresholds, THCudaTensor *weights,
  THCudaTensor *reliability, THCudaTensor *selection,
  THCudaTensor *gradOutput, THCudaTensor *gradInput,
  THCudaTensor *gradReliability, THCudaTensor *gradSelection,
  float slope, float scale)
{
  // Resize and initialize result tensor.
  THCudaTensor_resizeAs(state, gradInput, input);
  //THCudaTensor_zero(state, gradInput);

  // int nFeatures;
  int nScales;
  int nBatches;
  int inputHeight;
  int inputWidth;

 MINMAX_assertSameGPU(state, 5, input, thresholds, weights, reliability, selection);
 MINMAX_assertSameGPU(state, 5, input, gradOutput, gradInput, gradReliability, gradSelection);

 if (THCudaTensor_nDimension(state, input) == 4)
 {
   /* sizes */
  //  nFeatures   = 1;
   nScales     = THCudaTensor_size(state, input, 0);
   nBatches    = THCudaTensor_size(state, input, 1);
   inputHeight = THCudaTensor_size(state, input, 2);
   inputWidth  = THCudaTensor_size(state, input, 3);
 }
 else if (THCudaTensor_nDimension(state, input) == 5)
 {
  //  nFeatures   = THCudaTensor_size(state, input, 0);
   nScales     = THCudaTensor_size(state, input, 1);
   nBatches    = THCudaTensor_size(state, input, 2);
   inputHeight = THCudaTensor_size(state, input, 3);
   inputWidth  = THCudaTensor_size(state, input, 4);
 }
 else
 {
   THArgCheck(false, 2, "4D or 5D tensor expected");
 }

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  input = THCudaTensor_newContiguous(state, input);

  // View 4d input as 5d with leading single dimension
  if (THCudaTensor_nDimension(state, input) == 4)
  {
    THCudaTensor_resize5d(state, input, 1, nScales, nBatches, inputHeight, inputWidth);
  }
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_resizeAs(state, gradReliability, input);
  THCudaTensor_resizeAs(state, gradSelection, input);

  // Collapse batch, height and width dimensions
  THCDeviceTensor<float, 3> cudaInput  = devicetensor<3>(state, input);
  THCDeviceTensor<float, 1> cudaThresholds = devicetensor<1>(state, thresholds);
  THCDeviceTensor<float, 2> cudaWeights = devicetensor<2>(state, weights);
  THCDeviceTensor<float, 3> cudaReliability = devicetensor<3>(state, reliability);
  THCDeviceTensor<float, 3> cudaSelection = devicetensor<3>(state, selection);
  THCDeviceTensor<float, 3> cudaGradOutput = devicetensor<3>(state, gradOutput);
  THCDeviceTensor<float, 3> cudaGradInput = devicetensor<3>(state, gradInput);
  THCDeviceTensor<float, 3> cudaGradReliability = devicetensor<3>(state, gradReliability);
  THCDeviceTensor<float, 3> cudaGradSelection = devicetensor<3>(state, gradSelection);

  long totalElements = THCudaTensor_nElement(state, input) / nScales;

  cuda_ParametricMin_updateGradInput<<<
    GET_BLOCKS(totalElements), CUDA_NUM_THREADS,
    0, THCState_getCurrentStream(state)>>>(
    cudaInput, cudaThresholds, cudaWeights,
    cudaReliability, cudaSelection,
    cudaGradOutput, cudaGradInput,
    cudaGradReliability, cudaGradSelection,
    slope, totalElements, scale);

  // cleanup
  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, input);
}

// void THNN_CudaParametricMin_accGradParameters(
//   THCState *state,
//   THCudaTensor *reliability,
//   THCudaTensor *gradReliability, THCudaTensor *gradSelection,
//   THCudaTensor *gradThresholds, THCudaTensor *gradWeights,
//   float scale, bool updateWeights)
// {
//   MINMAX_assertSameGPU(state, 5, reliability, gradReliability, gradSelection, gradThresholds, gradWeights);
//
//   int nScales = THCudaTensor_size(state, reliability, 1);
//   long totalElements = THCudaTensor_nElement(state, reliability) / nScales;
//
//   if (updateWeights) {
//
//     THCudaTensor_transpose(state, reliability, NULL, 0, 1);
//     THCudaTensor_transpose(state, gradSelection, NULL, 0, 1);
//
//     THCudaTensor_view(state, reliability, nScales, totalElements);
//     THCudaTensor_view(state, gradReliability, nScales, totalElements);
//
//     THCudaTensor_
//
//     THCDeviceTensor<float, 2> cudaGradWeights = devicetensor<2>(state, gradWeights);
//     THCDeviceTensor<float, 2> cudaReliability = devicetensor<2>(state, reliability);
//     THCDeviceTensor<float, 2> cudaGradReliability = devicetensor<2>(state, gradReliability);
//   }
//
//   THCudaTensor_mul
//
//
//
//   cuda_ParametricMin_accGradParameters<<<
//     GET_BLOCKS(totalElements), CUDA_NUM_THREADS,
//     0, THCState_getCurrentStream(state)>>>(
//       cudaReliability, cudaGradReliability, cudaGradSelection,
//       cudaGradWeights, cudaGradThresholds, scale
//     )
// }
