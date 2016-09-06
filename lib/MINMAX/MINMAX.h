#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THIndexTensor THCudaTensor
#define THIndexTensor_(NAME) THCudaTensor_ ## NAME

#define THIntegerTensor THCudaTensor
#define THIntegerTensor_(NAME) THCudaTensor_ ## NAME

TH_API void THNN_CudaMinMaxPooling_updateOutput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *thresholds, THCudaTensor *mask,
	THCudaTensor *output, THCudaTensor *indices,
	int kT, int kW, int kH,
  int dT, int dW, int dH,
  int padT, int padW, int padH,
  bool ceilMode);
TH_API void THNN_CudaMinMaxPooling_updateGradInput(
  THCState *state,
  THCudaTensor *input, THCudaTensor *mask,
  THCudaTensor *gradOutput, THCudaTensor *gradInput,
  THCudaTensor *indices,
  int dT, int dW, int dH,
  int padT, int padW, int padH);

TH_API void THNN_CudaParametricMin_updateOutput(
  THCState *state,
  THCudaTensor *input, THCudaTensor *thresholds,
  THCudaTensor *weights, THCudaTensor *output,
  THCudaTensor *reliability, THCudaTensor *selection,
  float slope);
TH_API void THNN_CudaParametricMin_updateGradInput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *thresholds, THCudaTensor *weights,
  THCudaTensor *reliability, THCudaTensor *selection,
  THCudaTensor *gradOutput, THCudaTensor *gradInput,
  THCudaTensor *gradReliability, THCudaTensor *gradSelection,
  float slope, float scale);
