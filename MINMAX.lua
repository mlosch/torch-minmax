local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local MINMAX = {}

-- load libMINMAX
MINMAX.C = ffi.load(package.searchpath('libMINMAX', package.cpath))

local THCState_ptr = ffi.typeof('THCState*')

function MINMAX.getState()
   return THCState_ptr(cutorch.getState());
end

local MINMAX_h = require 'minmax.MINMAX_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THNN.h
MINMAX_h = MINMAX_h:gsub("\n#[^\n]*", "")
MINMAX_h = MINMAX_h:gsub("^#[^\n]*\n", "")

local preprocessed = string.gsub(MINMAX_h, 'TH_API ', '')

local replacements =
{
   {
      ['THTensor'] = 'THCudaTensor',
      ['THIndexTensor'] = 'THCudaTensor',
      ['THIntegerTensor'] = 'THCudaTensor',
      ['THIndex_t'] = 'float',
      ['THInteger_t'] = 'float'
   }
}

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_Cuda([%a%d_]+)') do
      t[#t+1] = n
   end
   return t
end

-- build function table
local function_names = extract_function_names(MINMAX_h)

THNN.kernels['torch.CudaTensor'] = THNN.bind(MINMAX.C, function_names, 'Cuda', MINMAX.getState)
torch.getmetatable('torch.CudaTensor').THNN = THNN.kernels['torch.CudaTensor']

return MINMAX
