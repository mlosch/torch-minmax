local minmaxtest = torch.TestSuite()
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}

--e.g.: th -lcunn -e "nn.testcuda{'Sigmoid_forward'}"

local function pointwise_forward(proto_module, name, max_error)
   local size = math.random(1,100)

   local tm = {}
   local title = string.format(name..'.forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   if name == 'Sqrt' then input:abs() end
   local sconv = proto_module
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = proto_module:clone():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), max_error, 'error on state (forward) ')
end

local function pointwise_backward(proto_module, name, max_error)
   local size = math.random(1,100)

   local tm = {}
   local title = string.format(name..'.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   if name == 'Sqrt' then input:abs() end
   local gradOutput = torch.randn(size)
   local sconv = proto_module
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = proto_module:clone():cuda()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), max_error, 'error on state (backward) ')
end

local function pointwise_transposed(proto_module, name, max_error)
   max_error = max_error or 1e-7
   local tm = {}
   local title = name .. '.transposed'
   times[title] = tm

   local input = torch.Tensor(11, 19):uniform(-1, 1)
   if name == 'Sqrt' then
      input:uniform(0.1, 1)
   end
   local inputCUDA = input:clone():cuda()

   local cuda_module = proto_module:clone():cuda()

   -- transpose the inputs and DON'T make contiguous
   input = input:transpose(1, 2)
   inputCUDA = inputCUDA:transpose(1, 2)

   local output = proto_module:forward(input)
   local outputCUDA = cuda_module:forward(inputCUDA)

   local error = outputCUDA:float() - output
   mytester:assertlt(error:abs():max(), max_error, 'error on state (forward) ')

   local gradOutput = torch.Tensor(11, 19):uniform(-1, 1)
   local gradOutputCUDA = gradOutput:clone():cuda()

   gradOutput = gradOutput:transpose(1, 2)
   gradOutputCUDA = gradOutputCUDA:transpose(1, 2)

   local gradInput = proto_module:backward(input, gradOutput)
   local gradInputCUDA  = cuda_module:backward(inputCUDA, gradOutputCUDA)

   local error = gradInputCUDA:float() - gradInput
   mytester:assertlt(error:abs():max(), max_error,  'error on state (backward) ')
end

function minmaxtest.MinMaxPooling_forward()
   local kT = math.random(3, 7) -- featues
   local kH = 1 --math.random(3, 7)
   local kW = 1 --math.random(3, 7)
   local dT = 1 --math.random(1, 13)
   local dH = 1 --math.random(1, 13)
   local dW = 1 --math.random(1, 13)
   local iT = math.random(kT*2, 60)
   local iH = math.random(kH*2, 60)
   local iW = math.random(kW*2, 60)
   local padT = 0 --math.random(0,kT/2-1)
   local padH = 0 --math.random(0,kH/2-1)
   local padW = 0 --math.random(0,kW/2-1)
   local iF = math.random(1, 16) -- bachsize
   local oT = math.floor((iT - kT + 2*padT) / dT + 1)
   local oH = math.floor((iH - kH + 2*padH) / dH + 1)
   local oW = math.floor((iW - kW + 2*padW) / dW + 1)

   local tm = {}
   local title = string.format('MinMaxPooling.forward %dx%dx%dx%d o %dx%dx%d (%dx%dx%d)-> %dx%dx%dx%d',
                           iF, iT, iH, iW, kT, kH, kW, dT, dH, dW, iF, oT, oH, oW)
   times[title] = tm

   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.MinMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH):float()
   local output = layer:forward(input)
   local timer = torch.Timer()
   for i = 1,nloop do
      output = layer:forward(input)
   end
   tm.cpu = timer:time().real

   local inputCUDA = input:cuda()
   local layerCUDA = layer:clone():cuda()
   local outputCUDA = layerCUDA:forward(inputCUDA)
   timer:reset()
   for i = 1,nloop do
      outputCUDA = layerCUDA:forward(inputCUDA)
   end
   cutorch.synchronize()
   tm.gpu = timer:time().real

   local error = outputCUDA:float() - output
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function minmaxtest.MinMaxPooling_backward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local iT = math.random(kT*2, 60)
   local iH = math.random(kH*2, 60)
   local iW = math.random(kW*2, 60)
   local padT = math.random(0,kT/2-1)
   local padH = math.random(0,kH/2-1)
   local padW = math.random(0,kW/2-1)
   local iF = math.random(1, 16) -- features
   local oT = math.floor((iT - kT + 2*padT) / dT + 1)
   local oH = math.floor((iH - kH + 2*padH) / dH + 1)
   local oW = math.floor((iW - kW + 2*padW) / dW + 1)

   local tm = {}
   local title = string.format('VolumetricMaxPooling.backward %dx%dx%dx%d o %dx%dx%d (%dx%dx%d) -> %dx%dx%dx%d',
                               iF, iT, iH, iW, kT, kH, kW, dT, dH, dW, iF, oT, oH, oW)
   times[title] = tm

   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH):float()
   local output = layer:forward(input)
   local gradOutput = output:clone():uniform(-1, 1)

   local gradInput = layer:backward(input, gradOutput)
   local timer = torch.Timer()
   for i = 1,nloop do
      gradInput = layer:backward(input, gradOutput)
   end
   tm.cpu = timer:time().real

   local inputCUDA = input:cuda()
   local layerCUDA = layer:clone():cuda()
   local outputCUDA = layerCUDA:forward(inputCUDA)
   local gradOutputCUDA = gradOutput:cuda()
   local gradInputCUDA = layerCUDA:backward(inputCUDA, gradOutputCUDA)

   timer:reset()
   for i = 1,nloop do
      gradInputCUDA = layerCUDA:backward(inputCUDA, gradOutputCUDA)
   end
   cutorch.synchronize()
   tm.gpu = timer:time().real

   local error = gradInputCUDA:float() - gradInput
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (backward) ')
end

function minmaxtest.getParameters()
  -- tensors are non-contiguous but compact; they can be gathered
  local L = nn.Linear(10,10):cuda()
  L.weight = torch.CudaTensor(10,10):t():fill(1)
  local tmp = torch.CudaTensor(10,10):fill(2)
  L.bias = tmp:select(1,2)
  local P = L:getParameters()
  mytester:asserteq(L.weight:mean(), 1)
  mytester:asserteq(L.bias:mean(), 2)
  mytester:asserteq(L.weight:storage(), L.bias:storage())
  mytester:asserteq(P:nElement(), 110)
  mytester:asserteq(P:storage():size(), 110)
  mytester:assertlt(L.bias[{ {10} }]:storageOffset() - 1, L.bias:storage():size())
end

local function setUp()
   cutorch.setDevice(1)
end

for k,v in pairs(minmaxtest.__tests) do
   minmaxtest.__tests[k] = function()
      setUp()
      v()
   end
end

local function initSeed(seed)
   seed = seed or os.time()
   -- ensure that you can reproduce a failing test
   print('seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   cutorch.manualSeedAll(seed)
end

function nn.testcuda(tests, print_timing, n_loop, seed)
   nloop = n_loop or nloop
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   initSeed(seed)
   mytester = torch.Tester()
   mytester:add(minmaxtest)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
   if print_timing then
       print ''
       print ' ------------------------------------------------------------------------------------------------'
       print '|  Module                                                                          |  Speedup    |'
       print ' ------------------------------------------------------------------------------------------------'
       for module,tm in pairs(times) do
           local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
           print(str)
       end
       print ' ------------------------------------------------------------------------------------------------'
   end
end

-- add alias, in same format as eg cutorch.test()
cunn = cunn or {}
cunn.test = nn.testcuda
