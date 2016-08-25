local MinMaxPooling, parent = torch.class('nn.MinMaxPooling', 'nn.Module')
local THNN = require 'nn.THNN'

function MinMaxPooling:__init(thresholds, gamma, nFeatures, batchNormed)
  parent.__init(self)

  assert(#thresholds == 2 and
    type(thresholds[1]) == 'table' and
    type(thresholds[2]) == 'table' and
    #thresholds[1] == #thresholds[2])

  self.kT = nFeatures
  self.kH = 1
  self.kW = 1
  self.dT = 1
  self.dW = 1
  self.dH = 1

  self.padT = 0
  self.padW = 0
  self.padH = 0

  self.batchNormed = batchNormed or false

  self.gamma = gamma
  self.thresholds = torch.Tensor(thresholds)
  self.mask = torch.Tensor()
  self.indices = torch.Tensor()

  self.dimension = 2
  self.running_mean = nil
  self.running_var = nil
  self.running_std = nil
  self.momentum = 0.1
  self.eps = 1e-5

  self.train = true
  self.ceil_mode = false
end


function MinMaxPooling:_lazyInit(input)
  local nscales = input:size(self.dimension)
  self.running_mean = self.running_mean or (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor(nscales):fill(0) or torch.Tensor(nscales):fill(0))
  self.running_var = self.running_var or (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor(nscales):fill(1) or torch.Tensor(nscales):fill(1))
  self.running_std = self.running_std or (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor(nscales):fill(0) or torch.Tensor(nscales):fill(0))
end


function MinMaxPooling:updateMeanStd(input)
  -- transform input to shape: 3 x nel
  local input_t = input:transpose(self.dimension,1)
  local nel = input_t:nElement() / input_t:size(1)
  input_t = input_t:reshape(input_t:size(1), nel)

  -- calculate mean of input
  local mean = input_t:mean(2):squeeze()

  -- calculate variance of input
  local sum = mean:clone():fill(0)

  for i=1,sum:size(1) do
    sum[i] = input_t[i]:add(-mean[i]):pow(2):sum()
  end

  -- update running averages
  self.running_mean = torch.add(torch.mul(mean ,self.momentum), torch.mul(self.running_mean, (1 - self.momentum)))

  local unbiased_var = sum / (nel - 1)
  self.running_var = torch.add(torch.mul(unbiased_var, self.momentum), torch.mul(self.running_var, (1 - self.momentum)))

  self.running_std = torch.sqrt(torch.add(self.running_var, self.eps))

end

local function makeContiguous(self, input, gradOutput)
  if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
  end
  if gradOutput then
    if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
    end
  end
  return input, gradOutput
end

-- function MinMaxPooling:CPU_forward(input)
--    local input_t = input:transpose(self.dimension,1)
--    local nel = input_t:nElement() / input_t:size(1)
--    local nF = input_t:size(1)
--    input_t = input_t:reshape(input_t:size(1), nel)
--
--    local mask = input_t.new():resizeAs(input_t):fill(0)
--
--    for i=1,nel do
--       local min = nF
--       for k=1,nF-1 do
--          if --TODO
--       end
--    end
--
-- end

function MinMaxPooling:updateOutput(input)
  assert(input:size(self.dimension) == self.thresholds:size(2))

  input = makeContiguous(self, input)

  -- only update mean, std and thresholds during training
  if self.train then
    if not self.batchNormed then
      self:_lazyInit(input)
      self:updateMeanStd(input)
      self.thresholds[1] = torch.add(self.running_mean, -torch.mul(self.running_std, self.gamma))
      self.thresholds[2] = torch.add(self.running_mean, torch.mul(self.running_std,self.gamma))
    else
      -- batchnorm output is mu(x) == 0 and var(x) == 1
      self.thresholds[1]:fill(-self.gamma)
      self.thresholds[2]:fill(self.gamma)
    end
  end

  self.indices = self.indices or input.new()
  self.mask = self.mask or input.new()

  input.THNN.MinMaxPooling_updateOutput(
    input:cdata(),
    self.thresholds:cdata(),
    self.mask:cdata(),
    self.output:cdata(),
    self.indices:cdata(),
    self.kT, self.kW, self.kH,
    self.dT, self.dW, self.dH,
    self.padT, self.padW, self.padH,
    self.ceil_mode
  )

  return self.output

end

function MinMaxPooling:updateGradInput(input, gradOutput)
  input.THNN.MinMaxPooling_updateGradInput(
    input:cdata(),
    self.mask:cdata(),
    gradOutput:cdata(),
    self.gradInput:cdata(),
    self.indices:cdata(),
    self.dT, self.dW, self.dH,
    self.padT, self.padW, self.padH
  )
   return self.gradInput
end
