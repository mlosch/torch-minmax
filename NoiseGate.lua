local NoiseGate, parent = torch.class('nn.NoiseGate', 'nn.Module')
local THNN = require 'nn.THNN'

function NoiseGate:__init(nFeatures, slope)
  parent.__init(self)

  self.nFeatures = nFeatures
  self.slope = slope or 1
  self.threshold = torch.Tensor(nFeatures)
  self.gradThreshold = torch.Tensor(nFeatures)

  self.output = torch.Tensor()

  self.reliability = torch.Tensor()

  self.inner_input = torch.Tensor()

  self.gradReliability = torch.Tensor()

  self.recompute_backward = true

  self:reset()
end

function NoiseGate:reset(stdv)
  if stdv then
     stdv = stdv * math.sqrt(3)
  else
   --   stdv = 1./math.sqrt(self.threshold:size(1))
     stdv = 1./math.sqrt(self.threshold:nElement()*2)
  end
  self.threshold:apply(function()
     return torch.uniform(-stdv, stdv)
  end)
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

function NoiseGate:updateOutput(input)
  self.recompute_backward = true
  input = makeContiguous(self, input)

  self.output:resizeAs(input)

  -- set threshold of largest scale to -inf, so that reliability == 1
  -- self.threshold[self.threshold:size(1)] = -math.huge

  -- h(x) = z * sigma( -c/2 + cW * sigma( -c* (z - t) ) )
  local c = self.slope
  local threshold = self.threshold
  local K = threshold:size(1)
  local N = input:numel() / input:size(2)
  --self.inner_input = input:transpose(1,2):reshape(K, N)
  nn.utils.contiguousView(self.inner_input, input:transpose(1,2), K, -1)

  self.inner_input:add(-1, torch.expandAs(threshold:view(K,1), self.inner_input))
  self.inner_input:mul(c)

  input.THNN.Sigmoid_updateOutput(
    self.inner_input:cdata(),
    self.reliability:cdata()
  )

  local sz = input:size()
  self.output:resizeAs(input):copy(input)
  self.output:cmul(self.reliability:view(sz[2],sz[1],sz[3],sz[4]):transpose(1,2))
  --self.output:copy(self.selection:view(sz[2],sz[1],sz[3],sz[4]):transpose(1,2))
  return self.output

end

function NoiseGate:backwards(input, gradOutput, scale, gradInput, gradThreshold)
  self.recompute_backward = false
  input, gradOutput = makeContiguous(self, input, gradOutput)

  scale = scale or 1
  if gradInput then
    self.gradInput:resizeAs(gradOutput)
  end
  if gradThreshold then
    self.gradThreshold:resizeAs(self.threshold)
  end

  local c = self.slope
  local sz = input:size()
  local B, K = sz[1], sz[2]
  -- local B = input:size(1)
  -- local K = input:size(2)
  -- local N = input:numel() / K
  -- gradOutput:cmul(input)
  --local gradOutput_t = gradOutput:transpose(1,2):reshape(K, N)

  local gradOutput_t = torch.cmul(gradOutput, input):transpose(1,2):contiguous():view(K,-1)

  if gradInput then
    input.THNN.Sigmoid_updateGradInput(
      self.inner_input:cdata(),
      gradOutput_t:cdata(),
      self.gradReliability:cdata(),
      self.reliability:cdata()
    )

    grads = self.gradReliability:view(K, B, -1):transpose(2,1)
    self.gradInput:copy(grads):mul(c*scale)
    self.gradInput:add(scale, torch.cmul(gradOutput, self.reliability:view(sz[2],sz[1],sz[3],sz[4]):transpose(1,2) ))
  end

  -- update threshold
  if gradThreshold then
    --self.gradThreshold:fill(0)
    self.gradReliability:mul(-c*scale)
    self.gradThreshold:copy(self.gradReliability:sum(2):squeeze())
  end
end

function NoiseGate:updateGradInput(input, gradOutput)
  -- if self.recompute_backward then
    self:backwards(input, gradOutput, 1, self.gradInput)
  -- end
  return self.gradInput
end

function NoiseGate:accGradParameters(input, gradOutput, scale)
  -- if self.recompute_backward then
    self:backwards(input, gradOutput, scale, nil, self.gradThreshold)
  -- end
end

function NoiseGate:updateParameters(learningRate)
   parent.updateParameters(self, learningRate)
   self.threshold:add(-learningRate, self.gradThreshold)
end

function NoiseGate:clearState()
  self.gradReliability:set()
  self.inner_input:set()
  self.reliability:set()
  self.output:set()
end
