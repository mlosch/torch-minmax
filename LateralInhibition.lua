local LateralInhibition, parent = torch.class('nn.LateralInhibition', 'nn.Module')
local THNN = require 'nn.THNN'

function LateralInhibition:__init(nFeatures, affineWeights, debug)
  parent.__init(self)

  self.affine = affineWeights or false
  self.debug = debug or false
  self.bias = torch.Tensor(nFeatures) --torch.linspace(-1,0,nFeatures)

  if self.affine then
     self.weight = torch.eye(nFeatures)
     self.gradWeight = torch.zeros(nFeatures, nFeatures)
  else
     self.defaultW = torch.zeros(nFeatures, nFeatures)
     for i=2,nFeatures do
       for j=1,i-1 do
         self.defaultW[i][j] = -1
       end
     end
  end

  self.slope = 10.0

  self.output = torch.Tensor()

  self.reliability = torch.Tensor()
  self.selection = torch.Tensor()

  self.inner_input = torch.Tensor()
  self.outer_input = torch.Tensor()

  self.gradReliability = torch.Tensor()
  self.gradSelection = torch.Tensor()
  self.gradBias = torch.zeros(nFeatures)

  self.eye = torch.eye(nFeatures)

  self.recompute_backward = true

  self:reset()
end

function LateralInhibition:reset(stdv)
  if stdv then
     stdv = stdv * math.sqrt(3)
  else
   --   stdv = 1./math.sqrt(self.bias:size(1))
     stdv = 1./math.sqrt(self.bias:nElement()*2)
  end
  self.bias:apply(function()
     return torch.uniform(-stdv, stdv)
  end)

  if self.weight then
     self.weight:fill(1):add(-1,self.eye)
     self.weight:apply(function(x)
        return x*torch.uniform(-stdv, stdv)
     end)
     self.weight:add(self.eye)

     -- self.weight:apply(function()
     --    return torch.uniform(-stdv, stdv)
     -- end)
  end
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

function LateralInhibition:updateOutput(input)
  self.recompute_backward = true
  input = makeContiguous(self, input)

  self.output:resizeAs(input)

  -- set threshold of largest scale to -inf, so that reliability == 1
  -- self.bias[self.bias:size(1)] = -math.huge

  -- h(x) = z * sigma( -c/2 + cW * sigma( -c* (z - t) ) )
  local c = self.slope
  local bias = self.bias
  local K = bias:size(1)
  local N = input:numel() / input:size(2)
  --self.inner_input = input:transpose(1,2):reshape(K, N)
  nn.utils.contiguousView(self.inner_input, input:transpose(1,2), K, -1)

  --self.inner_input:add(-1, torch.expand(bias:view(K,1), K, N))
  self.inner_input:add(-1, torch.expandAs(bias:view(K,1), self.inner_input))
  self.inner_input:mul(c)

  -- print('inner_input:size()', self.inner_input:size())

  input.THNN.Sigmoid_updateOutput(
    self.inner_input:cdata(),
    self.reliability:cdata()
  )

  -- print('reliability:size()', self.reliability:size())

  local W = self.weight or self.defaultW
  -- set diagonal to 1s
  W:addcmul(-1, W, self.eye):add(self.eye)

  self.outer_input = torch.mm(W, self.reliability):add(-0.5):mul(c)
  -- print('outer_input:size()', self.outer_input:size())
  input.THNN.Sigmoid_updateOutput(
    self.outer_input:cdata(),
    self.selection:cdata()
  )

  -- print('selection:size()', self.selection:size())

  local sz = input:size()
  self.output:resizeAs(input):copy(input)
  self.output:cmul(self.selection:view(sz[2],sz[1],sz[3],sz[4]):transpose(1,2))
  --self.output:copy(self.selection:view(sz[2],sz[1],sz[3],sz[4]):transpose(1,2))
  return self.output

end

function LateralInhibition:backwards(input, gradOutput, scale, gradInput, gradBias, gradWeight)
  self.recompute_backward = false
  input, gradOutput = makeContiguous(self, input, gradOutput)

  scale = scale or 1
  if gradInput then
    self.gradInput:resizeAs(gradOutput)
  end
  if gradBias then
    self.gradBias:resizeAs(self.bias)
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
      self.outer_input:cdata(),
      gradOutput_t:cdata(),
      self.gradSelection:cdata(),
      self.selection:cdata()
    )

    local W = self.weight or self.defaultW
    local grads = torch.mm(W:t(), self.gradSelection)

    input.THNN.Sigmoid_updateGradInput(
      self.inner_input:cdata(),
      grads:cdata(),
      self.gradReliability:cdata(),
      self.reliability:cdata()
    )

    grads = self.gradReliability:view(K, B, -1):transpose(2,1)
    self.gradInput:copy(grads):mul(c*c*scale)
    self.gradInput:add(scale, torch.cmul(gradOutput, self.selection:view(sz[2],sz[1],sz[3],sz[4]):transpose(1,2) ))
  end

  -- update weight
  if self.gradWeight then
     local grads = torch.mm(self.gradSelection, self.reliability:t())
     self.gradWeight:copy(grads):mul(scale*c)
  end

  -- update bias
  if gradBias then
    self.gradReliability:mul(-c*c*scale)
    self.gradBias:add(self.gradReliability:sum(2):squeeze())
  end

  return self.gradBias
end

function LateralInhibition:updateGradInput(input, gradOutput)
  -- if self.recompute_backward then
    self:backwards(input, gradOutput, 1, self.gradInput)
  -- end
  return self.gradInput
end

function LateralInhibition:accGradParameters(input, gradOutput, scale)
  -- if self.recompute_backward then
    self:backwards(input, gradOutput, scale, nil, self.gradBias, self.gradWeight)
  -- end
  return self.gradBias
end

function LateralInhibition:clearState()
  self.gradReliability:set()
  self.gradSelection:set()
  self.inner_input:set()
  self.outer_input:set()
  self.selection:set()
  self.reliability:set()
  self.output:set()
end
