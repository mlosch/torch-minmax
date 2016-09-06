local ParametricMin, parent = torch.class('nn.ParametricMin', 'nn.Module')
local THNN = require 'nn.THNN'

function ParametricMin:__init(nScales, affineWeights, debug)
  parent.__init(self)

  self.affine = affineWeights or false
  self.debug = debug or false
  self.bias = torch.Tensor(nScales) --torch.linspace(-1,0,nScales)
  self.gradBias = torch.Tensor(nScales)

  self.nScales = nScales

  if self.affine then
     self.weight = torch.eye(nScales)
     self.gradWeight = torch.zeros(nScales, nScales)
  end

  self.slope = 10.0

  self.recompute_backward = true

  self:reset()
end

function ParametricMin:reset(stdv)
  if stdv then
     stdv = stdv * math.sqrt(3)
  else
   --   stdv = 1./math.sqrt(self.bias:size(1))
     stdv = 1./math.sqrt(self.nScales)
  end
  self.bias:apply(function()
     return torch.uniform(-stdv, stdv)
  end)

  if self.affine then
     self.weight:apply(function()
        return torch.uniform(-stdv, stdv)
     end)
  else
    self.W = torch.eye(self.nScales)
    for i=2,self.nScales do
      for j=1,i-1 do
        self.W[i][j] = -1
      end
    end
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

function ParametricMin:updateOutput(input)
  self.recompute_backward = true
  input = makeContiguous(self, input)

  self.reliability = self.reliability or input.new()
  self.selection = self.selection or input.new()

  local W = self.weight or self.W

  input.THNN.ParametricMin_updateOutput(
    input:cdata(), self.bias:cdata(),
    W:cdata(), self.output:cdata(),
    self.reliability:cdata(), self.selection:cdata(),
    self.slope
  );

  return self.output

end

function ParametricMin:backwards(input, gradOutput, scale, gradInput, gradBias, gradWeight)
  input, gradOutput = makeContiguous(self, input, gradOutput)

  self.gradReliability = self.gradReliability or input.new()
  self.gradSelection = self.gradSelection or input.new()

  scale = scale or 1

  if self.recompute_backward then

    local W = self.weight or self.W

    input.THNN.ParametricMin_updateGradInput(
      input:cdata(),
      self.bias:cdata(), W:cdata(),
      self.reliability:cdata(), self.selection:cdata(),
      gradOutput:cdata(), gradInput:cdata(),
      self.gradReliability:cdata(), self.gradSelection:cdata(),
      self.slope, scale
    )
    self.recompute_backward = false
  end

  return self.gradBias
end

function ParametricMin:updateGradInput(input, gradOutput)
  -- if self.recompute_backward then
    self:backwards(input, gradOutput, 1, self.gradInput)
  -- end
  return self.gradInput
end

function ParametricMin:accGradParameters(input, gradOutput, scale)
  --self:backwards(input, gradOutput, scale, nil, self.gradBias, self.gradWeight)
  -- for now we update bias and weight here
  if self.affine then
    local N = input:numel() / self.nScales
    local a = self.gradSelection:transpose(1,2):reshape(self.nScales, N)
    local b = self.reliability:transpose(1,2):reshape(self.nScales, N)
    self.gradWeight:copy(torch.mm(a, b:t())):mul(self.slope*scale)
  end

  self.gradBias:copy(self.gradReliability:sum(1):sum(3):sum(4):sum(5))
  self.gradBias:mul(-self.slope*self.slope*scale)
  return self.gradBias
end

function ParametricMin:clearState()
  nn.utils.clear(self, {
    'reliability',
    'selection',
    'gradReliability',
    'gradSelection'
  })
  return parent.clearState(self)
end
