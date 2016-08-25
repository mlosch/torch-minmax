local LateralInhibitionv2, parent = torch.class('nn.LateralInhibitionv2', 'nn.Module')
local THNN = require 'nn.THNN'

function LateralInhibitionv2:__init(nFeatures, nScales, affineWeights, debug)
  parent.__init(self)

  self.nFeatures = nFeatures
  self.nScales = nScales

  self.affine = affineWeights or false
  self.debug = debug or false
  self.bias = torch.Tensor(nFeatures, nScales) --torch.linspace(-1,0,nFeatures)

  if self.affine then
     self.weight = torch.zeros(nFeatures, nScales, nScales)
     for m=1,nFeatures do
        self.weight[m]:eye(nScales)
     end
     self.gradWeight = torch.zeros(nFeatures, nScales, nScales)
  else
     self.defaultW = torch.zeros(nFeatures, nScales, nScales)
     for m=1,nFeatures do
        local defaultW = self.defaultW[m]
        for i=2,nScales do
          for j=1,i-1 do
             defaultW[i][j] = -1
          end
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
  self.gradBias = torch.zeros(nFeatures, nScales)

  self._eye = torch.eye(nScales)
  self._ones = torch.Tensor()

  self.recompute_backward = true

  self:reset()
end

function LateralInhibitionv2:reset(stdv)
  if stdv then
     stdv = stdv * math.sqrt(3)
  else
   --   stdv = 1./math.sqrt(self.bias:size(1))
     stdv = 1./math.sqrt(self.nScales * self.nScales)
  end

  self.bias:uniform(-stdv, stdv)

  if self.weight then
     for m=1,self.nFeatures do
        local W = self.weight[m]
        W:fill(1):add(-1,self._eye)
        W:apply(function(x)
           return x*torch.uniform(-stdv, stdv)
        end)
        W:add(self._eye)
     end

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

-- nf x 4 x b x w x h
function LateralInhibitionv2:updateOutput(input)
  self.recompute_backward = true
  input = makeContiguous(self, input)

  assert(input:size(1) == self.nFeatures)
  assert(input:size(2) == self.nScales)

  self.output:resizeAs(input)

  -- set threshold of largest scale to -inf, so that reliability == 1
  -- self.bias[self.bias:size(1)] = -math.huge

  -- h(x) = z * sigma( -c/2 + cW * sigma( -c* (z - t) ) )
  local c = self.slope
  local nF = self.nFeatures
  local nS = self.nScales
  --self.inner_input = input:transpose(1,2):reshape(K, N)
  nn.utils.contiguousView(self.inner_input, input, nF, nS, -1)

  if not self._ones:isSameSizeAs(self.inner_input) then
     self._ones:resizeAs(self.inner_input):fill(1)
  end

  -- expand bias to batches of diagonal matrices
  local bias = input.new():resize(nF, nS, nS):fill(0)
  for m=1,self.nFeatures do
     local bt = bias[m]
     local bs = self.bias[m]
     for s=1,self.nScales do
        bt[{s,s}] = bs[s]
     end
  end

  self.inner_input:baddbmm(-1, bias, self._ones)
  self.inner_input:mul(c)

  -- print('inner_input:size()', self.inner_input:size())

  input.THNN.Sigmoid_updateOutput(
    self.inner_input:cdata(),
    self.reliability:cdata()
  )

  -- print('reliability:size()', self.reliability:size())

  local W = self.weight or self.defaultW
  -- set diagonal to 1s
  for m=1,self.nFeatures do
    local Wm = W[m]
    Wm:addcmul(-1, Wm, self._eye):add(self._eye)
  end

  self.outer_input = torch.bmm(W, self.reliability):add(-0.5):mul(c)
  -- print('outer_input:size()', self.outer_input:size())
  input.THNN.Sigmoid_updateOutput(
    self.outer_input:cdata(),
    self.selection:cdata()
  )

  -- print('selection:size()', self.selection:size())

  local sz = input:size()
  self.output:resizeAs(input):copy(input)
  self.output:cmul(self.selection:viewAs(input))
  --self.output:copy(self.selection:view(sz[2],sz[1],sz[3],sz[4]):transpose(1,2))
  return self.output

end

function LateralInhibitionv2:backwards(input, gradOutput, scale, gradInput, gradBias, gradWeight)
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
  local nF, nS = sz[1], sz[2]

  local gradOutput_t = torch.cmul(gradOutput, input):view(nF,nS,-1)

  if gradInput then
    input.THNN.Sigmoid_updateGradInput(
      self.outer_input:cdata(),
      gradOutput_t:cdata(),
      self.gradSelection:cdata(),
      self.selection:cdata()
    )

    local W = self.weight or self.defaultW
    local grads = torch.bmm(W:transpose(2,3), self.gradSelection)

    input.THNN.Sigmoid_updateGradInput(
      self.inner_input:cdata(),
      grads:cdata(),
      self.gradReliability:cdata(),
      self.reliability:cdata()
    )

    grads = self.gradReliability:viewAs(input)
    self.gradInput:copy(grads):mul(c*c*scale)
    self.gradInput:add(scale, torch.cmul(gradOutput, self.selection:viewAs(input)))
  end

  -- update weight
  if self.gradWeight then
     local grads = torch.bmm(self.gradSelection, self.reliability:transpose(2,3))
     self.gradWeight:copy(grads):mul(scale*c)
  end

  -- update bias
  if gradBias then
    self.gradReliability:mul(-c*c*scale)
    self.gradBias:add(self.gradReliability:sum(3):squeeze())
  end

  return self.gradBias
end

function LateralInhibitionv2:updateGradInput(input, gradOutput)
  -- if self.recompute_backward then
    self:backwards(input, gradOutput, 1, self.gradInput)
  -- end
  return self.gradInput
end

function LateralInhibitionv2:accGradParameters(input, gradOutput, scale)
  -- if self.recompute_backward then
    self:backwards(input, gradOutput, scale, nil, self.gradBias, self.gradWeight)
  -- end
  return self.gradBias
end

function LateralInhibitionv2:clearState()
  self.gradReliability:set()
  self.gradSelection:set()
  self.inner_input:set()
  self.outer_input:set()
  self.selection:set()
  self.reliability:set()
  self.output:set()
  self._ones:set()
end
