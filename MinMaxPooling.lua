local MinMaxPooling, parent = torch.class('nn.MinMaxPooling', 'nn.Module')
function MinMaxPooling:__init(thresholds, kT, kW, kH, dT, dW, dH, padT, padW, padH)
   parent.__init(self)

   dT = dT or kT
   dW = dW or kW
   dH = dH or kH

   self.kT = kT
   self.kH = kH
   self.kW = kW
   self.dT = dT
   self.dW = dW
   self.dH = dH

   self.padT = padT or 0
   self.padW = padW or 0
   self.padH = padH or 0


  self.thresholds = torch.Tensor(thresholds)
  self.mask = torch.Tensor()
  self.indices = torch.Tensor()

  self.dimension = 2
  self.running_mean = nil
  self.running_var = nil
  self.running_std = nil
  self.momentum = 0.1
  self.eps = 1e-3

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
  self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean

  local unbiased_var = sum / (nel - 1)
  self.running_var = self.momentum * unbiased_var + (1 - self.momentum) * self.running_var

  self.running_std = torch.sqrt(self.running_var + self.eps)

end

function MinMaxPooling:updateOutput(input)
  --print(input:size(), input:size(self.dimension))

  assert(input:size(self.dimension) == self.thresholds:size(1))

  self:_lazyInit(input)

  -- only update mean, std and thresholds during training
  if self.train then
    self:updateMeanStd(input)
    self.thresholds = self.running_mean - 2*self.running_std
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
