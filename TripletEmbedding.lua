--------------------------------------------------------------------------------
-- TripletEmbeddingCriterion
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr/May 15
--------------------------------------------------------------------------------

local TripletEmbeddingCriterion, parent = torch.class('nn.TripletEmbeddingCriterion', 'nn.Criterion')

function TripletEmbeddingCriterion:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 0.2
   self.Li = torch.Tensor()
   self.gradInput = {}
end

function TripletEmbeddingCriterion:updateOutput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.Li:resize(N)
   for i = 1, N do
      self.Li[i] = math.max(0, (a[i]-p[i])*(a[i]-p[i])+self.alpha-(a[i]-n[i])*(a[i]-n[i]))
      --print(self.Li[i])
   end
   self.output = self.Li:sum() / N
   return self.output
end

function TripletEmbeddingCriterion:updateGradInput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   if torch.type(a) == 'torch.CudaTensor' then -- if buggy CUDA API
      self.gradInput[1] = (n - p):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
      self.gradInput[2] = (p - a):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
      self.gradInput[3] = (a - n):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
   else -- otherwise
      self.gradInput[1] = self.Li:gt(0):diag():type(a:type()) * (n - p) * 2/N
      self.gradInput[2] = self.Li:gt(0):diag():type(a:type()) * (p - a) * 2/N
      self.gradInput[3] = self.Li:gt(0):diag():type(a:type()) * (a - n) * 2/N
   end
   return self.gradInput
end
