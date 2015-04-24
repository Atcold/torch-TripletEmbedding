--------------------------------------------------------------------------------
-- Test function for TripletLoss
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr 15
--------------------------------------------------------------------------------

require 'nn'
require 'TripletEmbedding'
colour = require 'trepl.colorize'
local b = colour.blue

torch.manualSeed(0)

batch = 3
embeddingSize = 5

-- Ancore
a = torch.rand(batch, embeddingSize)
print(b('ancore:')); print(a)
-- Positive
p = torch.rand(batch, embeddingSize)
print(b('positive:')); print(p)
-- Negativep
n = torch.rand(batch, embeddingSize)
print(b('negative:')); print(n)


loss = nn.TripletEmbeddingCriterion(.2)
print(colour.red('loss: '), loss:forward({a, p, n}), '\n')
print(b('gradInput[1]:')); print(loss:backward({a, p, n})[1])

-- Jacobian test
d = 1e-6
jacobian = torch.zeros(a:size())

for i = 1, a:size(1) do
   for j = 1, a:size(2) do
      pert = torch.zeros(a:size())
      pert[i][j] = d
      outA = loss:forward({a - pert, p, n})
      outB = loss:forward({a + pert, p, n})
      jacobian[i][j] = (outB - outA)/(2*d)
   end
end

print(b('jacobian:')); print(jacobian)
