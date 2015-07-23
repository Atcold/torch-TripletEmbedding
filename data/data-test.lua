torch.manualSeed(0)
math.randomseed(0)

data = require 'data'

-- Shuffling training data
data.select('train') -- or 'test'

-- Initialise embeddings
embDim = 5
data.initEmbeddings(embDim)

-- Get train and test number of batches
batchDim = 10
trainBatches = data.getNbOfBatches(batchDim).train
print('trainBatches: ', trainBatches)

-- Get training batch nb 1
batch = data.getBatch(1, 'train')
print('batch = {aImg, pImg, nImg}: ', batch)

-- Send batch to cuda
collectgarbage()
data.toCuda(batch)
print('Batch sent to GPU memory')

-- Saving embeddings
data.saveEmb(torch.randn(batchDim, embDim), 1, 'train')
print('Embedding saved for fast training')


-- Snippet from the training script, provided for reference only
--[[
if opt.negPopMode == 'soft-neg' then
   -- Fetch triplet
   x = data.getBatch(batchNb, 'train', 'soft-neg1')
   if opt.cuda then data.toCuda(x) end
   -- Update embeddings
   local anchorsEmb = model.modules[1]:forward(x[1])
   local positiveEmb = model.modules[2]:forward(x[2])
   data.saveEmb(anchorsEmb, batchNb, 'train', positiveEmb)
   -- Fetch new triplet
   x = data.getBatch(batchNb, 'train', 'soft-neg2')
elseif opt.negPopMode == 'hard-neg' then
   -- Fetch triplet
   x = data.getBatch(batchNb, 'train')
   if opt.cuda then data.toCuda(x) end
   -- Update embeddings
   local anchorsEmb = model.modules[1]:forward(x[1])
   data.saveEmb(anchorsEmb, batchNb, 'train')
   -- Fetch new triplet
   x = data.getBatch(batchNb, 'train')
elseif opt.negPopMode == 'fast-hard-neg' then
   -- Fetch triplet
   x = data.getBatch(batchNb, 'train')
end
if opt.cuda then data.toCuda(x) end

y = model:forward(x)

if opt.negPopMode == 'fast-hard-neg' then
   data.saveEmb(y[1], batch, 'train')
end
--]]
