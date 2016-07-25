require 'xlua'
require 'optim'
require 'nn'
dofile './provider.lua'
lapp = require 'pl.lapp'
local tablex = require 'pl.tablex'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.0)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
   --nWorkers                 (default 2)            number of EASGD workers
   --tau                      (default 5)            tau for EASGD
   --alpha                    (default 0.001)        alpha for EASGD
]]

print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

print( '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
model:add(cast(dofile('models/'..opt.model..'.lua')))
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end

print(model)

print( '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)

-------------------------------------------------------

W, gW = model:getParameters()

local worker = {}
for i = 1, opt.nWorkers do
    worker[i] = {}
    worker[i].model = model:clone()
    worker[i].W, worker[i].gW = worker[i].model:getParameters()
end


print('==>' ..' setting criterion')
criterion = cast(nn.CrossEntropyCriterion())


print('==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

for i=1, opt.nWorkers do
    worker[i].state = tablex.deepcopy(optimState)
end

----------------------------------------------------------

function train()
    model:training() -- sets batchnorm and dropout into training mode. Modes: :training() / :evaluate()
    for i=1, opt.nWorkers do
        worker[i].model:training()
    end
    epoch = epoch or 1

    -- drop learning rate every "epoch_step" epochs
    if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end

    print( '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    local indices = {}
    for i=1, opt.nWorkers do
        indices[i] = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
        -- remove last element so that all the batches have equal size
        indices[i][#indices[i]] = nil
    end

    local tic = torch.tic()

    local current = 1
    local max = #indices[1]

    while current <= max do
        xlua.progress(current, max)

        for i = 1, opt.nWorkers do
                    -- elastic cost
            if current % opt.tau == 0 then
                local diff = opt.alpha * (worker[i].W - W)
                worker[i].W:add(-1, diff)
                W:add(diff)
            end

            local feval = function(x)
                if x ~= worker[i].W then worker[i].W:copy(x) end
                worker[i].gW:zero()

                if current > max then return end
                local inputs = provider.trainData.data:index(1, indices[i][current])
                local targets =  cast(provider.trainData.labels:index(1, indices[i][current]))
                current = current + 1

                local outputs = worker[i].model:forward(inputs)
                local f = criterion:forward(outputs, targets)
                local df_do = criterion:backward(outputs, targets)
                worker[i].model:backward(inputs, df_do)

                -- confusion:batchAdd(outputs, targets)

                return f, worker[i].gW
            end
            optim.sgd(feval, worker[i].W, worker[i].state)
        end

    end

    confusion:updateValids()
    print(('Train accuracy: '..'%.2f'..' %%\t time: %.2f s'):format(
            confusion.totalValid * 100, torch.toc(tic)))

    train_acc = confusion.totalValid * 100

    confusion:zero()
    epoch = epoch + 1
end


function test(model)
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    print( '==>'.." testing")
    local bs = 125
    for i=1,provider.testData.data:size(1),bs do
        local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
        confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
    end

    confusion:updateValids()
    print('Test accuracy:', confusion.totalValid * 100)

    -- save model every 50 epochs
    if epoch % 50 == 0 then
        local filename = paths.concat(opt.save, 'model.net')
        print('==> saving model to '..filename)
        torch.save(filename, model:get(3):clearState())
    end

    confusion:zero()
end


for i=1,opt.max_epoch do
    train()
    test(model)
    for i=1, opt.nWorkers do
        test(worker[i].model)
    end
end
