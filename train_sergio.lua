require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.optim_updates'
npy4th = require 'npy4th'
-------------------------------------------------------------------------------
-- input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('MM Twitter')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_npy','/homedtic/fbarbieri/sergionose/npy/')
cmd:option('-input_code',434,'434 or 664')
cmd:option('-feat_size_audio',4096,'The number of audioual features')
cmd:option('-feat_size_visual',2048,'The number of visual features')
-- Select model
cmd:option('-model','tanh','relu or tanh')
cmd:option('-mapping',-1,'1 map visual to audio, 2 map audio to visual, other map both in a MM space (V1, A2)')
cmd:option('-crit','cosine','What criterion to use (only cosine so far)')
cmd:option('-margin',0.5,'Negative samples margin: L = max(0, cos(x1, x2) - margin)')
cmd:option('-num_layers', 1, 'number of hidden layers')
cmd:option('-hidden_size',500,'The hidden size of the discriminative layer')
cmd:option('-output_size',200,'The dimension of the output vector (shared space)')
cmd:option('-k',1,'The slope of sigmoid')
cmd:option('-scale_output',0,'Whether to add a sigmoid at teh output of the model')
-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',10,'what is the batch size in number of images per batch')
cmd:option('-batch_size_real',-1,'real value of the batch with the negative examples')
cmd:option('-neg_samples',2,'number of negative examples for each good example')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
-- Optimization: for the model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')

cmd:option('-learning_rate_decay_every', 1000, 'every how many iterations LR decay')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')

cmd:option('-optim_alpha',0.001,'alpha for adagrad|rmsprop|momentum|adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-weight_decay',0,'Weight decay for L2 norm')
-- Evaluation/Checkpointing
cmd:option('-train_size', 211475, 'how many users to use for training set')
cmd:option('-val_size', 500, 'how many users to use for validation set') --26434
cmd:option('-test_size', 64477, 'how many users to use for test set')
cmd:option('-save_checkpoint_every', -1, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'cp/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-output_path', '/homedtic/fbarbieri/sergionose/mm_vectors/', 'folder to save output vectors')
cmd:option('-save_output', -1, 'whether to save or not the output vectors')
cmd:option('-beta',1,'beta for f_x')
-- misc
cmd:option('-id', 'idcp', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-verbose',false,'How much info to give')
cmd:option('-print_every',10,'Print some statistics')
cmd:option('-revert_params',-1,'Reverst parameters if you are doing worse on the validation')
cmd:text()


------------------------------------------------------------------------------
-- Basic Torch initializations
------------------------------------------------------------------------------
local opt = cmd:parse(arg)
--opt.id = '_'..opt.model..'_h@'..opt.hidden_size..'_k@'..'_scOut@'..opt.scale_output..'_w@'..opt.weight_decay..'_lr@'..opt.learning_rate..'_dlr@'..opt.learning_rate_decay_every
print(opt)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
  print('Using GPU')
end

opt.batch_size_real = opt.batch_size + opt.batch_size * 2 * opt.neg_samples

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader
--loader = DataLoader{train_size = opt.train_size, val_size = opt.val_size, json_file = opt.input_json, h5_file_audio = opt.input_npy_audio, h5_file_visual = opt.input_npy_visual, label_format = opt.crit, feat_size_audio = opt.feat_size_audio, feat_size_visual = opt.feat_size_visual, gpu = opt.gpuid, output_size = opt.output_size}

loader = DataLoader(opt)
local feat_size_audio = loader:getFeatSizeaudio()
local feat_size_visual = loader:getFeatSizeVisual()

-------------------------------------------------------------------------------
-- Initialize the network
-------------------------------------------------------------------------------
local protos = {}
local model1 
--print(string.format('Parameters are model=%s feat_size=%d, output_size=%d\n',opt.model, feat_size,output_size))
-- create protos from scratch
if opt.model == 'relu' then
  model1 = require 'models.model1_relu'
  protos.model = model1.model(opt)
  elseif opt.model == 'tanh' then
  model1 = require 'models.model1_tanh'
  protos.model = model1.model(opt)
  else
    print(string.format('Wrong model:%s',opt.model))
  end

--add criterion
protos.criterion = nn.CosineEmbeddingCriterion(opt.margin)
protos.criterion.sizeAverage = false

-- ship protos to GPU
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
local params, grad_params = protos.model:getParameters()
params:uniform(-0.08, 0.08)
print('total number of parameters in model: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

collectgarbage()

-------------------------------------------------------------------------------
-- forward test set and save to file
-------------------------------------------------------------------------------
local function save_output_vectors(rank)
  protos.model:evaluate()

  datasets = {'train','val','test'}
  for dataCount = 1, 3 do
    dataset = datasets[dataCount]
    print (dataset)
    local size
    if dataset == 'train' then
      size = opt.train_size
    elseif dataset == 'val' then
      size = 26434 --opt.val_size --ALEEEERRRRTTTT!!!!!!!!!!!!!! im using a smaller val to be faster
    elseif dataset == 'test' then
      size = opt.test_size
    else
      error('error: unknown dataset - ' .. split)
    end

    -- initialize one tensor per modality
    local audio_out = torch.FloatTensor(size, opt.output_size)
    local visual_out = torch.FloatTensor(size, opt.output_size)
    i = 1
    for i = 1,size do
      -- forward
      local data = loader:getOneTensor(i,dataset)
      local input = {}
      table.insert(input,data.audio)
      table.insert(input,data.visual)
      local output = protos.model:forward(input)
      audio_out[i] = output[1]:float()
      --audio_out[i] = audio_out[i] / audio_out[i]:norm()
      visual_out[i] = output[2]:float()
      --visual_out[i] = visual_out[i] / visual_out[i]:norm()
    end

    --save to npy array
    local timestamp = os.clock()
    local audiofile = opt.output_path .. 'mm_' ..  opt.input_code .. '_' .. dataset .. '_' .. rank .. '_' .. size .. '_' .. opt.mapping .. '_'.. opt.output_size .. '_' .. opt.learning_rate .. '_' .. timestamp .. '.npy'
    npy4th.savenpy(audiofile, audio_out)

    local visualfile = opt.output_path .. 'mm_' .. opt.input_code .. '_' .. dataset .. '_' .. rank .. '_' .. size .. '_' .. opt.mapping .. '_'.. opt.output_size .. '_' .. opt.learning_rate .. '_' .. timestamp .. '.npy'
    npy4th.savenpy(visualfile, visual_out)

    print('A: ' .. audiofile)
    print('V: ' .. visualfile)
  end
end

-------------------------------------------------------------------------------
-- Evaluate validation set (loss)
-------------------------------------------------------------------------------
local function eval_split(split)
  protos.model:evaluate()
  local total_loss = 0
  local maxIter =  torch.floor(opt.val_size / opt.batch_size)

  for v=1,maxIter do
    -- forward
    local data = loader:getBatch(opt,split,false)
    local input = {}
    table.insert(input,data.audio)
    table.insert(input,data.visual)
    local output = protos.model:forward(input)
    local predicted = output
    local labels = data.labels
    local loss = protos.criterion:forward(predicted, labels)
    total_loss = total_loss + loss
  end

  -- local total_examples = maxIter * opt.batch_size + maxIter * opt.batch_size * 2 *  opt.neg_samples
  local total_examples = maxIter * opt.batch_size + maxIter * opt.batch_size * 2 *  opt.neg_samples

  avg_loss = total_loss / total_examples
  return avg_loss
end

------------------------------------
-- Evaluate Validation Set (Ranking, from one to the other modality)
------------------------------------
local function eval_ranking()
  protos.model:evaluate()

  local maxIter = torch.floor(opt.val_size / opt.batch_size)
  local val_size_real = maxIter * opt.batch_size

  -- initialize one table per modality
  audio_out = torch.CudaTensor(val_size_real, opt.output_size)
  visual_out = torch.CudaTensor(val_size_real, opt.output_size)

  i = 1
  for _ = 1,maxIter do
    -- forward
    local data = loader:getBatch(opt,'val',false) -- without negative samples
    local input = {}
    table.insert(input,data.audio)
    table.insert(input,data.visual)
    local output = protos.model:forward(input)
    
    -- fill the out matrixes with the batch outputs
    audio_out_batch = output[1]
    visual_out_batch = output[2]
    for j = 1,opt.batch_size do
      audio_out[i] = audio_out_batch[j]
      visual_out[i] = visual_out_batch[j]
      i = i + 1
    end
  end

  -- normalize
  local r_norm_audio = audio_out:norm(2,2)
  audio_out:cdiv(r_norm_audio:expandAs(audio_out))

  local r_norm_visual = visual_out:norm(2,2)
  visual_out:cdiv(r_norm_visual:expandAs(visual_out))

  -- cosine
  local cosine = audio_out * visual_out:transpose(1,2)

  -- trace
  local sim = cosine:float():trace() / cosine:size(1) 

  -- ranking
  local ranking_audio = torch.Tensor(val_size_real)
  local top5t=0
  for s = 1,val_size_real do
    _,index = torch.sort(cosine:select(1,s),true) -- sort rows

    local not_found = true
    local q = 1
    while not_found do
      if index[q]==s then 
        ranking_audio[s] = q
        not_found = false
        if q <= 5 then top5t = top5t+1 end 
      end
      q = q + 1
    end
  end 

  local top5v=0
  local ranking_visual = torch.Tensor(val_size_real)
  for s = 1,val_size_real do
    --print('(visual) sorting')
    _,index = torch.sort(cosine:select(2,s),true) -- sort columns

    local not_found = true
    local q = 1
    while not_found do
      if index[q]==s then 
        ranking_visual[s] = q
        not_found = false
        if q <= 5 then top5v = top5v+1 end 
      end
      q = q + 1
    end
  end 

  avgAudio = torch.median(ranking_audio)[1]
  avgVisual = torch.median(ranking_visual)[1]

  return avgAudio, avgVisual, top5t, top5v, sim
end


-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  protos.model:training() -- some flag, didnt undestand
  grad_params:zero()      -- very important to set deltaW to zero!

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data
  local data = loader:getBatch(opt,'train',true)
  -- load audio and visual vectors, and fill the input table
  local input = {}
  table.insert(input,data.audio)
  table.insert(input,data.visual)

  -- get predicted ({audio,visual})
  local output = protos.model:forward(input)
  local labels = data.labels

  -- forward the criterion
  -- COSINE: measures the loss given an input x = {x1, x2}, a table of two Tensors, and a Tensor label y with values 1 or -1
  local loss = protos.criterion:forward(output, labels)

  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dpredicted = protos.criterion:backward(output, labels)

  -- backprop to  model
  local dummy = unpack(protos.model:backward(input, dpredicted))

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  return loss / opt.batch_size_real
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local iter = 0
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_acc_history = {}
local val_prop_acc_history = {}
local best_combined_rank = 9999999 -- set this high so it uptate it for sure
local old_params
local checkpoint_path = opt.checkpoint_path .. 'cp_id' .. opt.id ..'.cp'
local learning_rate = opt.learning_rate

--local data = loader:getBatch(opt,'test',false) --without negative samples
--print(data.audio[1])

local timerTot = torch.Timer()
while true do  

    local timer = torch.Timer()
    -- eval loss/gradient
    local losses = lossFun()
    --if iter % opt.losses_log_every == 0 then 
    --  loss_history[iter] = losses 
      -- print(string.format('train loss - iter %d: %f', iter, losses))
    --end

    local time = timer:time().real
    local timeTot = timerTot:time().real  

    -- decay the learning rate
    if iter%opt.learning_rate_decay_every == 0 and opt.learning_rate_decay_every >= 0 then
      --local frac = (iter - opt.learning_rate_decay_every) / opt.learning_rate_decay_every
      --local decay_factor = math.pow(0.5, frac)
      local decay_factor = opt.learning_rate_decay
      learning_rate = learning_rate * decay_factor
      --print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. learning_rate)
    end

    
    -- perform a parameter update
    if opt.optim == 'rmsprop' then
      rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
    elseif opt.optim == 'adagrad' then
      adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
    elseif opt.optim == 'sgd' then
      sgd(params, grad_params, opt.learning_rate)
    elseif opt.optim == 'sgdm' then
      sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
    elseif opt.optim == 'sgdmom' then
      sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
    elseif opt.optim == 'adam' then
      adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
    else
      error('bad option opt.optim')
    end
    
   
    -- TODO apply normalization after param update
    --if opt.weight_decay >0 then
    --  for _,w in ipairs(reg) do
    --    w:add(-(opt.weight_decay*learning_rate), w)
    --  end
    --end

    -- stopping criterions
    iter = iter + 1
    if iter % 100 == 0 then collectgarbage() end
    if loss0 == nil then loss0 = losses end
    if losses > loss0 * 20 then
      print('loss seems to be exploding, quitting.')
      break
    end
    if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

    -- evaluate validation set
    if iter % opt.print_every == 0 then 
      local raudio, rvisual, top5t, top5v, sim = eval_ranking()
      local combined_rank = raudio + rvisual
      
      --revert parameters if you didnt learn the validation better then before
      --if combined_rank > best_combined_rank then
      --  if opt.revert_params >= 1 then
      --    params = old_params
      --  end
      --else
      --  best_combined_rank = combined_rank
      --  old_params = params
      --end

      if combined_rank < best_combined_rank then
        best_combined_rank = combined_rank

        -- save output vectors (forward test set)
        if opt.save_output > 0 and combined_rank < 280 then --iter > 200 then 
          print('Found better model! Saving npy...')
          save_output_vectors(combined_rank)
        end


      end

      local epoch = iter*opt.batch_size / opt.train_size
      print(string.format('e:%.2f (i:%d) train/val loss: %f/%f  audio/visual rank: %.0f(%d)/%.0f(%d) BC:%d sim: %.4f  batch/total time: %.4f / %.0f', epoch, iter, losses, eval_split('val'), raudio, top5t, rvisual, top5v, best_combined_rank, sim, time, timeTot/60))
      print(string.format("lr= %6.4e grad norm = %6.4e, param norm = %6.4e, grad/param norm = %6.4e", learning_rate, grad_params:norm(), params:norm(), grad_params:norm() / params:norm()))
    end

  end
