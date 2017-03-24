require 'cutorch'
require 'cunn'
npy4th = require 'npy4th'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  self.train_size = opt.train_size
  self.val_size = opt.val_size
  self.test_size = opt.test_size

  -- open the npy files an load tensors for train, val and test
  self.tensor_audio = {}
  --print('DataLoader loading npy file (audio): ', opt.input_npy_audio)
  self.tensor_audio['train'] = npy4th.loadnpy(opt.input_npy .. "train_561.npy"):cuda()
  self.tensor_audio['val'] = npy4th.loadnpy(opt.input_npy .. "val_561.npy"):cuda()
  self.tensor_audio['test'] = npy4th.loadnpy(opt.input_npy .. "test_561.npy"):cuda()
  print(self.tensor_audio)

  self.tensor_visual = {}
  self.tensor_visual['train'] = npy4th.loadnpy(opt.input_npy .. "train" .. opt.input_code .. ".npy"):cuda()
  self.tensor_visual['val'] = npy4th.loadnpy(opt.input_npy .. "val" .. opt.input_code .. ".npy"):cuda()
  self.tensor_visual['test'] = npy4th.loadnpy(opt.input_npy .. "test" .. opt.input_code .. ".npy"):cuda()
  print(self.tensor_visual)

  --Initialize indexes 
  self.split_ix = {}
  self.iterators = {}

  self:resetIndices('train')
  self:resetIndices('val')
  self:resetIndices('test')

  -- TODO print with for
  print('Assigned to train: ', (#self.split_ix['train'])[1])
  print('Assigned to val: ', (#self.split_ix['val'])[1])
  print('Assigned to test: ', (#self.split_ix['test'])[1])

end

function DataLoader:resetIndices(split)
  if split == 'train' then
    self.split_ix[split] = torch.randperm(self.train_size)
    elseif split == 'val' then
      self.split_ix[split] = torch.randperm(self.val_size) --maybe this is not necessary (rand over val?)
      elseif split == 'test' then
        self.split_ix[split] = torch.Tensor():range(1,self.test_size)
      else
        error('error: unknown split - ' .. split)
  end
    self.iterators[split] = 1
end

function DataLoader:getFeatSizeaudio()
        return self.feat_size_audio
end

function DataLoader:getFeatSizeVisual()
        return self.feat_size_visual
end

function DataLoader:getOneTensor(id_tensor, split)
    local data = {}
    --print(id_tensor)
    --print(split)
    data.audio = self.tensor_audio[split]:narrow(1, id_tensor, 1)
    data.visual = self.tensor_visual[split]:narrow(1, id_tensor, 1)
    --print(data)
    return data
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - ...
  - ...
--]]
function DataLoader:getBatch(opt, split, negSamFlag)
  
  local split_ix = self.split_ix[split]

  local batch_size = opt.batch_size
  if negSamFlag then 
    batch_size = batch_size * 3
  end

  --assert(split_ix, 'split ' .. split .. ' not found.')

  -- initialize one table per modality
  --local audio_batch = torch.FloatTensor(batch_size, opt.feat_size_audio):fill(0)
  --local visual_batch = torch.FloatTensor(batch_size, opt.feat_size_visual):fill(0)
  --local label_batch = torch.FloatTensor(batch_size):fill(0)
  local audio_batch = torch.CudaTensor(batch_size, opt.feat_size_audio):fill(0)
  local visual_batch = torch.CudaTensor(batch_size, opt.feat_size_visual):fill(0)
  local label_batch = torch.CudaTensor(batch_size):fill(0)

  local max_index = (#split_ix)[1]

  --if you are going to overflow, reset the indices 
  --(i know we are not processing some examples at the end, but they are random every time so it shouldn't harm)
  future_index = self.iterators[split] + batch_size
  if future_index >= max_index then
    self:resetIndices(split)
  end

  local si = self.iterators[split] -- use si for semplicity but update the self.iterator later
  local i = 1

  --OPT BATCH SIZE, ojo (porque los negatives los pongo solo si hace falta)
  for _dum = 1,opt.batch_size do
    ix = split_ix[si]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. si)
    audio_batch[i] = self.tensor_audio[split]:narrow(1, ix, 1)
    local t_norm = audio_batch[i]:norm()
    if t_norm == 0 then print('Audio norm is 0') break end
    audio_batch[i] = audio_batch[i] / t_norm 

    visual_batch[i] = self.tensor_visual[split]:narrow(1, ix, 1)
    local v_norm = visual_batch[i]:norm()
    if v_norm == 0 then print('Visual norm is 0')  break end
    visual_batch[i] = visual_batch[i] / v_norm

    label_batch[i] = 1

    i = i+1

    -- negative examples
    if negSamFlag then
      --get random example (verify it's not the same as the positive)
      local rand_ix = ix
      while rand_ix == ix do
        rand_ix = math.random(self.train_size) 
      end

      -- add negative audio samples
      audio_batch[i] = self.tensor_audio[split]:narrow(1, rand_ix, 1)
      audio_batch[i] = audio_batch[i] / audio_batch[i]:norm()

      visual_batch[i] = self.tensor_visual[split]:narrow(1, ix, 1)
      visual_batch[i] = visual_batch[i] / visual_batch[i]:norm()

      label_batch[i] = -1 --negative examples, we want them to be far
      i = i+1

      -- add negative visual samples
      audio_batch[i] = self.tensor_audio[split]:narrow(1, ix, 1)
      audio_batch[i] = audio_batch[i] / audio_batch[i]:norm()

      visual_batch[i] = self.tensor_visual[split]:narrow(1, rand_ix, 1)
      visual_batch[i] = visual_batch[i] / visual_batch[i]:norm()

      label_batch[i] = -1 --negative examples, we want them to be far
      i = i+1
    end

    si = si + 1
  end
  self.iterators[split] = si

  local data = {}
  data.audio = audio_batch
  data.visual = visual_batch
  data.labels = label_batch


  return data

end
