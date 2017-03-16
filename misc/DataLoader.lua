require 'hdf5'
local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  -- store settings locally
  self.batch_size = opt.batch_size
  self.batch_size_with_neg = opt.batch_size_real
  self.neg_samples = opt.neg_samples
  self.gpuid = opt.gpuid
  self.train_size = opt.train_size
  self.val_size = opt.val_size
  self.test_size = opt.test_size
  self.feat_size_text = opt.feat_size_text
  self.feat_size_visual = opt.feat_size_visual
  -- self. = opt.

  -- load the json file which contains additional information about the dataset
  print('DataLoader loading json file: ', opt.input_json)
  self.info = utils.read_json(opt.input_json)
  

  self.d_txt = "/tweets"
  self.d_im  = "/images"

  -- open the hdf5 files
  print('DataLoader loading h5 file (text): ', opt.input_h5_text)
  self.h5_file_text = hdf5.open(opt.input_h5_text, 'r')
  print(self.h5_file_text:read(self.d_txt):dataspaceSize())
  print('DataLoader loading h5 file (visual): ', opt.input_h5_visual)
  self.h5_file_visual = hdf5.open(opt.input_h5_visual, 'r')
  print(self.h5_file_visual:read(self.d_im):dataspaceSize())

  --local text = self.h5_file_text:read('/tweets'):partial({2,2},{1,self.feat_size_text})
  --print("-----------------------------",text)
  
  -- TODO some sanity checks

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
  -- all the indexes referes to the original tables (hd5f)
  local gap

  if split == 'train' then
    self.split_ix[split] = torch.randperm(self.train_size)
    elseif split == 'val' then
      gap = torch.Tensor(self.val_size):fill(self.train_size)
      self.split_ix[split] = torch.randperm(self.val_size):add(1, gap)
      elseif split == 'test' then
        --gap = torch.Tensor(self.test_size):fill(self.train_size+self.val_size)
        --self.split_ix[split] = torch.Tensor():range(1,self.test_size):add(1, gap)
        self.split_ix[split] = torch.Tensor():range(1,250000)
      else
        error('error: unknown split - ' .. split)
  end

    self.iterators[split] = 1
end

function DataLoader:getFeatSizeText()
        return self.feat_size_text
end

function DataLoader:getFeatSizeVisual()
        return self.feat_size_visual
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - ...
  - ...
--]]
function DataLoader:getBatch(split, negSamFlag)
  
  local split_ix = self.split_ix[split]

  local batch_size_real = self.batch_size
  if negSamFlag then 
    batch_size_real = self.batch_size_with_neg--with negative examples
  end

  assert(split_ix, 'split ' .. split .. ' not found.')

  -- initialize one table per modality
  local text_batch = torch.FloatTensor(batch_size_real, self.feat_size_text):fill(0)
  local visual_batch = torch.FloatTensor(batch_size_real, self.feat_size_visual):fill(0)
  local label_batch = torch.FloatTensor(batch_size_real):fill(0)

  local max_index = (#split_ix)[1]

  --if you are going to overflow, reset the indices 
  --(i know we are not processing some examples at the end, but they are random every time so it shouldn't harm)
  future_index = self.iterators[split] + batch_size_real
  if future_index >= max_index then
    self:resetIndices(split)
  end

  local si = self.iterators[split] -- use si for semplicity but update the self.iterator later
  local i = 1

  for _dum = 1,self.batch_size do
    ix = split_ix[si]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. si)
    -- positive examples
    -- read text e visual vectors from the dataset (read line at position ix)
    -- TODO maybe put all in a tensor if u have enough ram
    text_batch[i] = self.h5_file_text:read(self.d_txt):partial({ix,ix},{1,self.feat_size_text})
    local t_norm = text_batch[i]:norm()
    if t_norm == 0 then print('Textual norm is 0') break end
    text_batch[i] = text_batch[i] / t_norm 

    visual_batch[i] = self.h5_file_visual:read(self.d_im):partial({ix,ix},{1,self.feat_size_visual})
    local v_norm = visual_batch[i]:norm()
    if v_norm == 0 then print('Visual norm is 0')  break end
    visual_batch[i] = visual_batch[i] / v_norm

    label_batch[i] = 1


    i = i+1

    -- negative examples
    if negSamFlag then
      for _dumNeg = 1,self.neg_samples do
        --get random example (verify it's not the same as the positive)
        local rand_ix = ix
        while rand_ix == ix do
          rand_ix = math.random(self.train_size) 
        end

        -- add negative text samples
        text_batch[i] = self.h5_file_text:read(self.d_txt):partial({rand_ix,rand_ix},{1,self.feat_size_text})
        local t_norm = text_batch[i]:norm()
        if t_norm == 0 then print('Textual norm is 0') break end
        text_batch[i] = text_batch[i] / t_norm


        visual_batch[i] = self.h5_file_visual:read(self.d_im):partial({ix,ix},{1,self.feat_size_visual})
        local v_norm = visual_batch[i]:norm()
        if v_norm == 0 then print('Visual norm is 0')  break end
        visual_batch[i] = visual_batch[i] / v_norm

        label_batch[i] = -1 --negative examples, we want them to be far
        i = i+1

        -- add negative visual samples
        text_batch[i] = self.h5_file_text:read(self.d_txt):partial({ix,ix},{1,self.feat_size_text})
        text_batch[i] = text_batch[i] / text_batch[i]:norm()

        visual_batch[i] = self.h5_file_visual:read(self.d_im):partial({rand_ix,rand_ix},{1,self.feat_size_visual})
        visual_batch[i] = visual_batch[i] / visual_batch[i]:norm()

        label_batch[i] = -1 --negative examples, we want them to be far
        i = i+1
      end
    end

    si = si + 1
  end
  self.iterators[split] = si

  local data = {}
  if self.gpuid<0 then
    data.text = text_batch
    data.visual = visual_batch
    data.labels = label_batch
  else
    data.text = text_batch:cuda()
    data.visual = visual_batch:cuda()
    data.labels = label_batch:cuda()
  end

  return data

end
