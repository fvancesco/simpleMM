require 'nn' 
--require 'nngraph'
require 'misc.Peek'
local model1 = {}
function model1.model(opt)

  -- nn with 2 inputs 1 outputs
  
  -- language part
  local audio_part = nn.Sequential()
  audio_part:add(nn.Linear(opt.feat_size_audio,opt.hidden_size))
  audio_part:add(nn.ReLU(true))
  if dropout > 0 then audio_part:add(nn.Dropout(dropout)) end
  -- visual part
  local visual_part = nn.Sequential()
  visual_part:add(nn.Linear(opt.feat_size_visual,opt.hidden_size))
  visual_part:add(nn.ReLU(true))
  if dropout > 0 then visual_part:add(nn.Dropout(dropout)) end

  -------- MM NET ---------
  -- 2 input 2 output
  local net = nn.ParallelTable()
  net:add(audio_part)
  net:add(visual_part)

  local out = nn.Sequential()
  out:add(net)
  out:add(nn.JoinTable(2))
  out:add(nn.ReLU())
  out:add(nn.Linear(hidden_size*2,output_size))
  out:add(nn.LogSoftMax())

  return out
end 
return model1
