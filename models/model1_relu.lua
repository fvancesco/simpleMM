require 'nn' 
require 'nngraph'
require 'misc.Peek'
--require 'misc.LinearNB

local model1 = {}
function model1.model(opt)

  -- nn with 2 inputs 2 outputs

  -- fill input table ({audio, visual})
  local input = {}
  local audio = nn.Identity()()
  local image = nn.Identity()()
  table.insert(input, audio)
  table.insert(input, image)

  --audio
  local resultAudio
  if opt.mapping == 1 then 
    resultAudio = audio --dont do any mapping, just output same audio vector
  else 
    -- first layer
    local h0t = nn.Linear(opt.feat_size_audio, opt.hidden_size)(audio)
    local h1t = nn.ReLU()(h0t)
    --other layers
    for _ = 1,opt.num_layers-1 do
      h0t = nn.Linear(opt.hidden_size, opt.hidden_size)(h1t)
      h1t = nn.ReLU()(h0t)
    end
    resultAudio = nn.Linear(opt.hidden_size, opt.output_size)(h1t)
  end 

  -- visual
  local resultVisual
  if opt.mapping == 2 then 
    resultVisual = image --dont do any mapping, just output same visual vector
  else 
    local h0v = nn.Linear(opt.feat_size_visual, opt.hidden_size)(image)
    local h1v = nn.ReLU()(h0v)
    --other layers
    for _ = 1,opt.num_layers-1 do
      h0v = nn.Linear(opt.hidden_size, opt.hidden_size)(h1v)
      h1v = nn.ReLU()(h0v)
    end
    resultVisual = nn.Linear(opt.hidden_size, opt.output_size)(h1v)
  end
  
  -- fill output table ({resultAudio, resultVisual})  
  local output = {}
  table.insert(output, resultAudio)
  table.insert(output, resultVisual)
     
  local model = nn.gModule(input, output)
   
  return model
end

return model1