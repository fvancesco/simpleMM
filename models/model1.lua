require 'nn' 
require 'nngraph'
require 'misc.Peek'
--require 'misc.LinearNB

local model1 = {}
function model1.model(mapping, feat_size_text, feat_size_visual, num_layers, hidden_size, output_size)

  -- nn with 2 inputs 2 outputs

  -- fill input table ({text, visual})
  local input = {}
  local text = nn.Identity()()
  local image = nn.Identity()()
  table.insert(input, text)
  table.insert(input, image)
  local resultText
--[[
  -- text
  tweetIdxs = ...
  local tweetEmbeddings = LookupTable(params.vocab_size, 100)(tweetIdxs)
  local sum = nn.Sum(dimension=1, sizeAverage=true) --or 2 TODO
  local h1t = nn.Tanh()(sum)
  local outnn = nn.Linear(feat_size_visual, hidden_size)(h1t)
]]

  if mapping == 1 then 
    resultText = text --dont do any mapping, just output same text vector
  else 
    -- first layer
    local h0t = nn.Linear(feat_size_text, hidden_size)(text)
    local h1t = nn.Tanh()(h0t)
    --other layers
    for _ = 1,num_layers-1 do
      h0t = nn.Linear(hidden_size, hidden_size)(h1t)
      h1t = nn.Tanh()(h0t)
    end
    resultText = nn.Linear(hidden_size, output_size)(h1t)
  end 

  -- visual
  -- first layer
  local h0v = nn.Linear(feat_size_visual, hidden_size)(image)
  local h1v = nn.Tanh()(h0v)
  --other layers
  for _ = 1,num_layers-1 do
    h0v = nn.Linear(hidden_size, hidden_size)(h1v)
    h1v = nn.Tanh()(h0v)
  end
  local resultVisual = nn.Linear(hidden_size, output_size)(h1v)


  -- fill output table ({resultText, resultVisual})  
  local output = {}
  table.insert(output, resultText)
  table.insert(output, resultVisual)
     
  local model = nn.gModule(input, output)
   
  return model
end

return model1




--[[

  local h0t
  local h1t = text
  for _ =1,num_layers do
    h0t = nn.Linear(feat_size_text, hidden_size)(h1t)
    h1t = nn.Tanh()(h0t)
  end
  local resultText = nn.Linear(hidden_size, output_size)(h1t)

  
  local h0t = nn.Linear(feat_size_text, hidden_size)(text)
  local h1t = nn.Tanh()(h0t)
  local resultText = nn.Linear(hidden_size, output_size)(h1t)
  
  local h0v = nn.Linear(feat_size_visual, hidden_size)(image)
  local h1v = nn.Tanh()(h0v)
  local resultVisual = nn.Linear(hidden_size, output_size)(h1v)
--]]