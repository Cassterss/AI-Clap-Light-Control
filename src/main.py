import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
root = r"C:\Users\Cassterss\Documents\Python\code\clump_ai\dataset_clups\1\07_Clap_08_90_SP.wav"
notmal_samples = 16000

x,y = sf.read(root)
x = torch.tensor(x, dtype=torch.float32)
print(x.size()[0])
if x.ndim == 2:
    x = x.mean(dim=1)
    
    


if x.size()[0] < 16000:
    zeros = notmal_samples - x.size()[0]
    left_pad = zeros // 2
    right_pad = zeros - left_pad
    print("left_pad:", left_pad)
    print("right_pad:", right_pad)
    x = F.pad(x,(left_pad, right_pad))

print(x.size())
maxs = x.abs().argmax().item()
print(maxs)
if x.size()[0] > 16000:
    maxs = x.abs().argmax().item()
    rol = notmal_samples // 2
    if maxs - rol < 0:
        need = rol - maxs
        end = maxs + rol + need
        maxs_with_zeros = F.pad(x,(need, 0))
        x = maxs_with_zeros[0:end]


    elif maxs + rol > x.size()[0]:
        start = maxs - rol
        end = maxs + rol
        need  = end -x.size()[0]
        maxs_with_zeros = F.pad(x, (0, need))
        x = maxs_with_zeros[start:end]
    else: 
        start = maxs - rol
        end = maxs + rol
        x = x[start:end]
         
x = x.unsqueeze(0)
print(x.size())