import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import sounddevice as sd 
import serial

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv1d(1, 32, 11, padding=5)
        self.l2 = nn.Conv1d(32, 64, 11, padding=5)
        self.l3 = nn.Conv1d(64, 128, 11,padding=5)
        self.l4 = nn.Conv1d(128, 64, 11,padding=5)
        self.flatten = nn.Flatten()
        self.max = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.l5 = nn.Linear(64 * 2000, 1)



    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.max(x)

        x = self.l3(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.max(x)

        x = self.flatten(x)
        x = self.l5(x)
        return x 

model = model()
model.load_state_dict(torch.load(r'C:\Users\Cassterss\Documents\Python\code\clump_ai\claps_AI.pth'))
model.eval()
root = r'C:\Users\Cassterss\Documents\Python\code\clump_ai\clap.wav'
nr = 16000
audio = sd.rec(
    16000,
    samplerate=nr,
    channels=1,
    dtype="float32"
)
print('rec')
sd.wait()
audio = audio.squeeze(1)
sound = torch.tensor(audio, dtype=torch.float32)
print(sound.size())
nr = 16000     
if sound.ndim == 2:
    sound = sound.mean(dim=1)
            
if sound.size()[0] < nr:
    need = nr - sound.size()[0]
    left = need // 2
    right = need - left
    sound = F.pad(sound, (left,right))
elif sound.size()[0] > nr:
    peak_sound = sound.abs().argmax().item()
            
    if peak_sound - (nr // 2) < 0:
        need_left = (nr // 2) - peak_sound
        sound = F.pad(sound, (need_left, 0))
        start = (peak_sound + need_left) - (nr // 2)
        end = (peak_sound + need_left) + (nr // 2)
        sound = sound[start:end]
    elif peak_sound + (nr // 2) > sound.size()[0]:
        needd = peak_sound + (nr // 2) - sound.size()[0]
        sound = F.pad(sound, (0, needd))
        start = peak_sound - (nr // 2)
        end = peak_sound + (nr // 2)
        sound = sound[start:end]
    else: 
        start = peak_sound - (nr // 2)
        end = peak_sound + (nr // 2)
        sound = sound[start:end]


buffer = torch.zeros(16000)
esp = serial.Serial("COM3", 115200)
print("Connected!")
while True:

    audio = sd.rec(
        3200,
        samplerate=16000,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = torch.tensor(audio).squeeze(1)

    buffer = torch.cat((buffer, audio))

    buffer = buffer[-16000:]
    window = buffer.clone()
    peak = buffer.abs().argmax().item()

    if peak - (nr // 2) < 0:
        need = (nr // 2) - peak
        window = F.pad(window, (need, 0))
        start = (peak + need) - (nr // 2)
        end = (peak + need) + (nr // 2)
        window = window[start:end]
    else: 
        need = (peak + 8000) - nr
        window = F.pad(window, (0, need))
        start = peak - 8000
        end = peak + 8000
        window = window[start:end]



    x = window.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        y = torch.sigmoid(model(x))

    if y.item() >= 0.55:
        esp.write(b"ON\n")
        print(y.item())
    else: 
        esp.write(b"OFF\n")
        print(y.item())



    