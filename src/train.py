import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import os
clap = r'C:\Users\Cassterss\Documents\Python\code\clump_ai\dataset_clups\1'
noclap = r'C:\Users\Cassterss\Documents\Python\code\clump_ai\dataset_clups\0'
nr = 16000
class data(Dataset):
    def __init__(self):
        super().__init__()
        self.x = []
        self.y = []

        for name in os.listdir(clap):
            root = os.path.join(clap, name)
            sound,rate = sf.read(root)
            sound = torch.tensor(sound, dtype=torch.float32)
            
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
            else:
                pass

            sound = sound.unsqueeze(0)
            self.x.append(sound)
            y_otv = torch.tensor(1, dtype=torch.long)
            self.y.append(y_otv)
        for name in os.listdir(noclap):
                    root = os.path.join(noclap, name)
                    sound,rate = sf.read(root)
                    sound = torch.tensor(sound, dtype=torch.float32)
                    
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
                    else:
                        pass
        
                    sound = sound.unsqueeze(0)
                    self.x.append(sound)
                    y_otv = torch.tensor(0, dtype=torch.long)
                    self.y.append(y_otv)




    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x,y
datas = data()
dataload = DataLoader(dataset=datas, batch_size=32, shuffle=True)

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
model.train()
loss_fn = nn.BCEWithLogitsLoss()
optom = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 100
persent = 0

for epoch in range(epochs):
    epoch_loss = 0
    for x,y in dataload:
        y_pred = model(x)
        y = y.unsqueeze(1).float()
        loss = loss_fn(y_pred, y)

        optom.zero_grad()
        loss.backward()
        optom.step()
        epoch_loss += loss.item()

    if epoch % 1 == 0:
        persent += 1
        print("Model train is ", persent, "%")
        epoch_loss /= len(dataload)

        print(f"Epoch {epoch+1}, Loss = {epoch_loss:.4f}")

torch.save(model.state_dict(), "claps_AI.pth")
print("model was saved")

