# 👏 Clap AI

Clap AI is a real-time audio classification project built with Python and PyTorch.

The model listens to microphone audio, detects whether a clap occurred, and sends a command to an ESP32 to control a physical light.

The main idea of the project is to connect a neural network with the real world:

`Microphone → Audio Processing → Neural Network → Prediction → ESP32 → Light`

---

## 🚀 How it works

1. Audio is captured from the microphone.
2. The waveform is converted to a Mel Spectrogram.
3. The PyTorch model classifies the sound.
4. The model predicts one of two classes:

```text
0 → No Clap
1 → Clap
When a clap is detected, Python sends a command through Serial.
ESP32 receives the command and turns the light on or off.
🧠 Audio preprocessing

The project uses Mel Spectrograms instead of raw audio directly.

Main parameters:

Sample Rate: 16000 Hz
n_fft: 400
hop_length: 160
n_mels: 64

Pipeline:

Waveform
   ↓
Resampling
   ↓
Mel Spectrogram
   ↓
AmplitudeToDB
   ↓
Neural Network

This converts audio into a frequency-time representation that is easier for the model to learn from.

🛠 Tech Stack
Python
PyTorch
Torchaudio
NumPy
SoundDevice
SoundFile
PySerial
ESP32
Arduino
⚡ Real-time detection

The model can work with live microphone input.

Microphone
    ↓
Audio Buffer
    ↓
Preprocessing
    ↓
Model
    ↓
Clap probability
    ↓
ESP32 command

If the detected probability is high enough, the program sends a command such as:

ON
OFF

to the ESP32 through the serial port.

📁 Project Structure
Clap-AI/
│
├── dataset/
│   ├── clap/
│   └── no_clap/
│
├── model/
│   └── claps_AI.pth
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── live_detection.py
│
├── esp32/
│   └── light_control.ino
│
├── requirements.txt
└── README.md
🧪 Training

The model is trained as a binary audio classifier:

Clap
vs
No Clap

During training:

Audio
↓
Mel Spectrogram
↓
Model
↓
Prediction
↓
Loss
↓
Backpropagation

The trained weights are saved and later used for real-time inference.

💡 Why I built this

I wanted to create something where AI does more than just output a prediction on the screen.

In this project, the neural network interacts with a real physical device.

A sound from the real world becomes:

sound
→ data
→ neural network
→ decision
→ physical action

This project is part of my experiments with PyTorch, audio processing, embedded systems and AI agents.

🔮 Future Improvements
Better resistance to background noise
Larger and more diverse dataset
Lower detection latency
Better real-time audio buffering
Multiple sound classes
Wake-word detection
Smart-home integration
Integration with a larger AI assistant
👨‍💻 Author

Created by Cassterss
