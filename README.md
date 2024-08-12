# SpeechModel-Pretrainer-Lightning
Ever since I started my journey in Deep learning, I have explored various domains, but recently the IndicVoices dataset caught my attention and motivated me to train the wav2vec2 model using Lightning AI. This project embodies my long-standing interest in harnessing the power of wav2vec2 with the efficiency and scalability of Lightning AI.

## Environment Setup

### Prerequisites
Ensure you are either on Linux or using Windows Subsystem for Linux (WSL).

### Using Python's `venv`
1. Create and activate a virtual environment:
   ```bash
   python -m venv speech_pretrain
   source speech_pretrain/bin/activate
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Using Conda
1. Create and activate a new environment:
   ```bash
   conda create -n speech_pretrain python=3.10.14 -y
   conda activate speech_pretrain
   ```
2. Install necessary packages:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   conda install lightning ffmpeg-python -y
   pip install -r requirements.txt
   ```

## Dataset

### Downloading Dataset
Navigate to the dataset directory:
```bash
cd datasets/IndicVoices/
```
Links to download the audio files are available in `datasets/IndicVoices/indicVoices.train.txt`. Please note, you'll need an authentication token to access the files. Register for the competition to obtain your token, and append it to the download URLs in the format `?token=<TOKEN>`. Store your token in a `.env` file as follows:
```
INDICVOICES_TOKEN=<TOKEN>
```
To download the dataset, execute:
```bash
python download.py
```

### Extracting Data
To unzip all `.tgz` files and correctly save them to the respective folders:
```bash
for filename in v*/*.tgz; do
  tar -xvf "$filename" -C "$(dirname "$filename")"
done
```

## Processing Dataset
Converting audio to safetensors significantly reduces the loading time for training. To process the audio files:
```bash
python wav2safetensors.py
```
This script processes waveforms into safetensors, optimizing training time.

## Training

To start training the model, use the following command, which outlines each parameter clearly:
```bash
python wav2vec2_pretrain.py \
  --model_name facebook/wav2vec2-large-xlsr-53 \
  --train_path dataset-manifest/train.data.txt \
  --val_path dataset-manifest/validation.data.txt \
  --training_steps 400000
```

Follow these steps carefully to ensure a smooth setup and effective training of the model using the IndicVoices dataset. Happy training!
