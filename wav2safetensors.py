import torchaudio, torch, os
from transformers import Wav2Vec2FeatureExtractor
from safetensors.torch import save_file, load_file
from multiprocessing import Pool
import pandas as pd
from tqdm.auto import tqdm

MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

def convert_to_sf(audio_path):
    destination = f"{audio_path.rstrip('.wav')}.safetensors"
    if os.path.exists(destination):
        return destination
    audio, sr = torchaudio.load(audio_path)
    audio = torchaudio.functional.resample(audio, sr, feature_extractor.sampling_rate)
    inputs = feature_extractor(
        audio[0], 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=20*feature_extractor.sampling_rate, 
        return_tensors="pt",
        truncation=True
    )
    batch = {
        "input_values": inputs.input_values[0],
        "input_length": torch.LongTensor([len(inputs.input_values[0])]),
    }
    if inputs.attention_mask is not None:
        batch["attention_mask"] = inputs.attention_mask[0]
    
    save_file(batch, destination)
    return destination

def main():
    audios = pd.read_csv("filenames.data.txt", header=None)
    with Pool(4) as p:
        result = list(tqdm(p.imap(convert_to_sf, audios[0]), total=len(audios[0])))
    return result

if __name__ == "__main__":
    processed = main()
    with open("filenames.sf.txt", "w") as f:
        f.write("\n".join(processed))