import torch
import time
import soundfile as sf
from transformers import AutoTokenizer, Wav2Vec2Processor, HubertForCTC
from datasets import load_dataset

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").cpu()
    
data, _ = sf.read('./samples_jfk.wav')
input_values = processor(data, return_tensors="pt").input_values.cpu()  # Batch size 1
processor.tokenizer.save_pretrained("tokenizer.json", tokenizer_only=True)
start = time.time()
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
end = time.time()
transcription = processor.decode(predicted_ids[0])
print("===========", end-start)
print(transcription)
print("===========")
