#!/usr/bin/env python
# coding: utf-8

# In[32]:


from datasets import load_dataset
from datasets import Audio
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split

lj_speech = load_dataset("lj_speech")


# In[33]:

#train_test_ds = datasets.load_dataset("lj_speech", split="train+test")
print(lj_speech)


# In[34]:


lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=22050))


# In[31]:


print(lj_speech["train"][0])


# In[12]:


#from transformers import AutoProcessor

#processor = AutoProcessor.from_pretrained("openai/whisper-medium.en")


# In[17]:


def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"])
    
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return batch


# In[18]:


lj_speech = lj_speech.map(prepare_dataset, remove_columns=lj_speech["train"].column_names)


# In[ ]:


MAX_DURATION_IN_SECONDS = 10.0

def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS


# In[ ]:


lj_speech["train"] = lj_speech["train"].filter(is_audio_length_in_range, input_columns=["input_length"])

