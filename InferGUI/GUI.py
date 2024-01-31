import tkinter as tk
from tkinter import Label, Entry, Text, Button, messagebox
import numpy as np
import torch
import json
from threading import Thread
from hparams import create_hparams
from model import Tacotron2
from meldataset import MAX_WAV_VALUE
from models import Generator
from audio_processing import griffin_lim
from text import text_to_sequence
from scipy.io.wavfile import write
from env import AttrDict

class TTSApp:
    def __init__(self, root):
        self.root = root
        root.title("TTS Inference GUI")

        # Create widgets
        self.label = Label(root, text="Enter Text:")
        self.text_entry = Entry(root)
        self.text_entry.insert(0, "Put your text here...")  # Default text for testing
        self.output_text = Text(root, height=5, width=50)
        self.output_text.insert(tk.END, "Output will be displayed here")
        self.output_text.config(state=tk.DISABLED)  # Disable text editing
        self.infer_button = Button(root, text="Infer and Save to WAV", command=self.run_inference)

        # Place widgets in grid
        self.label.grid(row=0, column=0, padx=10, pady=5)
        self.text_entry.grid(row=0, column=1, padx=10, pady=5)
        self.infer_button.grid(row=1, column=0, columnspan=2, pady=10)
        self.output_text.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        # Initialize variables
        self.hifigan, self.h = self.get_hifigan()
        self.model, self.hparams = self.get_Tactron2()
        self.inference_thread = None

    def get_hifigan(self):
        conf = "config_v1.json"
        with open(conf) as f:
            json_config = json.loads(f.read())
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        hifigan = Generator(h).to(torch.device("cpu"))
        state_dict_g = torch.load("Hifigan", map_location=torch.device("cpu"))
        hifigan.load_state_dict(state_dict_g["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan, h

    def has_MMI(self, STATE_DICT):
        return any(True for x in STATE_DICT.keys() if "mi." in x)

    def get_Tactron2(self):
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        hparams.max_decoder_steps = 3000
        hparams.gate_threshold = 0.25
        model = Tacotron2(hparams)
        state_dict = torch.load("TTS_FirstTraining", map_location=torch.device("cpu"))['state_dict']
        if self.has_MMI(state_dict):
            raise Exception("ERROR: This notebook does not currently support MMI models.")
        model.load_state_dict(state_dict)
        _ = model.eval()
        return model, hparams

    def run_inference(self):
        if self.inference_thread and self.inference_thread.is_alive():
            messagebox.showinfo("Info", "Inference is already running.")
            return

        text = self.text_entry.get()
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)  # Clear previous output
        self.output_text.insert(tk.END, f"Inferred Text: {text}\n")
        self.output_text.config(state=tk.DISABLED)

        self.inference_thread = Thread(target=self.infer_thread, args=(text, False))
        self.inference_thread.start()

    def infer_thread(self, text, pronounciation_dictionary):
        try:
            for i in [x for x in text.split("\n") if len(x)]:
                if not pronounciation_dictionary:
                    if i[-1] != ";":
                        i = i + ";"
                else:
                    i = ARPA(i)
                with torch.no_grad():
                    sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
                    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
                    mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)

                    # Generate audio with HiFIGAN
                    y_g_hat = self.hifigan(mel_outputs_postnet.float())
                    audio = y_g_hat.squeeze()
                    audio = audio * MAX_WAV_VALUE

                    # Save the audio to a WAV file
                    output_wav_file = f"output_{i[:10]}.wav"
                    write(output_wav_file, self.hparams.sampling_rate, audio.cpu().numpy().astype("int16"))

                    # Update GUI
                    self.root.after(0, self.update_output_text, f"Saved to WAV: {output_wav_file}\n")

        except Exception as e:
            self.root.after(0, self.show_error, f"Error during inference: {e}\n")

    def update_output_text(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state=tk.DISABLED)

    def show_error(self, message):
        messagebox.showerror("Error", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = TTSApp(root)
    root.mainloop()
