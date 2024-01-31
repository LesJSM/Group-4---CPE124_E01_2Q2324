import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
from PyQt5.QtCore import Qt
import numpy as np
import pygame
from io import BytesIO
import scipy.io.wavfile as wavfile
import torch
import json
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator

class TTSApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('TTS Inference GUI')

        # Create widgets
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Enter text here")

        self.output_text = QTextEdit(self)
        self.output_text.setPlaceholderText("Output will be displayed here")
        self.output_text.setReadOnly(True)

        self.infer_button = QPushButton('Infer and Play', self)
        self.infer_button.clicked.connect(self.run_inference)

        # Set up layout
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel('Input Text:'))
        vbox.addWidget(self.text_input)
        vbox.addWidget(self.infer_button)
        vbox.addWidget(QLabel('Output:'))
        vbox.addWidget(self.output_text)

        self.setLayout(vbox)

    def run_inference(self):
        text = self.text_input.text()
        self.output_text.clear()

        # Call your inference function here
        # You may want to redirect stdout to capture print statements
        # Example:
        # sys.stdout = EmittingStream(textWritten=self.output_text.append)
        
        # Replace this with your actual inference logic
        hifigan, h = get_hifigan()
        model, hparams = get_Tactron2()

        for i in [x for x in text.split("\n") if len(x)]:
            if not pronounciation_dictionary:
                if i[-1] != ";": i = i + ";"
            else:
                i = ARPA(i)
            with torch.no_grad():
                sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
                mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

                # Generate audio with HiFIGAN
                y_g_hat = hifigan(mel_outputs_postnet.float())
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE

                # Display output text
                self.output_text.append(f"Inferred Text: {i}")

                # Play the audio using pygame
                play_audio(audio.cpu().numpy().astype("int16"), hparams.sampling_rate)

def get_hifigan():
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

def get_Tactron2():
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.max_decoder_steps = 3000 # Max Duration
    hparams.gate_threshold = 0.25 # Model must be 25% sure the clip is over before ending generation
    model = Tacotron2(hparams)
    state_dict = torch.load("TTS_FirstTraining", map_location=torch.device("cpu"))['state_dict']
    if has_MMI(state_dict):
        raise Exception("ERROR: This notebook does not currently support MMI models.")
    model.load_state_dict(state_dict)
    _ = model.eval()
    return model, hparams

def play_audio(audio_data, sampling_rate):
    pygame.mixer.init(frequency=sampling_rate)
    sound = pygame.sndarray.make_sound(audio_data)
    sound.play()
    pygame.time.delay(int(sound.get_length() * 1000))  # delay to allow audio playback

def has_MMI(STATE_DICT):
    return any(True for x in STATE_DICT.keys() if "mi." in x)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tts_app = TTSApp()
    tts_app.show()
    sys.exit(app.exec_())
