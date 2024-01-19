import flet as ft
import tensorflow as tf

model = tf.keras.models.load_model('insertmodelname')

text2convert = ft.TextField(label='Input text here',multiline= True,color='orange')
btn = ft.ElevatedButton("Convert to speech")

def main(page: ft.Page):
    page.title = 'Sample TTS App'
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    text2convert = ft.TextField(label='Input text here',multiline= True,color='orange')
    page.add(text2convert)

    btn = ft.ElevatedButton("Convert to speech")
    page.add(btn)
   

def convert_click(e):
    text2convert.value = str(text2convert.value)
    page.update()

def tts(insertuserinput):


    


    #file_picker = ft.FilePicker()
    #page.overlay.append(file_picker)
    #page.update()
    #speech=ft.Audio(autoplay=True)
    #page.overlay.append(audio1)
    #page.add(
        #ft.Text("This is an app with background audio."),
        #ft.ElevatedButton("Stop playing", on_click=lambda _: audio1.pause()),
    #)
ft.app(target=main)
