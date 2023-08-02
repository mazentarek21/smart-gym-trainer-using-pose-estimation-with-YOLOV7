'''
from playsound import playsound

def exDone():
    playsound("C:\\Users\\mazen\\Desktop\\yolov7-pose-estimation-main\\goodjob.mp3")
    '''
    
import pydub
import wave
import threading
import pyaudio
def play_audio():
    wf = wave.open("goodjob.wav", 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()


    t = threading.Thread(target=play_audio)
    t.start()

