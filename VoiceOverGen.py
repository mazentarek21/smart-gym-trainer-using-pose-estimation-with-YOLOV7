from gtts import gTTS


filename = "example.txt"
with open(filename, "r") as file:
    # Read the contents of the file into a string
    txt = file.read()
    language = 'en'
    myobj = gTTS(text=txt, lang=language, slow=False)
    myobj.save("audios/KeepGoing.wav")