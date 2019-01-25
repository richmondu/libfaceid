from enum import Enum
import os





INPUT_TACOTRON_MODEL = "tacotron-20180906/model.ckpt"


class SpeechSynthesizerModels(Enum):

    TTSX3               = 0 # real-time, no .wav file generated
    TACOTRON            = 1 # generates .wav file during training
    GOOGLECLOUD         = 2 # generates .mp3 file during training, requires internet access
    DEFAULT = TTSX3


class SpeechSynthesizer:

    def __init__(self, model=SpeechSynthesizerModels.DEFAULT, path=None, path_output=None, training=True):
        self._base = None
        if model == SpeechSynthesizerModels.TTSX3:
            self._base = SpeechSynthesizer_TTSX3(path, path_output, training)
        elif model == SpeechSynthesizerModels.TACOTRON:
            self._base = SpeechSynthesizer_TACOTRON(path, path_output, training)
        elif model == SpeechSynthesizerModels.GOOGLECLOUD:
            self._base = SpeechSynthesizer_GOOGLECLOUD(path, path_output, training)
        #print("Synthesizer loaded!")

    def synthesize(self, text, outputfile):
        self._base.synthesize(text, outputfile)
        #print("Synthesized text={} output={}".format(text, outputfile))

    def synthesize_name(self, name):
        text = "Hello " + name
        outputfile = name
        self.synthesize(text, outputfile)

    def synthesize_datasets(self, path_datasets):
        for (_d, names, _f) in os.walk(path_datasets):
            #print("names " + str(names))
            for name in names:
                self.synthesize_name(name)
            break

    def playaudio(self, path, name, block=True):
        try:
            if (name is not None) and (name != "Unknown") and (name != "Fake"):
                self._base.playaudio(path, name, block)
        except:
            print("SpeechSynthesizer playaudio EXCEPTION")
            pass


class SpeechSynthesizer_TTSX3:

    def __init__(self, path, path_output, training):
        self._training = training
        if not self._training:
            import pyttsx3 # lazy loading
            try:
                self._synthesizer = pyttsx3.init()
            except Exception as e:
                print("pyttsx3 exception " + str(e))
            self._synthesizer.setProperty('voice', 'english')
            
    def synthesize(self, text, outputfile):
        pass

    def playaudio(self, path, name, block):
        #print("SpeechSynthesizer_TTSX3")
        if not self._training:
            text = "Hello " + name
            if block:
                self._synthesizer.say(text)
                self._synthesizer.runAndWait()
            else:
                from threading import Thread # lazy loading
                self._thread =  None
                self._thread = Thread(target=self.thread_play, kwargs=dict(text=text))
                self._thread.setDaemon(True)
                self._thread.start()

    def thread_play(self, text="None"):
        self._synthesizer.say(text)
        self._synthesizer.runAndWait()
        self._thread = None
        #import time
        #time.sleep(1)


class SpeechSynthesizer_TACOTRON:

    _file_extension = ".wav"

    def __init__(self, path, path_output, training):
        self._training = training
        if self._training:
            from libfaceid.tacotron.synthesizer import Synthesizer # lazy loading
            self._path_output = path_output
            self._synthesizer = Synthesizer()
            self._synthesizer.load(path + INPUT_TACOTRON_MODEL)

    def synthesize(self, text, outputfile):
        if self._training:
            with open(self._path_output + outputfile + self._file_extension, 'wb') as file:
                file.write(self._synthesizer.synthesize(text))

    def playaudio(self, path, name, block):
        #print("SpeechSynthesizer_TACOTRON")
        filename = os.path.abspath(path + "/" + name + self._file_extension)
        SpeechSynthesizer_Utils().playaudio(filename, block)


class SpeechSynthesizer_GOOGLECLOUD:

    _file_extension = ".mp3"

    def __init__(self, path, path_output, training):
        self._training = training
        self._path_output = path_output

    def synthesize(self, text, outputfile):
        if self._training:
            from gtts import gTTS # lazy loading
            tts = gTTS(text)
            tts.save(self._path_output + outputfile + self._file_extension)

    def playaudio(self, path, name, block):
        #print("SpeechSynthesizer_GOOGLECLOUD")
        filename = os.path.abspath(path + "/" + name + self._file_extension)
        SpeechSynthesizer_Utils().playaudio(filename, block)


class SpeechSynthesizer_Utils:
    def __init__(self):
        self._option = 0
        
    def playaudio(self, filename, block):
        if self._option == 0:
            from playsound import playsound # lazy loading
            #print("playsound")
            playsound(filename, block)
        else:
            pass
            
