from enum import Enum
import os





INPUT_TACOTRON_MODEL = "tacotron-20180906/model.ckpt"


class TextToSpeechSynthesizerModels(Enum):

    TTSX3               = 0
    TACOTRON            = 1
    DEFAULT = TTSX3


class TextToSpeechSynthesizer:

    def __init__(self, model=TextToSpeechSynthesizerModels.DEFAULT, path=None, path_output=None, training=True):
        self._base = None
        if model == TextToSpeechSynthesizerModels.TTSX3:
            self._base = TextToSpeechSynthesizer_TTSX3(path, path_output, training)
        elif model == TextToSpeechSynthesizerModels.TACOTRON:
            self._base = TextToSpeechSynthesizer_TACOTRON(path, path_output, training)
        print("Synthesizer loaded!")

    def synthesize(self, text, outputfile):
        self._base.synthesize(text, outputfile)
        print("Synthesized text={} output={}".format(text, outputfile))

    def synthesize_name(self, name):
        text = "Hello " + name
        outputfile = name + ".wav"
        self.synthesize(text, outputfile)

    def synthesize_datasets(self, path_datasets):
        for (_d, names, _f) in os.walk(path_datasets):
            print("names " + str(names))
            for name in names:
                self.synthesize_name(name)
            break

    def playaudio(self, path, name, block=True):
        self._base.playaudio(path, name, block=True)


class TextToSpeechSynthesizer_TTSX3:

    def __init__(self, path, path_output, training):
        self._training = training
        if not self._training:
            import pyttsx3 # lazy loading
            self._synthesizer = pyttsx3.init()
            self._synthesizer.setProperty('voice', 'english')

    def synthesize(self, text, outputfile):
        pass

    def playaudio(self, path, name, block):
        #print("TextToSpeechSynthesizer_TTSX3")
        if not self._training:
            text = "Hello " + name
            self._synthesizer.say(text)
            self._synthesizer.runAndWait()


class TextToSpeechSynthesizer_TACOTRON:

    def __init__(self, path, path_output, training):
        self._training = training
        if self._training:
            from libfaceid.tacotron.synthesizer import Synthesizer # lazy loading
            self._path_output = path_output
            self._synthesizer = Synthesizer()
            self._synthesizer.load(path + INPUT_TACOTRON_MODEL)

    def synthesize(self, text, outputfile):
        if self._training:
            with open(self._path_output + outputfile, 'wb') as file:
                file.write(self._synthesizer.synthesize(text))

    def playaudio(self, path, name, block):
        #print("TextToSpeechSynthesizer_TACOTRON")
        from playsound import playsound # lazy loading
        try:
            filename = path + "/" + name + ".wav"
            playsound(filename, block)
        except:
            pass

