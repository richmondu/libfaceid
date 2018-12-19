from enum import Enum
import os
from playsound import playsound





INPUT_TACOTRON_MODEL = "tacotron-20180906/model.ckpt"


class TextToSpeechSynthesizerModels(Enum):

    TACOTRON            = 0
    DEFAULT = TACOTRON


class TextToSpeechSynthesizerUtils:

    def __init__(self):
        pass

    def playaudio(self, path, name, block=True):
        try:
            filename = self.getfile(path, name)
            playsound(filename, block)
        except:
            pass

    def getfile(self, path, name):
        return path + "/" + name + ".wav"


class TextToSpeechSynthesizer:

    def __init__(self, model=TextToSpeechSynthesizerModels.DEFAULT, path=None, path_output=None):
        self._base = None
        if model == TextToSpeechSynthesizerModels.TACOTRON:
            self._base = TextToSpeechSynthesizer_TACOTRON(path, path_output)
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


class TextToSpeechSynthesizer_TACOTRON:

    def __init__(self, path, path_output):
        from libfaceid.tacotron.synthesizer import Synthesizer # lazy loading
        self._path_output = path_output
        self._synthesizer = Synthesizer()
        self._synthesizer.load(path + INPUT_TACOTRON_MODEL)

    def synthesize(self, text, outputfile):
        with open(self._path_output + outputfile, 'wb') as file:
            file.write(self._synthesizer.synthesize(text))

