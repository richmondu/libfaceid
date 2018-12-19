from enum import Enum





class TextToSpeechSynthesizerModels(Enum):

    TACOTRON            = 0
    DEFAULT = TACOTRON


class TextToSpeechSynthesizer:

    def __init__(self, model=TextToSpeechSynthesizerModels.DEFAULT, path=None, path_output=None):
        self._base = None
        if model == TextToSpeechSynthesizerModels.TACOTRON:
            self._base = TextToSpeechSynthesizer_TACOTRON(path, path_output)
            print("Synthesizer model loaded!")

    def synthesize(self, text, outputfile):
        self._base.synthesize(text, outputfile)
        print("Synthesizer text synthesized! text={} output={}".format(text, outputfile))


class TextToSpeechSynthesizer_TACOTRON:

    def __init__(self, path, path_output):
        from tacotron.synthesizer import Synthesizer # lazy loading
        self._path_output = path_output
        self._synthesizer = Synthesizer()
        self._synthesizer.load(path + "tacotron-20180906/model.ckpt")

    def synthesize(self, text, outputfile):
        with open(self._path_output + outputfile, 'wb') as file:
            file.write(self._synthesizer.synthesize(text))

# Tacotron Usage
#synthesizer = TextToSpeechSynthesizer(model=TextToSpeechSynthesizerModels.DEFAULT, path="../models/synthesis/", path_output="../datasets/")
#synthesizer.synthesize("Hello Name", "Name.wav")


