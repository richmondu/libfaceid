from enum import Enum



class SpeechRecognizerModels(Enum):

    GOOGLECLOUD = 0
    DEFAULT = GOOGLECLOUD


class SpeechRecognizer:

    def __init__(self, model=SpeechRecognizerModels.DEFAULT, path=None):
        if model == SpeechRecognizerModels.GOOGLECLOUD:
            self._base = SpeechRecognizer_GOOGLECLOUD()

    def start(self, words, callback):
        self._base.start(words, callback)

    def stop(self):
        self._base.stop()


class SpeechRecognizer_GOOGLECLOUD:

    def __init__(self):
        import speech_recognition # lazy loading
        self._r = speech_recognition.Recognizer()
        self._m = speech_recognition.Microphone()

    def start(self, words, user_callback):
        self._user_callback = user_callback
        self._trigger_words = [word.lower() for word in words]
        #print(self._trigger_words)
        with self._m as source:
            self._r.adjust_for_ambient_noise(source)
        self._listener = self._r.listen_in_background(self._m, self.callback)

    def stop(self):
        self._listener(wait_for_stop=True)

    def callback(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio)
            if text in self._trigger_words:
                self._user_callback(text)
            else:
                print(text + " - unknown")
        except:
            pass

