from enum import Enum



# Added these accounts for testing purposes only
CLIENT_KEY_WITAI    = "KCSDQHNAJ74ORMOG2PFOO4NRZBJVAIDT"
CLIENT_ID_HOUNDIFY  = "n33EnhuJaA6DZpYqRwiYBg=="
CLIENT_KEY_HOUNDIFY = "C06_n6A37PS4WDRb3Zi7mejyZ9BbLABya-n6nAsskyt7ZEos5PHZKsanRJlXuYhg5yPtDC6k9Xb--B1iVumNVA=="


class SpeechRecognizerModels(Enum):

    GOOGLECLOUD = 0 # requires internet access
    WITAI       = 1 # requires internet access, requires key from https://wit.ai
    HOUNDIFY    = 2 # requires internet access, requires key and id from https://www.houndify.com
    DEFAULT = GOOGLECLOUD


class SpeechRecognizer:

    def __init__(self, model=SpeechRecognizerModels.DEFAULT, path=None):
        if model == SpeechRecognizerModels.GOOGLECLOUD:
            self._base = SpeechRecognizer_Common(model)
        elif model == SpeechRecognizerModels.WITAI:
            self._base = SpeechRecognizer_Common(model)
        elif model == SpeechRecognizerModels.HOUNDIFY:
            self._base = SpeechRecognizer_Common(model)

    def start(self, words, callback):
        self._base.start(words, callback)

    def stop(self):
        self._base.stop()


class SpeechRecognizer_Common:

    def __init__(self, model):
        import speech_recognition # lazy loading
        self._r = speech_recognition.Recognizer()
        self._m = speech_recognition.Microphone()
        self._model = SpeechRecognizerModels(model)

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
        # recognize input from microphone
        try:
            if self._model == SpeechRecognizerModels.GOOGLECLOUD:
                text = recognizer.recognize_google(audio)
            elif self._model == SpeechRecognizerModels.WITAI:
                text = recognizer.recognize_wit(audio, key=CLIENT_KEY_WITAI)
                text = text.lower()
            elif self._model == SpeechRecognizerModels.HOUNDIFY:
                text = recognizer.recognize_houndify(audio, client_id=CLIENT_ID_HOUNDIFY, client_key=CLIENT_KEY_HOUNDIFY)
        except:
            if self._model != SpeechRecognizerModels.GOOGLECLOUD:
                text = recognizer.recognize_google(audio)

        # check if detected text is in the list of words to recognize
        try:
            if text is not None and text in self._trigger_words:
                if self._user_callback is not None:
                    self._user_callback(text)
            else:
                print(text + " - unknown")
        except:
            pass

