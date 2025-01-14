import os

from pocketsphinx import AudioFile


class SphinxKeywordDetector():
    def __init__(self, target_transcription, threshold=1e-20, verbose=False):
        self.target_transcription = target_transcription
        self.verbose = verbose
        self.kws_config = {
            'verbose': self.verbose,
            'keyphrase': self.target_transcription,
            'kws_threshold': threshold,
            'lm': False,
        }

    def detect(self, file_name):

        kws_results = []
        audio = AudioFile(audio_file=file_name, **self.kws_config)

        for phrase in audio:
            try:
                result = phrase.segments(detailed=True)
            except TypeError as e:
                #print('Caught type error while generating stitched sample')
                #print('Using filename: %s' % file_name)
                #raise e
                continue
            # TODO:: confirm that when multiple keywords are detected, every detection is valid
            if len(result) == 1:
                start_time = result[0][2] * 10
                end_time = result[0][3] * 10
                if self.verbose:
                    print('%4sms ~ %4sms' % (start_time, end_time))
                kws_results.append((start_time, end_time))

        return kws_results
