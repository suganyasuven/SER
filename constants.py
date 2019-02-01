import os

DATA_AUDIO_DIR = 'IEMOCAP'
# DATA_AUDIO_DIR = '/Users/philipperemy/Downloads/UrbanSound8K/audio'
TARGET_SR = 16000
OUTPUT_DIR = 'SPLITTED_FILES'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')

AUDIO_LENGTH = 96000

