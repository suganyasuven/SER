import pickle
from glob import iglob
from shutil import rmtree
import numpy as np

from constants import *
from model_data import read_audio_from_filename


def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def del_folder(path):
    try:
        rmtree(path)
    except:
        pass


del_folder(OUTPUT_DIR_TRAIN)
del_folder(OUTPUT_DIR_TEST)
mkdir_p(OUTPUT_DIR_TRAIN)
mkdir_p(OUTPUT_DIR_TEST)


def extract_class_id(wav_filename):
    return wav_filename.split('/')[1].split('_')[0]


def convert_data():
            for i, wav_filename in enumerate(iglob(os.path.join(DATA_AUDIO_DIR, '**/**.wav'),
                                                   recursive=True)):
                class_id = extract_class_id(wav_filename)
                print(class_id)
                audio_buf = read_audio_from_filename(wav_filename, target_sr=TARGET_SR)
                # normalize mean 0, variance 1
                audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
                original_length = len(audio_buf)
                print(i, wav_filename, original_length, np.round(np.mean(audio_buf), 4), np.std(audio_buf))
                if original_length < AUDIO_LENGTH:
                    audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
                    print('PAD New length =', len(audio_buf))
                elif original_length > AUDIO_LENGTH:
                    audio_buf = audio_buf[0:AUDIO_LENGTH]
                    print('CUT New length =', len(audio_buf))
                print('audio buf - ', audio_buf.shape)

                output_folder = OUTPUT_DIR_TRAIN
                if 'Ses01' in wav_filename:
                    output_folder = OUTPUT_DIR_TEST
                output_filename = os.path.join(output_folder, str(i) + '.pkl')

                out = {'class_id': class_id,
                       'audio': audio_buf,
                       'sr': TARGET_SR}
                with open(output_filename, 'wb') as w:
                    pickle.dump(out, w)


if __name__ == '__main__':
    convert_data()
