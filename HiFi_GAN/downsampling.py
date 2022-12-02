import librosa
from scipy.io.wavfile import write
from tqdm import tqdm


file_list = ['filelists/train_file_list.txt', 'filelists/val_file_list.txt', 'filelists/test_file_list.txt']  

for file in file_list:    
    print(f'make {file}')
    with open(file, 'r', encoding='utf-8') as fi:
            wav_files = [line.split('|')[0].replace('multi_speaker_emotion_dataset_2022',
                                                        'multi_speaker_emotion_dataset_2022_for_hifi') for line in fi.readlines()]
        
    for wav_path in tqdm(wav_files):
        y, s = librosa.load(wav_path, sr=22050)
        write(wav_path, s, y)
        
