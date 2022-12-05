import json
from tqdm import tqdm
import random
from scipy.io.wavfile import write
import librosa
import numpy as np
from tqdm import tqdm

def make_filelists():
    print('start make filelists')
    emotion_dict = {
        '분노':1,
        '기쁨':4,
        '슬픔':5,
        '무감정':0,
        '놀람':6
    }

    file_list = []

    with open('../../dataset/multi_speaker_emotion_dataset_2022/any/all.json', 'r') as f:
        info_dict = json.load(f)
        
    for i in tqdm(range(len(info_dict))):
        for info in info_dict[i]['sentences']:        
            file_name = info['voice_piece']['filename']
            sentence = info['voice_piece']['tr']
            speaker_id = info['id'][7:10]
            file_path = f'../dataset/multi_speaker_emotion_dataset_2022/any/splitted/A/{speaker_id}/'
            emotion_id = emotion_dict[info['style']['emotion']]
            file_list.append(f'{file_path + file_name}|{sentence}|{emotion_id}|{speaker_id}\n')
            
    with open('../../dataset/multi_speaker_emotion_dataset_2022/cin/all.json', 'r') as f:
        info_dict = json.load(f)
        
    for i in tqdm(range(len(info_dict))):
        for info in info_dict[i]['sentences']:        
            file_name = info['voice_piece']['filename']
            sentence = info['voice_piece']['tr']
            speaker_id = info['id'][7:10]
            file_path = f'../dataset/multi_speaker_emotion_dataset_2022/cin/splitted/K/{speaker_id}/'
            emotion_id = emotion_dict[info['style']['emotion']]
            file_list.append(f'{file_path + file_name}|{sentence}|{emotion_id}|{speaker_id}\n')
            
    with open('../../dataset/multi_speaker_emotion_dataset_2022/zhong/all.json', 'r') as f:
        info_dict = json.load(f)
        
    for i in tqdm(range(len(info_dict))):
        for info in info_dict[i]['sentences']:        
            file_name = info['voice_piece']['filename']
            sentence = info['voice_piece']['tr']
            speaker_id = info['id'][7:10]
            file_path = f'../dataset/multi_speaker_emotion_dataset_2022/zhong/splitted/S/{speaker_id}/'
            emotion_id = emotion_dict[info['style']['emotion']]
            file_list.append(f'{file_path + file_name}|{sentence}|{emotion_id}|{speaker_id}\n')
            
    random.shuffle(file_list)

    with open('filelists/train_file_list.txt', 'w+') as lf:
        lf.writelines(file_list[:-600])
        
    with open('filelists/val_file_list.txt', 'w+') as lf:
        lf.writelines(file_list[-600:-500])
        
    with open('filelists/test_file_list.txt', 'w+') as lf:
        lf.writelines(file_list[-500:])
          
def preprocess_audio():
    sr = 16000
    max_wav_value=32768.0
    trim_fft_size = 1024
    trim_hop_size = 256
    trim_top_db = 23
    silence_audio_size = trim_hop_size * 3
    file_list = ['filelists/train_file_list.txt', 'filelists/val_file_list.txt', 'filelists/test_file_list.txt']  
    
    for F in file_list:
        f = open(F, encoding='utf-8')
        R = f.readlines()
        f.close()
        print('='*5+F+'='*5)

        for i, r in enumerate(tqdm(R)):
            wav_file = '../' + r.split('|')[0]
            data, sampling_rate = librosa.core.load(wav_file, sr)
            data = data / np.abs(data).max() *0.999
            data_= librosa.effects.trim(data, top_db= trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]
            data_ = data_*max_wav_value
            data_ = np.append(data_, [0.]*silence_audio_size)
            data_ = data_.astype(dtype=np.int16)
            write(wav_file, sr, data_)

def make_mel_filelists():
    print('start make mel_filelists')
    file_paths = ['filelists/train_file_list.txt', 'filelists/val_file_list.txt', 'filelists/test_file_list.txt']
    save_paths = ['filelists/train_mel_file_list.txt', 'filelists/val_mel_file_list.txt', 'filelists/test_mel_file_list.txt']

    for file_path, save_path in zip(file_paths, save_paths):
        f = open(file_path, 'r', encoding='utf-8')
        wf = open(save_path, 'w', encoding='utf-8')

        while True:
            line = f.readline()
            if not line: break
            script = line.replace('.wav', '.pt')
            wf.write(script)
            print(script)

        wf.close()
        f.close()

if __name__ == '__main__':
    make_filelists()
    preprocess_audio()
    make_mel_filelists()
