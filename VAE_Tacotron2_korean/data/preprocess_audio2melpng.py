import matplotlib.pyplot as plt
from make_img import load_mel
# from src.utils.common.utils import load_wav_to_torch
import librosa
from tqdm import tqdm

# filelists = 'filelists/train_file_list.txt'
# filelists = 'filelists/val_file_list.txt'
filelists = 'filelists/test_file_list.txt'

f = open(filelists, 'r', encoding='utf-8')
lines = f.readlines()
f.close()

for line in tqdm(lines):
    wav_path = '../' + line.split('|')[0]  # .wav 경로

    # 저장할 png file 이름
    png_name = wav_path.replace('.wav', '.png')
    # print(png_name)

    # 기존 .pt파일을 png파일로 저장
    m = load_mel(wav_path)
    m = m.numpy() # 이미지로 만들기 위해 넘파이로 변경
    a, b = m.shape

    librosa.display.specshow(m)
    plt.Figure()
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig(png_name, bbox_inches='tight', pad_inches=0)
    plt.close()