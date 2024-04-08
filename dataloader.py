import torch
import csv
import numpy as np
import random

train_input_data_path = 'your_path'
train_mel_spectrogram_path = 'your_path'
test_input_data_path = 'your_path'
test_mel_spectrogram_path = 'your_path'

def trim_mel(mel_spect):
    num_frames = mel_spect.shape[1]

    num_frames_5s = 157  # 5초 동안의 프레임 수 계산하기
    start_frame = np.random.randint(0, num_frames - num_frames_5s)
    mel_spect_5s = mel_spect[:, start_frame:start_frame + num_frames_5s]
    mel_spect_5s = mel_spect_5s/80 + 1

    return mel_spect_5s


class DataLoader:
    def __init__(self):

        with open(train_input_data_path, 'r', encoding='utf-8') as csv_file:
            train_data_by_artist = {}
            csv_reader = csv.DictReader(csv_file)
            train_file_data = []
            train_file_id = []
            for row in csv_reader:
                train_file_data.append(row['file'])
                train_file_id.append(row['id'])
                curr_artist_id = row['id']
                curr_file = row['file']

                if curr_artist_id in train_data_by_artist:
                    train_data_by_artist[curr_artist_id].append(curr_file)
                else:
                    train_data_by_artist[curr_artist_id] = [curr_file]

        self.train_data = train_file_data
        self.train_id = train_file_id
        self.train_dic = train_data_by_artist

        with open(test_input_data_path, 'r', encoding='utf-8') as csv_file:
            test_data_by_artist = {}
            csv_reader = csv.DictReader(csv_file)
            test_file_data = []
            for row in csv_reader:
                test_file_data.append(row['file'])
                curr_artist_id = row['name']
                curr_file = row['file']

                if curr_artist_id in test_data_by_artist:
                    test_data_by_artist[curr_artist_id].append(curr_file)
                else:
                    test_data_by_artist[curr_artist_id] = [curr_file]

        self.test_data = test_file_data
        self.test_dic = test_data_by_artist

    def __getitem__(self,item):
        if item == 'train':
            return [self.train_data, self.train_id, self.train_dic]
        elif item == 'test':
            return [self.test_data, self.test_dic]


    def train_data_loader(self):
        path = train_mel_spectrogram_path
        sample_set = []

        for i in range(len(self.train_data)):
            train_curr_file = self.train_data[i]
            curr_artist_id = self.train_id[i]
            for j in range(10):
                curr_anchor = np.load(path + train_curr_file, allow_pickle=True)
                anchor_mel = np.array(trim_mel(curr_anchor)).reshape(1, 128, 157) # trim mel 할 때 random

                pos_songs = [song for song in self.train_dic[curr_artist_id]]
                pos_item = random.choice(pos_songs)  # id는 같은 곡 1개 random choice
                pos_mel = np.array(trim_mel(np.load(path + pos_item, allow_pickle=True))).reshape(1, 128, 157)

                neg_songs = [song for artist_id, songs in self.train_dic.items() if artist_id != curr_artist_id for song in songs]
                neg_item = random.sample(neg_songs, 5)  # id가 다른 모든 곡들 중에서 random 5개 choice해서 하나의 list로 저장됨
                neg_mel = np.array([trim_mel(np.load(path + song, allow_pickle=True)) for song in neg_item])


                a = np.concatenate((anchor_mel, pos_mel, neg_mel),axis=0)  # 현재 샘플 세트에 추가
                sample_set.append(a)

        return sample_set

    def test_data_loader(self):
        path = test_mel_spectrogram_path
        sample_set = []

        for i in range(len(self.test_data)):
            test_curr_file = self.test_data[i]
            curr_artist_id = test_curr_file.split(' ', 1)[0]
            curr_anchor = np.load(path + test_curr_file, allow_pickle=True)

            anchor_mel = np.array(trim_mel(curr_anchor)).reshape(1, 128, 157) # trim mel 할 때 random

            pos_songs = [song for song in self.test_dic[curr_artist_id]]
            pos_item = random.choice(pos_songs)  # id는 같은 곡 1개 random choice
            pos_mel = np.array(trim_mel(np.load(path + pos_item, allow_pickle=True))).reshape(1, 128, 157)

            neg_songs = [song for artist_id, songs in self.test_dic.items() if artist_id != curr_artist_id for song in songs]
            neg_item = random.sample(neg_songs, 5)  # id가 다른 모든 곡들 중에서 random 5개 choice해서 하나의 list로 저장됨
            neg_mel = np.array([trim_mel(np.load(path + song, allow_pickle=True)) for song in neg_item])

            a = np.concatenate((anchor_mel, pos_mel, neg_mel),axis=0)  # 현재 샘플 세트에 추가
            sample_set.append(a)

        return sample_set

    def generate_batch(self, batch_size=32):
        sample_set = self.train_data_loader()
        np.random.shuffle(sample_set)

        num_batches = len(sample_set) // batch_size
        batches = []

        for i in range(num_batches):
            batch = sample_set[i * batch_size: (i + 1) * batch_size]
            batches.append(batch)

        # 마지막 배치가 나머지 데이터를 포함하도록
        if len(sample_set) % batch_size != 0:
            last_batch = sample_set[num_batches * batch_size:]
            batches.append(last_batch)

        # np.random.shuffle(batches)

        return batches

    def generate_test_batch(self, batch_size=32):
        sample_set = self.test_data_loader()
        batches = random.sample(sample_set, 32)

        return batches

