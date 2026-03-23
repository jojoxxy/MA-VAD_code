import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        #audio_feature = np.load(self.df.loc[index]['path'][:-7] + '_audio.npy')
        #clip_feature = np.concatenate((clip_feature,audio_feature),axis=-1)
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length

class AudioCLIPDataset(data.Dataset):
    def __init__(self, dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.dim = dim
        self.test_mode = test_mode
        self.label_map = label_map

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        audio_feature = np.load(self.df.loc[index]['path'][:-7]+ '_audio.npy', allow_pickle=True)
        video_feature = np.load(self.df.loc[index]['path'], allow_pickle=True)

        if self.test_mode == False:
            audio_feature, length = tools.process_feat(audio_feature, self.dim)
            video_feature, length = tools.process_feat(video_feature, self.dim)
        else:
            audio_feature, length = tools.process_split(audio_feature, self.dim)
            video_feature, length = tools.process_split(video_feature, self.dim)

        audio_feature = torch.tensor(audio_feature)
        video_feature = torch.tensor(video_feature)
        label = self.df.loc[index]['label']
        return video_feature, audio_feature, label, length

class AudioDataset(data.Dataset):
    def __init__(self, dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.dim = dim
        self.test_mode = test_mode
        self.label_map = label_map

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        audio_feature = np.load(self.df.loc[index]['path'] + '_audio.npy')

        if self.test_mode == False:
            audio_feature, length = tools.process_feat(audio_feature, self.dim)
        else:
            audio_feature, length = tools.process_split(audio_feature, self.dim)
        video_feature = 0
        audio_feature = torch.tensor(audio_feature)
        label = self.df.loc[index]['label']
        return video_feature, audio_feature, label, length