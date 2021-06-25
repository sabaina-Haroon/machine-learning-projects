from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import config as cfg
import random
import pickle
import parameters as params
import json
import cv2
from tqdm import tqdm
import time


class pa3_train_dataset_generator(Dataset):
    
    def __init__(self, fraction=0.1, transform=None, shuffle=False, data_percentage=1.0):

        '''
        fraction: fraction of the unlabeled dataset (% of data from the 37k videos), I don't want to remove any of the labeled video in any epoch
        '''

        self.fraction=fraction
        
        self.labeled_datapaths = open(os.path.join(cfg.path_folder,'train_l.txt'),'r').read().splitlines()
        self.unlabeled_datapaths = open(os.path.join(cfg.path_folder,'train_u.txt'),'r').read().splitlines()

        self.classes= json.load(open(cfg.class_mapping))['classes']

        self.data_percentage = data_percentage

        self.shuffle= shuffle
        
        if self.shuffle:
            random.shuffle(self.labeled_datapaths)
            random.shuffle(self.unlabeled_datapaths)
        
        self.labeled_datapaths = self.labeled_datapaths[0: int(len(self.labeled_datapaths))]
        self.unlabeled_datapaths = self.unlabeled_datapaths[0: int((fraction)*len(self.unlabeled_datapaths))]

        self.data= self.labeled_datapaths + self.unlabeled_datapaths
        
        if self.shuffle:
            random.shuffle(self.data)
        
        self.data_limit = int(len(self.data)*self.data_percentage)
        self.data = self.data[0:self.data_limit]


    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):        
        full_clip, trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path = self.process_data(index)
        return full_clip, trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path


    def process_data(self, idx):
        
        # label_building
        vid_path = self.data[idx]

        if vid_path.split('/')[5] == 'UCF101':
            label = self.classes[vid_path.split('/')[7]]
            lu_bit = 1
        else:
            label = -1
            lu_bit = 0
        
        # clip_building
        full_clip, trimmed_clip, temporal_span_full, temporal_span_trimmed = self.build_clip(vid_path)

        return full_clip, trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path

    def build_clip(self, vid_path):
        try:
            count = -1
            
            cap=cv2.VideoCapture(vid_path)
            cap.set(1,0)
            frame_count = cap.get(7)

            skip_frames_full = frame_count/(params.num_frames-1)
            start_frame_full = 0 # here 0 can be replaced with random starting point, but, for now it's getting complicated
            
            
            skip_frames_trimmed = skip_frames_full/ params.skip_ratio

            full_clip_frames = [int(round(skip_frames_full)*f) for f in range(params.num_frames)]
            if full_clip_frames[-1] >= frame_count:
                full_clip_frames[-1] = int(frame_count-1)

            temporal_span_full = [full_clip_frames[0], full_clip_frames[-1]]
            start_frame_trimmed = np.random.randint(0,int(full_clip_frames[-1]-(params.num_frames_trimmed-1)*skip_frames_trimmed))

            trimmed_clip_frames = [start_frame_trimmed + int(round(f*skip_frames_trimmed)) for f in range(params.num_frames_trimmed)]
            temporal_span_trimmed = [trimmed_clip_frames[0], trimmed_clip_frames[-1]]
        


            if skip_frames_trimmed < params.min_trimmed_skip:
                print(f'Frame count: {frame_count}')
                print(f'Skip frame full: {skip_frames_full}')
                print(f'Skip frame trimmed: {skip_frames_trimmed}')
                print()
                # print(temp)
                print(f'Clip {vid_path} doesn`t have sufficient frames') # It is filtered out, still there might be some corner case
                return None, None, None, None
            else:
                full_clip = []
                trimmed_clip = []
                list_full = []
                list_trimmed = []
                while(cap.isOpened()): 
                    count+=1
                    if ((count not in full_clip_frames) and (count not in trimmed_clip_frames)) and (ret == True): 
                        ret, frame = cap.read()
                        continue
                    ret, frame = cap.read()
                    if ret == True:
                        # Resize 
                        frame1 = cv2.resize(frame, (112,112))

                        if (count in full_clip_frames):
                            full_clip.append(frame1)
                            list_full.append(count)

                        if (count in trimmed_clip_frames): 
                            trimmed_clip.append(frame1)
                            list_trimmed.append(count)
                    else:
                        break

                if len(full_clip) < params.num_frames:
                    # print(f'Full clip has {len(full_clip)} frames')
                    # print(f'Full video has {frame_count} frames')
                    # print(f'Full video suppose to have these frames: {full_clip_frames}')
                    # print(f'Actual video has           these frames: {list_full}')
                    # print(f'final count value is {count}')
                    if params.num_frames-len(full_clip)>=2:
                        print(f'Clip {vid_path} is missing {params.num_frames-len(full_clip)} frames')
                    for remaining in range(params.num_frames-len(full_clip)):
                        full_clip.append(frame1)
                        
                    # print(f'Now frame count in the full clip is {len(full_clip)}')

                if len(trimmed_clip) < params.num_frames_trimmed:
                    # print(f'Trimmed clip has {len(trimmed_clip)} frames')
                    # print(f'Trimmed video has {frame_count} frames')
                    # print(f'Trimmed video suppose to have these frames: {trimmed_clip_frames}')
                    # print(f'Actual video has           these frames: {list_trimmed}')
                    # print(f'final count value is {count}')
                    if params.num_frames_trimmed-len(trimmed_clip)>=2:
                        print(f'Clip {vid_path} is missing {params.num_frames_trimmed-len(trimmed_clip)} frames')
                    for remaining in range(params.num_frames_trimmed-len(trimmed_clip)): 
                        trimmed_clip.append(trimmed_clip[-1])
                        


                assert(len(full_clip)==params.num_frames)
                assert(len(trimmed_clip)==params.num_frames_trimmed)

                return full_clip, trimmed_clip, full_clip_frames, trimmed_clip_frames

        except:
            print(f'Clip {vid_path} Failed')
            return None, None, None, None


class pa3_test_dataset_generator(Dataset):
    
    def __init__(self, transform=None, shuffle=False, data_percentage=1.0):

        
        self.labeled_datapaths = open(os.path.join(cfg.path_folder,'test_l.txt'),'r').read().splitlines()
        self.classes= json.load(open(cfg.class_mapping))['classes']
        self.data_percentage = data_percentage
        self.shuffle= shuffle
        if self.shuffle:
            random.shuffle(self.labeled_datapaths)
        
        self.labeled_datapaths = self.labeled_datapaths[0: int(len(self.labeled_datapaths))]

        self.data= self.labeled_datapaths
        
        if self.shuffle:
            random.shuffle(self.data)
        
        self.data_limit = int(len(self.data)*self.data_percentage)
        self.data = self.data[0:self.data_limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):        
        trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path = self.process_data(index)
        return trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path


    def process_data(self, idx):
        
        # label_building
        vid_path = self.data[idx]

        if vid_path.split('/')[5] == 'UCF101':
            label = self.classes[vid_path.split('/')[7]]
            lu_bit = 1
        else:
            label = -1
            lu_bit = 0
        
        # clip_building
        trimmed_clip, temporal_span_full, temporal_span_trimmed = self.build_clip(vid_path)

        return trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path

    def build_clip(self, vid_path):

        count = -1
        
        cap=cv2.VideoCapture(vid_path)
        cap.set(1,0)
        frame_count = cap.get(7)
        
        
        start_frame_trimmed = int(frame_count/2 - params.num_frames_trimmed*params.validation_skip_rate/2)

        trimmed_clip_frames = [start_frame_trimmed + int(round(f*params.validation_skip_rate)) for f in range(params.num_frames_trimmed)]
        temporal_span_trimmed = [trimmed_clip_frames[0], trimmed_clip_frames[-1]]
      
        if frame_count < 2*params.num_frames:
            print(f'Clip doesn`t have sufficient frames, frame count {frame_count}')
            return None, None, None
        else:
            try: 
                trimmed_clip = []
                list_full = []
                list_trimmed = []
                while(cap.isOpened()): 
                    count+=1
                    ret, frame = cap.read()

                    if ((count not in trimmed_clip_frames)) and (ret == True): 
                        
                        continue
                    if ret == True:
                        
                        frame1 = cv2.resize(frame, (112,112))

                        if (count in trimmed_clip_frames): 
                            trimmed_clip.append(frame1)
                            list_trimmed.append(count)

                    else:
                        break
                        

                if len(trimmed_clip) < params.num_frames_trimmed:
                    print(f'Trimmed clip has {len(trimmed_clip)} frames')
                    print(f'Trimmed video has {frame_count} frames')
                    print(f'Trimmed video suppose to have these frames: {trimmed_clip_frames}')
                    print(f'Actual video has           these frames: {list_trimmed}')
                    print(f'final count value is {count}')
                    for remaining in range(params.num_frames_trimmed-len(trimmed_clip)):
                        trimmed_clip.append(trimmed_clip[-1])
                        if remaining==1:
                            print('2 frames missing')


                assert(len(trimmed_clip)==params.num_frames_trimmed)

                full_clip_frames= [0]
                return trimmed_clip, full_clip_frames, trimmed_clip_frames

            except:
                print(f'Clip {vid_path} Failed!')
                return None, None, None



def collate_fn1(batch):
    
    full_clip, trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path = [], [], [], [], [], [], []

    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None or item[3] == None or item[4] == None or item[5] == None or item[6] == None):
            full_clip.append(torch.from_numpy(np.asarray(item[0],dtype='f')))
            trimmed_clip.append(torch.from_numpy(np.asarray(item[1],dtype='f')))
            label.append(item[2])
            lu_bit.append(item[3])
            temporal_span_full.append(np.asarray(item[4]))
            temporal_span_trimmed.append(np.asarray(item[5]))
            vid_path.append(item[6])

    full_clip = torch.stack(full_clip, dim=0)
    trimmed_clip = torch.stack(trimmed_clip, dim=0)


    return full_clip, trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path

def collate_fn2(batch):
    
    trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path = [], [], [], [], [], []

    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None or item[3] == None or item[4] == None or item[5] == None):
            trimmed_clip.append(torch.from_numpy(np.asarray(item[0],dtype='f')))
            label.append(item[1])
            lu_bit.append(item[2])
            temporal_span_full.append(np.asarray(item[3]))
            temporal_span_trimmed.append(np.asarray(item[4]))
            vid_path.append(item[5])

    trimmed_clip = torch.stack(trimmed_clip, dim=0)


    return trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path
    
if __name__ == '__main__':
    
    # for num_workers in [0, 2, 4, 8, 16, 20]:
    # dataset = pa3_train_dataset_generator(fraction=1.0, transform=None, shuffle=True, data_percentage=1.0)
    # train_dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn1) #keep the num_workers=1 while dubugging
    train_dataset = pa3_train_dataset_generator(fraction=0.0, transform=None, shuffle=True, data_percentage=1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn1)
    # print(f'Steps per epoch: {len(dataset)/params.batch_size}')
    t=time.time()
    pickle_loc= '/home/ishan/acv-pa3/vis2/'
    pbar=tqdm(total=int(len(train_dataset)/params.batch_size))
    label_count = np.zeros((1,102))
    for i,(full_clip, trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path) in enumerate(train_dataloader):
        if i%100 == 0:
            print(f'Full_clip shape is {full_clip.permute(0,1,4,2,3).shape}')
            print(f'Trimmed_clip shape is {trimmed_clip.shape}')
        for ii in label:
            label_count[0,ii]+=1
            
                    
            # print(f'label is {label}')
            # print(f'lu_bit is {lu_bit}')
            # print(f'temporal_span_full is {temporal_span_full}')
            # print(f'temporal_span_trimmed is {temporal_span_trimmed}')
            # print(f'vid_path is {vid_path}')

            # pickle.dump(full_clip, open(pickle_loc+'full_clip.pkl','wb'))
            # pickle.dump(trimmed_clip,open(pickle_loc+'trimmed_clip.pkl','wb'))
            # pickle.dump(label,open(pickle_loc+'label.pkl','wb'))
            # pickle.dump(lu_bit, open(pickle_loc+'lu_bit.pkl','wb'))
            # pickle.dump(temporal_span_full, open(pickle_loc+'temporal_span_full.pkl','wb'))
            # pickle.dump(temporal_span_trimmed, open(pickle_loc+'temporal_span_trimmed.pkl','wb'))
            # pickle.dump(vid_path, open(pickle_loc+'vid_paths.pkl','wb'))
            # exit()
    
        #     print(f'{i} batches gone')
        #     # print(len(full_clip))
        # print(f'Full_clip shape is {full_clip.shape}')
        # print(f'Trimmed_clip shape is {trimmed_clip.shape}')

        # #     # print(trimmed_clip)
        #     print(f'label is {label}')
        #     print(f'lu_bit is {lu_bit}')
        #     print(f'temporal_span_full is {temporal_span_full}')
        #     print(f'temporal_span_trimmed is {temporal_span_trimmed}')
        # #     print()
        #     # print(vid_path)
        pbar.update(1)
    print(f'Time taken to load data is {time.time()-t}')
    pbar.close()
    print(label_count)
    
    '''
    dataset = pa3_test_dataset_generator(transform=None, shuffle=True, data_percentage=1.0)
    test_dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn2) #keep the num_workers=1 while dubugging
    pbar=tqdm(total=int(len(dataset)/params.batch_size))

    for i,(trimmed_clip, label, lu_bit, temporal_span_full, temporal_span_trimmed, vid_path) in enumerate(test_dataloader):
        if i%100 == 999:
            print(f'Trimmed_clip shape is {trimmed_clip.shape}')
        pbar.update(1)
    pbar.close()
    '''


        
