import os
import glob
import torch
from torch.nn.parallel.data_parallel import DataParallel
import numpy as np
import cv2

from GaitEmbedding.GaitEmbedder import GaitEmbedder


class VanillaGaitEmbedder(GaitEmbedder):
    '''
    Class used for vanilla gait embedding method in a video or image file, inherits from GaitEmbedder.
    '''
    def __init__(self, pretrained_to_epoch=8):
        super().__init__()

        # set up gait embedding model:
        self.DetectorType = 'VanillaGaitEmbedder'


    def extract_features_from_mesh_object(self, mesh_object, bbox=None):
        mesh_information = None
        return mesh_information


    def get_embedding_for_frame_sequence(self, folder_path):
        kpts_filenames = glob.glob(os.path.join(folder_path, '*.kpts.npz'))
        print("Loading {} keypoints files from {}".format(len(kpts_filenames), folder_path))
        kpts_from_all_frames = []
        for fn in kpts_filenames:
            dat = np.load(fn)
            kpts_from_all_frames.append(dat['kpts'])

            embedding = np.array(kpts_from_all_frames)
        print(embedding)
        return embedding
