import sys
import os
import cv2
import glob
import numpy as np

import torchvision.transforms as transforms


class GaitEmbedder():
    '''
    Class used for body mesh detector in a video or image file.
    '''

    def __init__(self):
        # set up detector:
        self.EmbeddingType = 'General'
        self.embedding_model = None

        # set up transformer:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.input_img_transform = transforms.ToTensor()

    def extract_features_from_mesh_object(self, img, bbox=None):
        mesh_information = None
        return mesh_information

    def get_embedding_for_frame_sequence(self, folder_path):
        embedding = None
        return embedding
