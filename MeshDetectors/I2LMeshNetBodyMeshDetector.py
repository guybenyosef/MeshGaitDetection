import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
import numpy as np
import cv2

from MeshDetectors.BodyMeshDetector import BodyMeshDetector

from I2LMeshNet.config import cfg as I2LMeshNet_cfg
from I2LMeshNet.model import get_model as get_2LMeshNet_model
from I2LMeshNet.common.utils.smplpytorch.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from I2LMeshNet.common.utils.vis import vis_mesh, save_obj, render_mesh, vis_keypoints_with_skeleton, vis_keypoints
from I2LMeshNet.common.utils.preprocessing import process_bbox, generate_patch_image
from I2LMeshNet.common.utils.transforms import pixel2cam, cam2pixel

import torchvision.transforms as transforms
import torch.backends.cudnn as cud

class I2LMeshNetBodyMeshDetector(BodyMeshDetector):
    '''
    Class used for I2LMeshNet body mesh detector in a video or image file, inherits from BodyMeshDetector.
    '''
    def __init__(self, pretrained_to_epoch=8):
        super().__init__()

        # set up detector:
        self.DetectorType = 'I2LMeshNet'
        self.mseh_detector_model_path = "./Pretrained/snapshot_{}.pth.tar".format(pretrained_to_epoch)   #'Pretrained/snapshot_8.pth.tar'
        self.I2LMeshNet_cfg = I2LMeshNet_cfg
        self.I2LMeshNet_cfg.set_args(gpu_ids='0', stage='param')    # check run_mesh for usage of more than 1 gpu

        # SMPL joint set
        self.joint_num = 29 # original: 24. manually add nose, L/R eye, L/R ear
        self.joints_name = (
        'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe',
        'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
        'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
        self.flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) , (25,26), (27,28) )
        self.skeleton = (
        (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
        (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15),
        (24, 25), (24, 26), (25, 27), (26, 28))

        # SMPl mesh
        self.vertex_num = 6890
        smpl_layer = SMPL_Layer(gender='neutral', model_root=self.I2LMeshNet_cfg.smpl_path + '/smplpytorch/native/models')
        self.face = smpl_layer.th_faces.numpy()
        self.joint_regressor = smpl_layer.th_J_regressor.numpy()
        self.root_joint_idx = 0

        torch.backends.cudnn.benchmark = True   # Required -- enables parallel
        self.detector_model = self.load_mesh_detector(mesh_detector_model_path=self.mseh_detector_model_path)
        #self.detector_model = self.detector_model.to(self.device)

        # prepare input image
        self.input_img_transform = transforms.ToTensor()


    def detect_mesh_body_on_single_frame(self, img, bbox=None):
        original_img = img
        original_img_height, original_img_width = original_img.shape[:2]

        if bbox is None:
            margin = 5.0
            bbox = [margin, margin, original_img_width - margin, original_img_height - margin]

        # prepare bbox
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, self.I2LMeshNet_cfg.input_img_shape)
        img = self.input_img_transform(img.astype(np.float32)) / 255
        img = img.cuda()[None, :, :, :]

        # forward
        inputs = {'img': img}
        targets = {}
        meta_info = {'bb2img_trans': bb2img_trans}
        with torch.no_grad():
            out = self.detector_model(inputs, targets, meta_info, 'test')
        img = img[0].cpu().numpy().transpose(1, 2, 0)  # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
        mesh_lixel_img = out['mesh_coord_img'][0].cpu().numpy()
        mesh_param_cam = out['mesh_coord_cam'][0].cpu().numpy()

        # restore mesh_lixel_img to original image space and continuous depth space
        mesh_lixel_img[:, 0] = mesh_lixel_img[:, 0] / self.I2LMeshNet_cfg.output_hm_shape[2] * self.I2LMeshNet_cfg.input_img_shape[1]
        mesh_lixel_img[:, 1] = mesh_lixel_img[:, 1] / self.I2LMeshNet_cfg.output_hm_shape[1] * self.I2LMeshNet_cfg.input_img_shape[0]
        mesh_lixel_img[:, :2] = np.dot(bb2img_trans,
                                       np.concatenate((mesh_lixel_img[:, :2], np.ones_like(mesh_lixel_img[:, :1])),
                                                      1).transpose(1, 0)).transpose(1, 0)
        mesh_lixel_img[:, 2] = (mesh_lixel_img[:, 2] / self.I2LMeshNet_cfg.output_hm_shape[0] * 2. - 1) * (self.I2LMeshNet_cfg.bbox_3d_size / 2)

        # root-relative 3D coordinates -> absolute 3D coordinates
        focal = (1500, 1500)
        princpt = (original_img_width / 2, original_img_height / 2)
        root_xy = np.dot(self.joint_regressor, mesh_lixel_img)[self.root_joint_idx, :2]
        root_depth = 11250.5732421875  # obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)
        root_depth /= 1000  # output of RootNet is milimeter. change it to meter
        root_img = np.array([root_xy[0], root_xy[1], root_depth])
        root_cam = pixel2cam(root_img[None, :], focal, princpt)
        mesh_lixel_img[:, 2] += root_depth
        mesh_lixel_cam = pixel2cam(mesh_lixel_img, focal, princpt)
        mesh_param_cam += root_cam.reshape(1, 3)
        mesh_param_img = cam2pixel(mesh_param_cam, focal, princpt)

        #mesh_object = mesh_lixel_cam
        mesh_objects = [mesh_lixel_cam, mesh_param_cam, mesh_lixel_img, mesh_param_img]
        return mesh_objects

    def load_mesh_detector(self, mesh_detector_model_path):
        # snapshot load
        assert os.path.exists(mesh_detector_model_path), 'Cannot find model at ' + mesh_detector_model_path
        print('Load checkpoint from {}'.format(mesh_detector_model_path))
        model = get_2LMeshNet_model(self.vertex_num, self.joint_num, 'test')
        model = DataParallel(model).cuda()

        ckpt = torch.load(mesh_detector_model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.mseh_detector_model_path = self.mseh_detector_model_path
        return model

    def render_mesh_on_frame(self, img, mesh_object):
        # render mesh from lixel
        focal = (1500, 1500)
        img_height, img_width = img.shape[:2]
        princpt = (img_width / 2, img_height / 2)

        # render mesh from lixel
        vis_img = img.copy()
        mesh_lixel_cam = mesh_object
        rendered_img = render_mesh(vis_img, mesh_lixel_cam, self.face, {'focal': focal, 'princpt': princpt})

        return rendered_img

    def get_img_keypoints_from_mesh_object(self, mesh_object):
        mesh_lixel_img = mesh_object
        pose_coord = np.dot(self.joint_regressor, mesh_lixel_img)

        return pose_coord

    def vis_keypoints_on_frame(self, img, mesh_object):

        vis_img = img.copy()
        mesh_lixel_img = mesh_object

        # kps_lines = None # TODO
        kps = self.get_img_keypoints_from_mesh_object(mesh_object=mesh_lixel_img)

        rendered_img = vis_keypoints(vis_img, kps, alpha=1)
        #rendered_img = vis_mesh(vis_img, pose_coord)

        return rendered_img

    def detect_on_image(self, img_path, save_path):
        return super().detect_on_image(img_path, save_path)
