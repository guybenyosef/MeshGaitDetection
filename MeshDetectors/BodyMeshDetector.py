import sys
import os
import cv2
import glob
import numpy as np
import ipdb

import torchvision.transforms as transforms

class BodyMeshDetector():
    '''
    Class used for body mesh detector in a video or image file.
    '''
    def __init__(self):
        # set up device:
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Device is: {}".format(self.device))

        # set up detector:
        self.DetectorType = 'General'
        self.detector_model = None

        # set up transformer:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.input_img_transform = transforms.ToTensor()

    def detect_mesh_body_on_single_frame(self, img, bbox=None):
        mesh_objects = None
        return mesh_objects

    def load_mesh_detector(self, mesh_detector_model_path):
        model = None
        return model

    def render_mesh_on_frame(self, img, mesh_object):
        rendered_img = None
        return rendered_img

    def vis_keypoints_on_frame(self, img, mesh_object):
        rendered_img = None
        return rendered_img

        #vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1

    def detect_on_image(self, img_path, save_path):
        img = cv2.imread(img_path)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        output_img_path = os.path.join(save_path, "{}_on_{}.jpg".format(self.DetectorType, image_name))
        output_kpts_path = os.path.join(save_path, "{}_on_{}.kpts".format(self.DetectorType, image_name))

        # TODO: add other plots to folder:
        # mesh_output_folder = os.path.join(save_path, "{}_on_{}_mesh_output".format(self.DetectorType,
        #                                                                            os.path.splitext(image_name)[0]))
        # if not os.path.exists(mesh_output_folder):
        #     os.makedirs(mesh_output_folder)

        mesh_objects = self.detect_mesh_body_on_single_frame(img)
        mesh_lixel_cam = mesh_objects[0]
        mesh_lixel_img = mesh_objects[2]
        rendered_img = self.render_mesh_on_frame(img, mesh_lixel_cam)
        rendered_kpts_img = self.vis_keypoints_on_frame(img, mesh_lixel_img)

        kpts = self.get_img_keypoints_from_mesh_object(mesh_lixel_img)

        blank_frame = img * 0 + 200
        frame = np.concatenate((img, blank_frame, rendered_img, blank_frame, rendered_kpts_img), axis=1)
        cv2.imwrite(output_img_path, frame)
        np.savez(output_kpts_path, kpts=kpts)
        print("Saving output to {}".format(output_img_path))

        return output_img_path

    def detect_on_folder(self, folder_path, save_path, compose_visualization_video=False):

        all_files = []
        if os.path.isdir(folder_path):
            all_files = glob.glob(os.path.join(folder_path, "*.jpg"))

        save_to_folder = os.path.join(save_path, folder_path.replace("/", "_"))
        if not os.path.exists(save_to_folder):
            os.makedirs(save_to_folder)

        all_output_img_path = []
        for img_path in all_files:
            output_img_path = self.detect_on_image(img_path=img_path, save_path=save_to_folder)
            all_output_img_path.append(output_img_path)

        if compose_visualization_video:
            im_sh = cv2.imread(output_img_path).shape
            save_video_path = os.path.join(save_path, "{}_on{}.mp4".format(self.DetectorType, folder_path.replace("/", "_")))
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            fixed_w = 2000
            fixed_h = 1000
            out_video = cv2.VideoWriter(save_video_path, fourcc, 2.0, (fixed_w, fixed_h))# img.shape --> row (height), column (width), color (3)

            for i in range(len(all_output_img_path)):
                # in_frame = cv2.imread(all_files[i])
                # blank_frame = cv2.imread(all_output_img_path[i]) * 0 + 200
                frame = cv2.imread(all_output_img_path[i])
                #print(frame.shape)
                h = int(np.ceil((fixed_h-frame.shape[0]) * 0.5))
                w = int(np.ceil((fixed_w - frame.shape[1]) * 0.5))
                #ipdb.set_trace()
                frame = cv2.copyMakeBorder(frame, h, h, w, w, cv2.BORDER_CONSTANT)
                frame = frame[:fixed_h, :fixed_w, :]
                #print(frame.shape)
                #frame = np.concatenate((in_frame, blank_frame, out_frame), axis=1)
                out_video.write(frame)
            out_video.release()
            cv2.destroyAllWindows()
            print("Saved video to {}".format(save_video_path))