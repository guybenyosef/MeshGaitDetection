import sys
import os
import argparse

from GaitEmbedding.VanillaGaitEmbedder import VanillaGaitEmbedder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--input_path', type=str, dest='input_path', default='storage/_shared-data5_guy_data_forge_west_entrance_cam_2021.04.29.16.36.40_73_')
    parser.add_argument('-o', '--output_path', type=str, dest='output_path', default='storage')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    # argument parsing
    args = parse_args()

    GaitEmbedder = VanillaGaitEmbedder()

    #img_path = args.input_image_path #'input.jpg'
    # outputfolder_name = os.path.join(args.output_folder, "{}_mesh_output".format(
    #         os.path.splitext(os.path.basename(args.input_image_path))[0]))

    if os.path.isdir(args.input_path):
        em1 = GaitEmbedder.get_embedding_for_frame_sequence(folder_path=args.input_path)
    else:
        sys.exit("Not a directory")
