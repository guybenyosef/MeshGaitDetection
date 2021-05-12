import os
import argparse

from MeshDetectors.I2LMeshNetBodyMeshDetector import I2LMeshNetBodyMeshDetector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--input_path', type=str, dest='input_path', default='Samples/image2.jpg')
    #parser.add_argument('-b', '--bbox', type=float, dest='bbox', default=[139.41, 102.25, 222.39, 241.57])
    parser.add_argument('-o', '--output_path', type=str, dest='output_path', default='storage')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    # argument parsing
    args = parse_args()

    MeshDetector = I2LMeshNetBodyMeshDetector()

    #img_path = args.input_image_path #'input.jpg'
    # outputfolder_name = os.path.join(args.output_folder, "{}_mesh_output".format(
    #         os.path.splitext(os.path.basename(args.input_image_path))[0]))

    if os.path.isdir(args.input_path):
        MeshDetector.detect_on_folder(folder_path=args.input_path, save_path=args.output_path, compose_visualization_video=True )
    else:
        MeshDetector.detect_on_image(img_path=args.input_path, save_path=args.output_path)


#s