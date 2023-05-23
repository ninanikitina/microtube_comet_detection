import cv2
import numpy as np
import glob
import os


if __name__ == '__main__':

    filename = "Experiment-246"
    folder1_path = r'D:\BioLab\microtub_comet_detecting\temp\raw_img'
    folder2_path = r"D:\BioLab\microtub_comet_detecting\temp\reconstructed_mask"
    video_path = os.path.join("movies/",
                              filename + "_double" + '.mov')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frameSize = (2048 * 2, 2048)
    video = cv2.VideoWriter(video_path, fourcc, 3, frameSize)
    #
    # a = r"D:\BioLab\microtub_comet_detecting\analysis_data\tiff_for_matlab\Experiment-246"
    # for filename_1 in glob.glob(os.path.join(a, "*.tif")):
    #     print(filename_1)



    for filename_1 in glob.glob(os.path.join(folder1_path, "*.png")):

        for filename_2 in glob.glob(os.path.join(folder2_path, "*.png")):
            if os.path.basename(filename_1) == os.path.basename(filename_2):
                print(os.path.basename(filename_1))
                img_1 = cv2.imread(filename_1)
                img_2 = cv2.imread(filename_2)
                im_h = cv2.hconcat([img_2, img_1])
                video.write(im_h)
                print(f"/n{filename_1} and {filename_2}")
                break
    cv2.destroyAllWindows()
    video.release()


