import os
from objects.BioformatReader import BioformatReader
from objects.ImageData import ImageData
import sys
import csv
import glob
import math
import cv2.cv2 as cv2
import numpy as np
from unet.predict import run_predict_unet

temp_folders = {
    "cut_8bit_img": 'temp/cut_img_for_unet',
    "cut_mask": 'temp/cut_mask',
    "img_w_cnt": 'temp/reconstructed_mask',
    "img_raw": 'temp/raw_img'
}

analysis_data_folders = {
    "for_matlab": 'analysis_data/tiff_for_matlab',
    "analysis": 'analysis_data/stat',
    "cnts_verification": 'analysis_data/nuclei_area_verification',
}


def make_padding(img, final_img_size):
    """
    Create padding for provided image
    :param img: image to add padding to
    :param final_img_size: tuple
    :return: padded image
    """
    h, w = img.shape[:2]
    h_out, w_out = final_img_size

    top = (h_out - h) // 2
    bottom = h_out - h - top
    left = (w_out - w) // 2
    right = w_out - w - left

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img


def cut_image(img, img_name, cut_img_size, output_folder):
    """
    Cut image (img) into smaller images of provided size (cut_img_size).
    Save small images to the specified folder with a distinctive name pattern: "img_name_y-Y_x-X.png"
    where Y and X are indexes of small cut images in the initial large image.
    Such naming needed for further image reconstruction.
    :param img: image to cut
    :param img_name: image name
    :param cut_img_size: size of final images
    :param output_folder: folder to save files to
    :return:
    """
    base_img_name = os.path.splitext(os.path.basename(img_name))[0]
    padded_img_size = (math.ceil(img.shape[0] / cut_img_size[0]) * cut_img_size[0],
                       math.ceil(img.shape[1] / cut_img_size[1]) * cut_img_size[1])

    padded_img = make_padding(img, padded_img_size)
    y_start = 0
    y_end = cut_img_size[0]
    y_order = 0
    while (padded_img_size[0] - y_end) >= 0:
        x_start = 0
        x_end = cut_img_size[1]
        x_order = 0
        while (padded_img_size[1] - x_end) >= 0:
            current_img = padded_img[y_start:y_end, x_start:x_end]
            img_path = os.path.join(output_folder,
                                    base_img_name + "_y-" + str(y_order) + '_x-' + str(x_order) + '.png')
            cv2.imwrite(img_path, current_img)
            x_start = x_end
            x_end = x_end + cut_img_size[1]
            x_order = x_order + 1
        y_start = y_end
        y_end = y_end + cut_img_size[0]
        y_order = y_order + 1
    return math.ceil(img.shape[0] / cut_img_size[0])


def prepare_folder(folder):
    """
    Create folder if it has not been created before
    or clean the folder
    ---
    Parameters:
    -   folder (string): folder's path
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    for f in glob.glob(folder + "/*"):
        os.remove(f)


def stitch_mask(input_folder, unet_img_size, num):
    img_col = []
    for i in range(num):
        img_row = []
        for img_path in glob.glob(os.path.join(input_folder, f"*_y-{i}_*.png")):
            nucleus_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_row.append(nucleus_img)
        img_col.append(cv2.hconcat(img_row))
    stitched_img = cv2.vconcat(img_col)
    # img_no_padding = remove_padding(stitched_img) #TODO this function shuld be created in case if img was padded before
    # cv2.imshow("stitched_mask", cv2.resize(stitched_img, (750, 750)))  # keep it for debugging
    # cv2.waitKey()
    return stitched_img


class Analyzer(object):
    def __init__(self, bioformat_imgs_path, mask_recognition_mode, nuc_threshold=None, unet_parm=None,
                 min_pixel_num=0, mask_channel_name="DAPI"):
        self.imgs_path = bioformat_imgs_path
        self.mask_recognition_mode = mask_recognition_mode
        self.nuc_threshold = nuc_threshold
        self.unet_parm = unet_parm
        self.min_pixels_num = min_pixel_num
        self.mask_channel_name = mask_channel_name

    def run_analysis(self):


        for folder in analysis_data_folders:
            prepare_folder(analysis_data_folders[folder])

        for i, filename in enumerate(os.listdir(self.imgs_path)):
            for folder in temp_folders:
                prepare_folder(temp_folders[folder])
            reader = BioformatReader(self.imgs_path, i, self.mask_channel_name)

            segmented_imgs = []
            raw_imgs = []
            for t_frame in range(reader.t_num):

                if self.mask_recognition_mode == 'unet':
                    nuc_mask = self.find_mask_based_on_unet(reader, t_frame)

                elif self.mask_recognition_mode == 'thr':
                    nuc_mask = self.find_mask_based_on_thr(reader, t_frame)
                else:
                    print("The recognition mode is not specified or specified incorrectly. Please use \"unet\" or \"thr\"")
                    sys.exit()

                channels_raw_data = reader.save_norm_raw_img(t_frame, temp_folders["img_raw"])
                img_data = ImageData(filename, channels_raw_data, nuc_mask, self.min_pixels_num, t_frame)
                img_with_cntrs = img_data.draw_and_save_cnts_for_channels(temp_folders["img_w_cnt"],
                                                         self.min_pixels_num, t_frame)
                img_data.save_tiff_for_matlab_input(analysis_data_folders["for_matlab"],
                                                    self.min_pixels_num, t_frame)
                segmented_imgs.append(img_with_cntrs)
                raw_imgs.append(channels_raw_data[0].img)
            filename = os.path.splitext(filename)[0]
            self.create_movie(filename, "mask", temp_folders["img_w_cnt"])
            self.create_movie(filename, "raw", temp_folders["img_raw"])
            self.create_double_movie(filename, temp_folders["img_w_cnt"], temp_folders["img_raw"])

    def create_double_movie(self, filename, folder1_path, folder2_path):
        video_path = os.path.join("movies/",
                                    filename + "_double" +'.avi')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        frameSize = (2048 * 2, 2048)
        video = cv2.VideoWriter(video_path, fourcc, 3, frameSize)
        for filename_1 in glob.glob(os.path.join(folder1_path,"*.png")):
            for filename_2 in glob.glob(os.path.join(folder2_path,"*.png")):
                if os.path.basename(filename_1) == os.path.basename(filename_2):
                    print(os.path.basename(filename_1))
                    img_1 = cv2.imread(filename_1)
                    img_2 = cv2.imread(filename_2)
                    im_h = cv2.hconcat([img_2, img_1])
                    video.write(im_h)
                    break
        cv2.destroyAllWindows()
        video.release()

    def create_movie(self, filename, identifier, folder_path):
        video_path = os.path.join("movies/",
                                    filename + "_" +identifier+'.avi')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        frameSize = (2048, 2048)
        video = cv2.VideoWriter(video_path, fourcc, 1, frameSize)
        for filename in glob.glob(os.path.join(folder_path,"*.png")):
            img = cv2.imread(filename)
            video.write(img)

        cv2.destroyAllWindows()
        video.release()


    # TODO: implement this function. The algorithm shuld be simular to finding mask in MatLab program. Apply filtering(noise redution) and theshold provided by user.
    def find_mask_based_on_thr(self, reader, t_frame):
        # use self.nuc_threshold
        # reader.depth -> 'uint8' -> 0..255 or 'uint16' ->
        nuc_mask = 1
        return nuc_mask

    def find_mask_based_on_unet(self, reader, t_frame):
        """
        Finds mask picture based on unet model. Since my GPU can handle only 512*512 images for prediction
        :param reader:
        :return:
        """
        nuc_img_8bit_norm, nuc_file_name = reader.read_mask_layers(norm=True, t_frame=t_frame)
        pieces_num = cut_image(nuc_img_8bit_norm, nuc_file_name, self.unet_parm.unet_img_size,
                               temp_folders["cut_8bit_img"])

        run_predict_unet(temp_folders["cut_8bit_img"], temp_folders["cut_mask"],
                         self.unet_parm.unet_model_path,
                         self.unet_parm.unet_model_scale,
                         self.unet_parm.unet_model_thrh)
        nuc_mask = stitch_mask(temp_folders["cut_mask"], self.unet_parm.unet_img_size, pieces_num)
        return nuc_mask

    def _remove_small_particles(self, mask):
        mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
        clean_mask = np.zeros(mask.shape)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < self.min_pixels_num:
                continue
            cv2.fillPoly(clean_mask, pts=[cnt], color=(255, 255, 255))
        return clean_mask.astype('uint8')
