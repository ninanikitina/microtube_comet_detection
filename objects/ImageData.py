import numpy as np
import cv2.cv2 as cv2
import os
import tifffile as tiff
from objects import Contour
from objects.Structures import NucAreaData, Signal
from libtiff import TIFF


class ImageData(object):
    def __init__(self, path, channels_raw_data, nuc_mask, nuc_area_min_pixels_num, t_frame):
        self.path = path
        self.channels_raw_data = channels_raw_data
        self.nuc_mask = nuc_mask
        self.cnts = Contour.get_mask_cnts(self.nuc_mask)
        self.t_frame = t_frame
        self.cells_data, self.cells_num = self._analyse_signal_in_nuc_area(nuc_area_min_pixels_num)


    def _analyse_signal_in_nuc_area(self, nuc_area_min_pixels_num):
        nuclei_area_data = []
        for cnt in self.cnts:
            mask = Contour.draw_cnt(cnt, self.nuc_mask.shape)
            center = Contour.get_cnt_center(cnt)
            area = cv2.contourArea(cnt)
            if area < nuc_area_min_pixels_num:  # if it is noise not a nuc
                continue
            nucleus_area_data = NucAreaData(center, area)
            signals = []
            for channel in self.channels_raw_data:
                cut_out_signal_img = np.multiply(mask, channel.img)
                signal_sum = np.matrix.sum(np.asmatrix(cut_out_signal_img))
                signal = Signal(channel.name, signal_sum)
                signals.append(signal)

            nucleus_area_data.update_signals(signals)
            nuclei_area_data.append(nucleus_area_data)
        return nuclei_area_data, len(nuclei_area_data)

    def draw_and_save_cnts_for_channels(self, output_folder, nuc_area_min_pixels_num, t_frame):
        base_img_name = os.path.splitext(os.path.basename(self.path))[0]
        cnts = [cnt for cnt in self.cnts if cv2.contourArea(cnt) > nuc_area_min_pixels_num]
        merged_img = []

        for channel in self.channels_raw_data:
            img_path = os.path.join(output_folder,
                                    base_img_name + '_' + channel.name + '.png')
            img_8bit = self.normalization(channel.img, 3000)
            # img_8bit = cv2.normalize(channel.img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.drawContours(img_8bit, cnts, -1, (255, 255, 50), 3)
            # cv2.imwrite(img_path, img_8bit) #uncomment if needed to save all channels separately for verification
            merged_img.append(img_8bit)

        color_img_path = os.path.join(output_folder,
                                     base_img_name + '_t_frame_' + str(t_frame).rjust(3, '0') + '.png')
        color_img = cv2.merge(merged_img)
        cv2.imwrite(color_img_path, color_img)
        return color_img

    def save_tiff_for_matlab_input(self, output_folder, nuc_area_min_pixels_num, t_frame):
        base_folder_name = os.path.splitext(os.path.basename(self.path))[0]
        cnts = [cnt for cnt in self.cnts if cv2.contourArea(cnt) > nuc_area_min_pixels_num]
        img_name = "MyMovie" + str(t_frame).rjust(3, '0') +'.tif'
        img_folder = output_folder + "/" + base_folder_name

        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        img_path = os.path.join(img_folder, img_name)
        img = self.channels_raw_data[0].img

        cv2.drawContours(img, cnts, -1, (2**16 - 1, 2**16 - 1, 2**16 - 1), 3)
        # tiff.imsave(img_path, img)
        # cv2.imwrite(img_path, img)

        tif = TIFF.open(img_path, mode='w')
        tif.write_image(img)

        a = 1

    def normalization(self, img, norm_th):
        img[np.where(img > norm_th)] = norm_th
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return img


