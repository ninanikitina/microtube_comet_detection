import time
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
import javabridge
import bioformats


def main():
    bioformat_imgs_path = r"D:\BioLab\img\Confocal_img\2023.03_EB1_Tracking_Test_#3\test_run"  # path to the folder that contains bio format images (czi, lif, ect) or path to the specific image
    mask_recognition_mode = "unet"  # "unet" or "trh" TODO Implement threshold option. Now only unet mode work
    mask_channel_name = "EGFP"

    unet_model = r"unet\models\CP_epoch198.pth"

    # Unet training process characteristics:
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    nuc_area_min_pixels_num = 0
    unet_parm = UnetParam(unet_model, unet_model_scale, unet_model_thrh, unet_img_size)
    nuc_theshold = 30
    javabridge.start_vm(class_path=bioformats.JARS)
    #TODO: add analysis_type variable mode_1:signal intensity mode_2: cells counting over time

    start = time.time()
    analyser = Analyzer(bioformat_imgs_path, mask_recognition_mode, nuc_theshold, unet_parm, nuc_area_min_pixels_num,
                        mask_channel_name)
    analyser.run_analysis()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


if __name__ == '__main__':
    main()
