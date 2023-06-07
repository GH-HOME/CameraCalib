# Camera calibration with Charuco marker

## Requirements
```
pip install opencv-python
pip install opencv-contrib-python==4.5.5.62
pip install numpy
pip install glob
pip install argparse
```


## How to build your marker board
- Go to [this page to generate marker pattern](https://calib.io/pages/camera-calibration-pattern-generator). You can find one example in `supp/calib.io_charuco_200x150_8x11_15_12_DICT_4X4.pdf`
- Print it to an A4 paper
- Paste it to a planar board


## Parameters

Please find the following parameters in [this page](https://calib.io/pages/camera-calibration-pattern-generator) when you generate the pattern 

- Charuco marker size: [White cell size, Black cell size] White cell size is corresponding to Checker size in the generated pattern. Black cell size can be calculated as `0.7462667 * Checker size`.
- Charuco marker dictionary: e.g. Aruco DICT_5x5
- Marker division: e.g. [25, 18]
- Image size: e.g. [2048, 1500]
- CameraID: e.g. `FLIR_16mm`

## Run 
- Put all image observations of the pattern in one folder, e.g. `supp/pattern_img`, you can find image examples [here](https://drive.google.com/drive/folders/1w_VAt4NYT4TeOBuxKc9zEmJA08rVFynX?usp=sharing)
- Run `python camera_calib_by_charuco_marker.py --pattern_dir supp/pattern_img --marker_division 25, 18 --board_size 7.5, 5.597 --image_size 2048 1500 --aruco_dict DICT_5X5`
- Find the calibration result in `supp/pattern_img`, where camera intrinsic is saved in `params_chaurco_FLIR_16mm_img_1500_2048.npz`
- Check the `reprojection_FLIR_16mm.avi` to see the reprojection error qualitatively.
- If you use the camera intrinsic in `params_camera_undist_img_size_1500_2048.npz` for undistorted version, please remember to undistort the input images after the calibration via

```
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
```
where `mapx` and `mapy` are saved in `params_camera_undist_img_size_1500_2048.npz`. After this undistortion process, you can use the the camera intrinsic in `params_camera_undist_img_size_1500_2048.npz` for the undistorted images.