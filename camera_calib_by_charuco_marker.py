from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from cv2 import aruco
import numpy as np

################################################################
# notice the size and scale of the image should change, search 1024 and self.scale
################################################################

from matplotlib import pyplot as plt
import glob
import  os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


class charuco_marker(object):
    """
    The basic operation of aruco marker
    """

    def __init__(self, board_size, marker_division, aruco_dict, camera_id):
        """
        :param board_size: The actual size of marker size and margin. e,g.[5, 1] means 5mm and 1mm
        :param marker_division: The number of marker in vertical and horizontal direction. e.g. [5, 7]
        """

        if aruco_dict == 'DICT_5X5':
            dict = aruco.DICT_5X5_1000
        elif aruco_dict == 'DICT_4x4':
            dict = aruco.DICT_4x4_1000
        elif aruco_dict == 'DICT_6X6':
            dict = aruco.DICT_6X6_1000
        elif aruco_dict == 'DICT_7X7':
            dict = aruco.DICT_7X7_1000
        else:
            print('The aruco_dict is not supported')

        self.aruco_dict = aruco.getPredefinedDictionary(dict)

        assert board_size is not None
        assert marker_division is not None
        squareLength = board_size[0]
        markerLength = board_size[1]

        self.board = aruco.CharucoBoard_create(marker_division[0], marker_division[1], squareLength, markerLength,
                                               self.aruco_dict)
        # self.board = aruco.GridBoard_create(marker_division[0], marker_division[1], squareLength, markerLength, self.aruco_dict)
        self.allIds = []
        self.allCorners2d = []
        self.allCorners3d = []
        self.valid_img_index = []
        self.total_cornersNum = (marker_division[0] - 1) * (marker_division[1] - 1)
        self.rejectThres = 1#0.15 * self.total_cornersNum
        self.camera_id = camera_id


    def detect_corners_oneimg(self, img, drawflag=False, draw_path=None):
        """
        detect 2D and 3D correspondence in one image
        :param img: image [H, W, c] read from opencv
        :return: 2D and its ids
        """
        assert img is not None

        if len(img.shape) == 2:
            img = img[:,:,np.newaxis]

        if img.shape[2] != 1:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, self.aruco_dict, parameters = parameters)
        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(img_gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            _, corners2, ids2 = aruco.interpolateCornersCharuco(corners, ids, img_gray, self.board)

            if corners2 is not None and ids2 is not None and len(corners2) > 3:
                if drawflag:
                    showim = aruco.drawDetectedMarkers(img, corners, ids)
                    cv2.imshow('', showim)
                    cv2.waitKey(10)
                return [corners2, ids2]
        return [None, None]

    def find_and_draw_correspondances(self, img, corners, ids, draw_flag=False):
        """
        :param img: original image
        :param corners: detected 2d corner points
        :param ids:  each id for 2d point
        :param draw_flag: True/False to select whether or not draw
        :return: [3D corner, 2D corner]
        """
        assert img is not None
        assert corners is not None
        assert ids is not None

        if draw_flag:
            # step 1: create a empty canvas to draw the 3D point position
            canvas_3d = np.ones_like(img) * 255
            draw_scale = 1  # the scale for draw 3D points
            point_scale = 1

        # step 2: find ind the corresponding 2D and 3D points
        obj_point3d = []
        img_point2d = []
        for i in range(len(ids)):
            id = ids[i][0]
            point3d = self.board.chessboardCorners[id]  # shape [3]
            point2d = corners[i].squeeze()             # shape [2]

            if draw_flag:
                # step 3: draw 2D and 3D point
                cv2.namedWindow("2d output", cv2.WINDOW_NORMAL)
                cv2.circle(img, (int(point2d[0]), int(point2d[1])), 7, (255, 0, 0), thickness=-1)
                cv2.putText(img, "{}".format(id), (int(point2d[0]), int(point2d[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.5*point_scale, (0,0,255),4)
                cv2.imshow("2d output", img)

                cv2.namedWindow("3d output", cv2.WINDOW_NORMAL)
                cv2.circle(canvas_3d, (int(point3d[0]*draw_scale), int(point3d[1]*draw_scale)), 5*point_scale, (255, 0, 0), thickness=-1)
                cv2.putText(canvas_3d, "{}".format(id), (int(point3d[0]*draw_scale), int(point3d[1]*draw_scale)), cv2.FONT_HERSHEY_TRIPLEX, 0.5*point_scale, (0, 0, 255),
                            1)
                cv2.imshow("3d output", canvas_3d)

                cv2.waitKey(5)
            obj_point3d.append(point3d)
            img_point2d.append(corners[i])

        obj_point3d = np.array(obj_point3d)
        img_point2d = np.array(img_point2d)

        return [obj_point3d, img_point2d]


    def reproject(self, camerapara, pathlist_path, save_folder_path, showflag = False, size=None):
        """
        :param camerapara:
        :param obj_point3ds:
        :param img_point2ds:
        :param img: RGB image
        :param showflag:  green point means GT points, blue points means re-projection points.
        :return: None
        """

        print('==> Now begin reprojection for debug. \n Blue and red points are re-projection and GT points respectively. \n They should be overlapped.')
        error_list = []

        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        videoWriter = cv2.VideoWriter(os.path.join(save_folder_path, 'reprojection_{}.avi'.format(self.camera_id)),
                                      fourcc, 24, (self.img_size[0], self.img_size[1]))

        tol_error = 0

        intrinsic, dist, rvecs, tvecs, img_size = camerapara


        obj_point3ds = self.allCorners3d
        img_point2ds = self.allCorners2d

        imgidx = self.valid_img_index[0]
        ipath = pathlist_path[imgidx]
        img = cv2.imread(ipath)
        img = cv2.resize(img, (size[0], size[1]))
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic, dist, (w, h), 1, (w, h))

        for i in range(len(self.valid_img_index)):
            points3d = obj_point3ds[i]
            points2d = img_point2ds[i]

            points2d_project, _ = cv2.projectPoints(points3d, rvecs[i], tvecs[i], intrinsic, distCoeffs = dist)
            error = cv2.norm(points2d, points2d_project, cv2.NORM_L2) / len(points3d)
            error_list.append(error)

            tol_error = tol_error + error
            print("error {}: {}".format(i, error))

            if showflag and pathlist_path is not None:
                imgidx = self.valid_img_index[i]
                ipath = pathlist_path[imgidx]
                img = cv2.imread(ipath)
                img = cv2.resize(img, (size[0], size[1]))
                # img = cv2.undistort(img, intrinsic, dist, None, newcameramtx)
                # x, y, w, h = roi
                # img = dst[y:y + h, x:x + w]
                for j in range(len(points2d_project)):
                    if showflag and img is not None:
                        cv2.namedWindow("reprojection output", cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
                        cv2.circle(img, (int(points2d_project[j][0,0]), int(points2d_project[j][0,1])), 2, (255, 0, 0), thickness=-1)
                        cv2.circle(img, (int(points2d[j][0,0]), int(points2d[j][0,1])), 3, (0, 255, 0), thickness=2)

                        cv2.imshow("reprojection output", img)
                        cv2.waitKey(100)
                videoWriter.write(img)

                # cv2.waitKey(20)
        print("average reprojection error is : ", tol_error / len(self.valid_img_index))
        videoWriter.release()
        return np.array(error_list)


    def get_undist_para(self, camerapara,  save_folder_path):

        intrinsic, dist, rvecs, tvecs, img_size = camerapara
        w, h = img_size
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic, dist, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(intrinsic, dist, None, newcameramtx, (w, h), 5)
        new_dist = np.zeros_like(dist)

        # how to use after?
        # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        # then newcameramtx is the camera intrinsic for the image dst without any distortion

        np.savez(os.path.join(save_folder_path, "params_camera_undist_img_size_{}_{}.npz".format(h, w)), intrinsic=newcameramtx, dist=new_dist, rvecs = rvecs,
                 tvecs= tvecs, mapx = mapx, mapy = mapy, roi = roi)

        return [newcameramtx, new_dist, roi, mapx, mapy]



    def calibrateSingleCamerafromImg(self, pathlist_path, parameter_path=None, save_folder_path = None, size = None):
        assert  pathlist_path is not None

        if parameter_path is not None:
            para = np.load(parameter_path)
            self.allIds, self.allCorners2d, self.allCorners3d, self.valid_img_index = para['ids'], para['corners2d'], para['corners3d'], para['valid_idx']
        else:
            imgNum = len(pathlist_path)
            self.img_size = size
            img = cv2.imread(pathlist_path[0])
            # img = np.load(pathlist_path[0])

            self.scale = img.shape[1] / size[0]
            valid_path = []
            for i in range(imgNum):
                ipath = pathlist_path[i]
                print("===> now process {} frame with path {}".format(i, ipath))

                # img = np.load(ipath)
                # img = np.uint8(img*255)
                img = cv2.imread(ipath)
                if size is not None:
                    img = cv2.resize(img, (size[0], size[1]))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # img_gray = np.uint8(img / 256)
                img_gray = cv2.equalizeHist(img_gray)



                assert img_gray is not None
                corners, ids = self.detect_corners_oneimg(img_gray, True)

                if corners is not None:
                    [obj_3d_pts, img_2d_pts] = self.find_and_draw_correspondances(img, corners, ids)

                    if obj_3d_pts is not None and img_2d_pts is not None and len(img_2d_pts) > self.rejectThres:
                        self.allIds.append(ids)
                        self.allCorners2d.append(img_2d_pts)
                        self.allCorners3d.append(obj_3d_pts)
                        self.valid_img_index.append(i)
                        valid_path.append(ipath)

            self.allIds = np.array(self.allIds)
            self.allCorners2d = np.array(self.allCorners2d)
            self.allCorners3d = np.array(self.allCorners3d)
            self.valid_img_index = np.array(self.valid_img_index)

            np.savez(os.path.join(save_folder_path, "id_corner_chaurco_{}.npz".format(self.camera_id))
                     , ids=np.array(self.allIds), corners2d=np.array(self.allCorners2d),
                     corners3d = np.array(self.allCorners3d), valid_idx = self.valid_img_index)

        print("==>Now calibrate the camera")
        f = 16 # in mm
        pixel_size = 3.45e-3 # in mm/pixel
        f_x =  f / pixel_size / self.scale
        cameraMatrixInit = np.array([[f_x, 0, self.img_size[0] / 2.],
                                     [0, f_x, self.img_size[1] / 2.],
                                     [0., 0., 1.]])

        distCoeffsInit = np.zeros((5, 1))
        flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO
        # flags = (cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_USE_INTRINSIC_GUESS  + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT)

        res, intrinsic, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.allCorners3d, self.allCorners2d, (self.img_size[0], self.img_size[1]),
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoeffsInit,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        camera_para = [intrinsic, dist, rvecs, tvecs, self.img_size]
        intrinsic_scaled = intrinsic * self.scale
        intrinsic_scaled[2,2] = 1
        print("reprojection error is", res)
        print(intrinsic_scaled)
        print(dist)
        print('valid_paths', valid_path)

        np.savez(os.path.join(save_folder_path, "params_chaurco_{}_img_{}_{}.npz".format(self.camera_id, self.img_size[1]*self.scale, self.img_size[0]*self.scale)),
                 res=res, intrinsic=intrinsic_scaled, dist=dist, tvecs = tvecs, rvecs = rvecs, img_size = self.img_size)

        np.savez(os.path.join(save_folder_path, "params_chaurco_{}_img_{}_{}.npz".format(self.camera_id, self.img_size[1], self.img_size[0])),
                 res=res, intrinsic=intrinsic, dist=dist, tvecs = tvecs, rvecs = rvecs, img_size = self.img_size)

        np.savetxt(os.path.join(save_folder_path, "valid_path.csv"), np.array(valid_path), fmt='%s', delimiter = ',')


        return camera_para


    def calibrateSingleCamerafromrechooseIdx(self, index):

        obj_3d_pts = self.allCorners3d[index]
        img_2d_pts = self.allCorners2d[index]
        img_size = self.img_size
        self.valid_img_index = self.valid_img_index[index]

        print("==>Now calibrate the camera")

        distCoeffsInit = np.zeros((5, 1))
        flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_TANGENT_DIST)
        res, intrinsic, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_3d_pts, img_2d_pts, img_size,
            distCoeffs=distCoeffsInit,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        camera_para = [intrinsic, dist, rvecs, tvecs]

        print("reprojection error is", res)
        print(intrinsic)
        print(dist)
        np.savez("params_chaurco.npz", res=res, intrinsic=intrinsic, dist=dist, rvecs = rvecs, tvecs= tvecs)

        return camera_para


def calibrate_camera(board_size, marker_division, folder_path, target_size, ext, aruco_dict, camera_id, verify_flag = False):
    img_list = glob.glob(os.path.join(folder_path, ext))
    myarucor = charuco_marker(board_size, marker_division, aruco_dict = aruco_dict, camera_id=camera_id)
    save_folder_path = folder_path
    camera_para = myarucor.calibrateSingleCamerafromImg(img_list, save_folder_path = save_folder_path, size = target_size)
    if verify_flag:
        myarucor.reproject(camera_para, img_list, save_folder_path, showflag=True, size = target_size)
    myarucor.get_undist_para(camera_para, save_folder_path)

    return camera_para


if __name__ == '__main__':
    # please use png/jpg as calibration format, not using raw bmp

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', default=[7.5, 5.597])
    parser.add_argument('--marker_division', default=[25, 18])
    parser.add_argument('--image_size', default=[2048, 1500])
    parser.add_argument('--pattern_dir', default='./supp/pattern_img')
    parser.add_argument('--ext', default='*.png')
    parser.add_argument('--aruco_dict', default='DICT_5X5')
    parser.add_argument('--camera_name', default='FLIR_16mm')


    args = parser.parse_args()
    print(args)

    board_size = [7.5, 5.597]
    marker_division = [25, 18]
    image_size = [2048, 1500] # please fix to the image original size if you want to undist
    folder_path = args.pattern_dir
    para = calibrate_camera(args.board_size, args.marker_division, args.pattern_dir, args.image_size,
                            args.ext, args.aruco_dict, args.camera_name, verify_flag = True)
    print("camera K: ", para)









