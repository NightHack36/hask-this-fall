import sys 
import math
import random
import requests as re
import numpy as np
from numpy import linalg as LA
import cv2
from scipy.spatial import distance
from openvino.inference_engine import IECore, IENetwork


def message(send):
    try:
        re.get('http://127.0.0.1:8000/mes/' + str(send))
        print(send)
    except:
        print('ошибка отправки')


def vzglag(mas):
    global gl
    global glaza
    ygl = int(sum(mas) / len(mas))
    if ygl > 0 and ygl < 91:
        message('eye:right down')

    elif ygl > 90 and ygl < 181:
        message('eye:left down')

    elif ygl < 0 and ygl > -91:
        message('eye:right up')

    else:
        message('eye:left up')


class ImageOpenVINOPreprocessing():
    def __init__(self):
        self.model_det = 'face-detection-adas-0001'
        self.model_hp = 'head-pose-estimation-adas-0001'
        self.model_gaze = 'gaze-estimation-adas-0002'
        self.model_lm = 'facial-landmarks-35-adas-0002'

        self.model_det = 'videoProccesing/intel/' + self.model_det + '/FP16/' + self.model_det
        self.model_hp = 'videoProccesing/intel/' + self.model_hp + '/FP16/' + self.model_hp
        self.model_gaze = 'videoProccesing/intel/' + self.model_gaze + '/FP16/' + self.model_gaze
        self.model_lm = 'videoProccesing/intel/' + self.model_lm + '/FP16/' + self.model_lm

        self._N = 0
        self._C = 1
        self._H = 2
        self._W = 3

        self.boundary_box_flag = True

        # Prep for face detection
        self.ie = IECore()

        self.net_det = IENetwork(model=self.model_det + '.xml', weights=self.model_det + '.bin')
        self.input_name_det = next(iter(self.net_det.inputs))  # Input blob name "data"
        self.input_shape_det = self.net_det.inputs[self.input_name_det].shape  # [1,3,384,672]
        self.out_name_det = next(iter(self.net_det.outputs))  # Output blob name "detection_out"
        self.exec_net_det = self.ie.load_network(network=self.net_det, device_name='CPU', num_requests=1)
        del self.net_det

        # Preparation for landmark detection
        self.net_lm = IENetwork(model=self.model_lm + '.xml', weights=self.model_lm + '.bin')
        self.input_name_lm = next(iter(self.net_lm.inputs))  # Input blob name
        self.input_shape_lm = self.net_lm.inputs[self.input_name_lm].shape  # [1,3,60,60]
        self.out_name_lm = next(iter(self.net_lm.outputs))  # Output blob name "embd/dim_red/conv"
        self.out_shape_lm = self.net_lm.outputs[self.out_name_lm].shape  # 3x [1,1]
        self.exec_net_lm = self.ie.load_network(network=self.net_lm, device_name='CPU', num_requests=1)
        del self.net_lm

        # Preparation for headpose detection
        self.net_hp = IENetwork(model=self.model_hp + '.xml', weights=self.model_hp + '.bin')
        self.input_name_hp = next(iter(self.net_hp.inputs))  # Input blob name
        self.input_shape_hp = self.net_hp.inputs[self.input_name_hp].shape  # [1,3,60,60]
        self.out_name_hp = next(iter(self.net_hp.outputs))  # Output blob name
        self.out_shape_hp = self.net_hp.outputs[self.out_name_hp].shape  # [1,70]
        self.exec_net_hp = self.ie.load_network(network=self.net_hp, device_name='CPU', num_requests=1)
        del self.net_hp

        # Preparation for gaze estimation
        self.net_gaze = IENetwork(model=self.model_gaze + '.xml', weights=self.model_gaze + '.bin')
        self.input_shape_gaze = [1, 3, 60, 60]
        self.exec_net_gaze = self.ie.load_network(network=self.net_gaze, device_name='CPU')
        del self.net_gaze

        self.spark_flag = False
        self.rez_eyes = []
        self.frame_num = 0


        self.rad_eyes_array = []
        self.max_eyes = 125


    def line(self, p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    def intersection(self, L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        x = Dx / D
        y = Dy / D
        return x, y

    def intersection_check(self, p1, p2, p3, p4):
        tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
        tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
        td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
        td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
        return tc1 * tc2 < 0 and td1 * td2 < 0

    def draw_gaze_line(self, img, coord1, coord2):
        cv2.line(img, coord1, coord2, (0, 0, 255), 2)

    def draw_spark(self, img, coord):
        if True:
            angle = random.random() * 2 * math.pi
            dia = random.randrange(10, 60)
            x = coord[0] + int(math.cos(angle) * dia - math.sin(angle) * dia)
            y = coord[1] + int(math.sin(angle) * dia + math.cos(angle) * dia)
            # cv2.line(img, coord, (x, y), (0, 255, 255), 2)

            cv2.circle(img, (x, y), 10, (255, 255, 0), 2)

    def normir(self, x1, y1, x2, y2):
        x2 = x2 - x1
        y2 = y2 - y1
        x1 = 0
        y1 = 0
        ygl = math.atan2(y2, x2) * 180 / math.pi
        return ygl

    def get_rad(self, x, y):
        return math.sqrt((x * x) + (y * y))

    def main(self, img):
        self.frame_num += 1
        out_img = img.copy()  # out_img will be drawn and modified to make an display image
        print(img.shape)
        img1 = cv2.resize(img, (self.input_shape_det[self._W], self.input_shape_det[self._H]))
        img1 = img1.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        img1 = img1.reshape(self.input_shape_det)
        res_det = self.exec_net_det.infer(inputs={self.input_name_det: img1})  # Detect faces

        gaze_lines = []
        for obj in res_det[self.out_name_det][0][0]:  # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
            if obj[2] > 0.75:  # Confidence > 75%
                xmin = abs(int(obj[3] * img.shape[1]))
                if xmin > 10:
                    xmin -= 10

                ymin = abs(int(obj[4] * img.shape[0]))
                if ymin > 10:
                    ymin -= 10

                xmax = abs(int(obj[5] * img.shape[1]))
                if xmax < img.shape[1] - 10:
                    xmax += 10

                ymax = abs(int(obj[6] * img.shape[0]))
                if ymax < img.shape[0] - 10:
                    ymax += 10

                class_id = int(obj[1])
                face = img[ymin:ymax, xmin:xmax]  # Crop the face image
                if self.boundary_box_flag == True:
                    cv2.rectangle(out_img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

                # Find facial landmarks (to find eyes)
                face1 = cv2.resize(face, (self.input_shape_lm[self._W], self.input_shape_lm[self._H]))
                face1 = face1.transpose((2, 0, 1))
                face1 = face1.reshape(self.input_shape_lm)
                res_lm = self.exec_net_lm.infer(inputs={self.input_name_lm: face1})  # Run landmark detection
                lm = res_lm[self.out_name_lm][0][:8].reshape(4,
                                                             2)  # [[left0x, left0y], [left1x, left1y], [right0x, right0y], [right1x, right1y] ]

                # Estimate head orientation (yaw=Y, pitch=X, role=Z)
                res_hp = self.exec_net_hp.infer(inputs={self.input_name_hp: face1})  # Run head pose estimation
                yaw = res_hp['angle_y_fc'][0][0]
                pitch = res_hp['angle_p_fc'][0][0]
                roll = res_hp['angle_r_fc'][0][0]

                _X = 0
                _Y = 1
                # Landmark position memo...   lm[1] (eye) lm[0] (nose)  lm[2] (eye) lm[3]
                eye_sizes = [abs(int((lm[0][_X] - lm[1][_X]) * face.shape[1])),
                             abs(int((lm[3][_X] - lm[2][_X]) * face.shape[1]))]  # eye size in the cropped face image
                eye_centers = [[int(((lm[0][_X] + lm[1][_X]) / 2 * face.shape[1])),
                                int(((lm[0][_Y] + lm[1][_Y]) / 2 * face.shape[0]))],
                               [int(((lm[3][_X] + lm[2][_X]) / 2 * face.shape[1])), int(((lm[3][_Y] + lm[2][_Y]) / 2 *
                                                                                         face.shape[
                                                                                             0]))]]  # eye center coordinate in the cropped face image
                if eye_sizes[0] < 4 or eye_sizes[1] < 4:
                    continue

                ratio = 0.7
                eyes = []
                for i in range(2):
                    # Crop eye images
                    x1 = int(eye_centers[i][_X] - eye_sizes[i] * ratio)
                    if x1 > 5:
                        x1 -= 3

                    x2 = int(eye_centers[i][_X] + eye_sizes[i] * ratio)
                    if x2 < img.shape[0] - 5:
                        x2 += 3

                    y1 = int(eye_centers[i][_Y] - eye_sizes[i] * ratio)
                    if y1 > 5:
                        y1 -= 3

                    y2 = int(eye_centers[i][_Y] + eye_sizes[i] * ratio)
                    if y2 < img.shape[0] - 5:
                        y2 += 3

                    eyes.append(cv2.resize(face[y1:y2, x1:x2].copy(), (
                    self.input_shape_gaze[self._W], self.input_shape_gaze[self._H])))  # crop and resize

                    # Draw eye boundary boxes
                    if self.boundary_box_flag == True:
                        cv2.rectangle(out_img, (x1 + xmin, y1 + ymin), (x2 + xmin, y2 + ymin), (0, 255, 0), 2)

                    # rotate eyes around Z axis to keep them level
                    if roll != 0.:
                        rotMat = cv2.getRotationMatrix2D(
                            (int(self.input_shape_gaze[self._W] / 2), int(self.input_shape_gaze[self._H] / 2)), roll,
                            1.0)
                        eyes[i] = cv2.warpAffine(eyes[i], rotMat,
                                                 (self.input_shape_gaze[self._W], self.input_shape_gaze[self._H]),
                                                 flags=cv2.INTER_LINEAR)
                    eyes[i] = eyes[i].transpose((2, 0, 1))  # Change data layout from HWC to CHW
                    eyes[i] = eyes[i].reshape((1, 3, 60, 60))

                hp_angle = [yaw, pitch, 0]  # head pose angle in degree
                res_gaze = self.exec_net_gaze.infer(inputs={'left_eye_image': eyes[0],
                                                            'right_eye_image': eyes[1],
                                                            'head_pose_angles': hp_angle})  # gaze estimation
                gaze_vec = res_gaze['gaze_vector'][
                    0]  # result is in orthogonal coordinate system (x,y,z. not yaw,pitch,roll)and not normalized
                gaze_vec_norm = gaze_vec / np.linalg.norm(gaze_vec)  # normalize the gaze vector

                vcos = math.cos(math.radians(roll))
                vsin = math.sin(math.radians(roll))
                tmpx = gaze_vec_norm[0] * vcos + gaze_vec_norm[1] * vsin
                tmpy = -gaze_vec_norm[0] * vsin + gaze_vec_norm[1] * vcos
                gaze_vec_norm = [tmpx, tmpy]

                # Store gaze line coordinations
                for i in range(2):
                    coord1 = (eye_centers[i][_X] + xmin, eye_centers[i][_Y] + ymin)
                    coord2 = (eye_centers[i][_X] + xmin + int((gaze_vec_norm[0] + 0.) * 300),
                              eye_centers[i][_Y] + ymin - int((gaze_vec_norm[1] + 0.) * 300))
                    gaze_lines.append([coord1, coord2, False])  # line(coord1, coord2); False=spark flag

        # Drawing gaze lines

        eyes = []
        isEyes = False
        if len(gaze_lines)>0:
         rad_eyes = self.get_rad(gaze_lines[0][0][0] - gaze_lines[0][1][0], gaze_lines[0][0][1] - gaze_lines[0][1][1])

         isEyes = False

         if len(self.rad_eyes_array) <= 10:
            self.rad_eyes_array.append(rad_eyes)
            isEyes = False
         else:
            self.rad_eyes_array.remove(self.rad_eyes_array[0])
            self.rad_eyes_array.append(rad_eyes)
            eyes_c = 0
            for rad in self.rad_eyes_array:
                if rad > self.max_eyes:
                    eyes_c += 1
            if eyes_c > 10:
                isEyes = True
            else:
                isEyes = False

         for gaze_line in gaze_lines:
            eyes.append(self.normir(gaze_line[0][0], gaze_line[0][1], gaze_line[1][0], gaze_line[1][1]))
            out_img = cv2.circle(out_img, (gaze_line[0][0], gaze_line[0][1]), 135, (255, 0, 0), 2)

            self.draw_gaze_line(out_img, (gaze_line[0][0], gaze_line[0][1]), (gaze_line[1][0], gaze_line[1][1]))


         try:
            self.rez_eyes.append(sum(eyes) / len(eyes))

            if len(self.rez_eyes) > 10:
                self.rez_eyes.remove(self.rez_eyes[0])
            if self.frame_num % 40 == 0:
                vzglag(self.rez_eyes)
         except:
            print('nan')

        return out_img, isEyes


"""
cam = cv2.VideoCapture(-1)
camx, camy = [(1920, 1080), (1280, 720), (800, 600), (480, 480)][1]     # Set camera resolution [1]=1280,720
cam.set(cv2.CAP_PROP_FRAME_WIDTH , camx)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)

imageclass = ImageOpenVINOPreprocessing()
imageclass.usage()

while True:
    ret, img = cam.read()
    if ret == False:
        break

    cv2.imshow("same test", imageclass.main(img))



    key = cv2.waitKey(1)
    if key == 27: break
    if key == ord(u'l'): laser_flag = True if laser_flag == False else False  # toggles laser_flag
    if key == ord(u'f'): flip_flag = True if flip_flag == False else False  # image flip flag
    if key == ord(u'b'): boundary_box_flag = True if boundary_box_flag == False else False  # boundary box flag
    if key == ord(u's'): spark_flag = True if spark_flag == False else False  # spark flag

cv2.destroyAllWindows()"""
