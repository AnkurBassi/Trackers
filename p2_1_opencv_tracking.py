import cv2  # opencv version 4.5.4-dev
import os
import sys
import numpy as np


class Tracking:
    # list of available trackers
    # note that 'BOOSTING', 'MEDIANFLOW', 'MIL', 'MOSSE', 'TLD' are legacy trackers
    # 'GOTURN', 'DASIAMRPN' are DL-based, which need to be provided with parameters
    LST_TRACKERS = ['CSRT', 'KCF',
                    'BOOSTING', 'MEDIANFLOW', 'MIL', 'MOSSE', 'TLD',
                    'GOTURN', 'DASIAMRPN']
    # list of available sequences in the folder "sequences"
    LST_SEQS = os.listdir('./sequences')

    def __init__(self, tracker_name, seq_name, show_tracking=True, save_frames=True):
        # default is to show the frames while tracking and save the resulting frames
        self.bool_show_tracking = show_tracking
        self.bool_save_frames = save_frames
        self.str_tracker_name = tracker_name
        self.str_seq_name = seq_name
        self.is_valid_tracker()
        self.is_valid_sequence()
        # create a cv2 video based on the given sequence
        self.dir_seq_folder = './sequences/' + self.str_seq_name + '/'
        self.cv_seq = cv2.VideoCapture(self.dir_seq_folder+'img/%04d.jpg')
        # create a list that contains the first frame bounding box of all target objects
        self.lst_bbox_ff = []
        self.update_lst_bbox_ff()
        # initialize the first frame of the sequence
        self.cv_first_frame = None
        self.update_first_frame()
        # create a multi tracker list for multi object tracking
        self.lst_multi_object = []
        self.update_lst_multi_object()
        # make sure there exists a folder to store the output (frames/video + fps + bounding box(es))
        self.dir_save_frames = "./results/" + self.str_seq_name + \
            "/" + self.str_tracker_name + "/output_frames/"
        self.check_dir_save_frames_exists()
        # create a list that stores fps
        self.lst_fps = []
        # create a list that stores the bounding boxes of the target object(s)
        self.lst_bbox_all_obj = [[] for _ in range(len(self.lst_bbox_ff))]

    # check tracker is supported
    def is_valid_tracker(self):
        if self.str_tracker_name not in self.LST_TRACKERS:
            print("The tracker ({}) is not supported. Please try another one.".format(
                self.str_tracker_name))
            print("Available trackers: {}".format(
                ', '.join(i for i in self.LST_TRACKERS)))
            sys.exit()

    # check sequence is available
    def is_valid_sequence(self):
        self.LST_SEQS = os.listdir('./sequences')
        if self.str_seq_name not in self.LST_SEQS:
            print("The sequence ({}, case-sensitive) was not found in the sequences folder. ".format(
                self.str_seq_name))
            sys.exit()

    # update first frame bounding box list
    def update_lst_bbox_ff(self):
        for object_idx, file_name in enumerate(os.listdir(self.dir_seq_folder)):
            if file_name.startswith('groundtruth_rect'):
                bbox_ff_str = open(os.path.join(
                    self.dir_seq_folder, file_name), "r").readline().strip().replace('\t', ',')
                bbox_ff = list(map(int, bbox_ff_str.split(',')))
                self.lst_bbox_ff.append(bbox_ff)
                # print("Obtained the initial bounding box (first frame) of Object {}:".format(object_idx+1))
                # print("{}: {}".format(os.path.join(self.dir_seq_folder, file_name), bbox_ff))

    # define a function that returns a cv.tracker class based on the given tracker name
    def return_tracker(self, str_tracker_name):
        if str_tracker_name == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif str_tracker_name == 'KCF':
            return cv2.TrackerKCF_create()
        elif str_tracker_name == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        elif str_tracker_name == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        elif str_tracker_name == 'MIL':
            return cv2.legacy.TrackerMIL_create()
        elif str_tracker_name == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif str_tracker_name == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        elif str_tracker_name == 'GOTURN':
            params = cv2.TrackerGOTURN_Params()
            params.modelTxt = "./pretrained_models/GOTURN/goturn.prototxt"
            params.modelBin = "./pretrained_models/GOTURN/goturn.caffemodel"
            return cv2.TrackerGOTURN_create(params)
        elif str_tracker_name == 'DASIAMRPN':
            params = cv2.TrackerDaSiamRPN_Params()
            params.model = "./pretrained_models/DaSiamRPN/dasiamrpn_model.onnx"
            params.kernel_r1 = "./pretrained_models/DaSiamRPN/dasiamrpn_kernel_r1.onnx"
            params.kernel_cls1 = "./pretrained_models/DaSiamRPN/dasiamrpn_kernel_cls1.onnx"
            return cv2.TrackerDaSiamRPN_create(params)

    # update/initialize the first frame cv2 object
    def update_first_frame(self):
        _, self.cv_first_frame = self.cv_seq.read()

    def update_lst_multi_object(self):
        for i in range(len(self.lst_bbox_ff)):
            tracker_temp = self.return_tracker(self.str_tracker_name)
            tracker_temp.init(self.cv_first_frame, self.lst_bbox_ff[i])
            self.lst_multi_object.append(tracker_temp)

    def check_dir_save_frames_exists(self):
        if not os.path.exists(self.dir_save_frames):
            os.makedirs(self.dir_save_frames)

    def track_target_object(self):
        print("Now tracking {} using {} ...".format(
            self.str_seq_name, self.str_tracker_name))
        self.img_curr_num = 1
        # run a loop to track the target object(s) in each frame
        while True:
            # read a new frame from the sequence
            bool_valid_frame, frame = self.cv_seq.read()
            if not bool_valid_frame:
                break

            lst_valid_tracking = []
            lst_bbox_tracking = []

            # update multi object tracker
            start_timer = cv2.getTickCount()  # (for fps calculation)
            for curr_object in self.lst_multi_object:
                bool_valid_tracking, bbox_curr_obj = curr_object.update(frame)
                lst_valid_tracking.append(bool_valid_tracking)
                lst_bbox_tracking.append(bbox_curr_obj)
            end_timer = cv2.getTickCount()  # (for fps calculation)

            # add new bounding box(es) to the frame if tracking was successful
            for bbox_idx, new_bbox in enumerate(lst_bbox_tracking):
                if lst_valid_tracking[bbox_idx]:
                    bbox_x = int(new_bbox[0])
                    bbox_y = int(new_bbox[1])
                    bbox_width = int(new_bbox[2])
                    bbox_height = int(new_bbox[3])
                    cv2.rectangle(frame, (bbox_x, bbox_y),
                                  (bbox_x+bbox_width, bbox_y+bbox_height), (0, 255, 0), 1)
                    cv2.putText(frame, "Object"+str(bbox_idx+1), (bbox_x, bbox_y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, "Object{}: success".format(bbox_idx+1), (0, 60+10*bbox_idx),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
                    self.lst_bbox_all_obj[bbox_idx].append(
                        [bbox_x, bbox_y, bbox_width, bbox_height])
                else:
                    cv2.putText(frame, "Object{}: failure".format(bbox_idx+1), (0, 60+10*bbox_idx),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
                    self.lst_bbox_all_obj[bbox_idx].append('_failed_')

            # calculate frames per second (fps)
            temp_fps = int(cv2.getTickFrequency() / (end_timer - start_timer))
            self.lst_fps.append(temp_fps)

            # add tracker name to the frame
            cv2.putText(frame, "Tracker: "+self.str_tracker_name, (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
            # add fps to the frame
            cv2.putText(frame, "FPS: "+str(temp_fps), (0, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

            if self.bool_save_frames:
                # save frame
                cv2.imwrite(os.path.join(self.dir_save_frames,
                            str(self.img_curr_num+1).zfill(4)+'.jpg'), frame)

            if self.bool_show_tracking:
                # show frame (bounding box + fracker name + fps)
                cv2.imshow("Visual Object Tracking", frame)
                # press q to stop tracking
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            self.img_curr_num += 1

    def save_fps(self):
        # save the fps
        fps_file = open(os.path.join(
            "./results/", self.str_seq_name, self.str_tracker_name, "fps.txt"), "w")
        for temp_fps in self.lst_fps:
            fps_file.write(str(temp_fps) + "\n")
        fps_file.close()

    def save_bbox(self):
        # save the resulting bounding box of each target object
        for obj_idx in range(len(self.lst_bbox_ff)):
            bbox_file = open(os.path.join("./results/", self.str_seq_name,
                             self.str_tracker_name, "bbox_tracked_object"+str(obj_idx+1)+".txt"), "w")
            for temp_bbox in self.lst_bbox_all_obj[obj_idx]:
                bbox_file.write(str(temp_bbox)[1:-1] + "\n")
            bbox_file.close()
