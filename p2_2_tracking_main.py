import os
import p2_1_opencv_tracking

# list of available trackers
# note that 'BOOSTING', 'MEDIANFLOW', 'MIL', 'MOSSE', 'TLD' are legacy trackers
# 'GOTURN', 'DASIAMRPN' are DL-based, which need to be provided with parameters
LST_TRACKERS = ['CSRT', 'KCF',
                'BOOSTING', 'MEDIANFLOW', 'MIL', 'MOSSE', 'TLD',
                'GOTURN', 'DASIAMRPN']

# list of available sequences in the folder "sequences"
LST_SEQS = os.listdir('./sequences')


if __name__ == '__main__':
    tracker_name = 'DASIAMRPN'.upper()  # uppercase
    seq_name = 'Jogging'  # case-sensitive
    object_tracking = p2_1_opencv_tracking.Tracking(tracker_name, seq_name,
                                                    show_tracking=True, save_frames=True)
    object_tracking.track_target_object()
    object_tracking.save_fps()
    object_tracking.save_bbox()
    print("Tracking was completed, please check the results folder.")
