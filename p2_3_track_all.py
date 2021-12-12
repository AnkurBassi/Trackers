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


for seq_name in LST_SEQS:
    tracker_name = 'DASIAMRPN'
    object_tracking = p2_1_opencv_tracking.Tracking(tracker_name, seq_name,
                                                    show_tracking=False, save_frames=True)
    object_tracking.track_target_object()
    object_tracking.save_fps()
    object_tracking.save_bbox()
