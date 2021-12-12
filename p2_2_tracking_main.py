import argparse
import os
import p2_1_opencv_tracking  # need opencv version 4.5.4-dev

# list of available trackers
# note that 'BOOSTING', 'MEDIANFLOW', 'MIL', 'MOSSE', 'TLD' are legacy trackers
# 'GOTURN', 'DASIAMRPN' are DL-based, which need to be provided with parameters
LST_TRACKERS = ['CSRT', 'KCF',
                'BOOSTING', 'MEDIANFLOW', 'MIL', 'MOSSE', 'TLD',
                'GOTURN', 'DASIAMRPN']

# list of available sequences in the folder "sequences"
LST_SEQS = os.listdir('./sequences')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tracker_name", nargs='?', default="DASIAMRPN",
                        help="Tracker name, choose a OpenCV tracker, default is DASIAMRPN", type=str)

    parser.add_argument("seq_name", nargs='?', default="Fish",
                        help="Sequence name, choose a sequence in the sequences folder, default is Fish", type=str)

    parser.add_argument('--show_tracking',
                        dest='show_tracking', action='store_true', help="Default is to show the frames while tracking")
    parser.add_argument('--dont_show_tracking',
                        dest='show_tracking', action='store_false', help="Default is to show the frames while tracking")
    parser.set_defaults(show_tracking=True)

    parser.add_argument('--save_frames',
                        dest='save_frames', action='store_true', help="Default is to save the resulting frames")
    parser.add_argument('--dont_save_frames',
                        dest='save_frames', action='store_false', help="Default is to save the resulting frames")
    parser.set_defaults(save_frames=True)

    args = parser.parse_args()

    object_tracking = p2_1_opencv_tracking.Tracking(args.tracker_name, args.seq_name,
                                                    show_tracking=args.show_tracking, save_frames=args.save_frames)
    object_tracking.track_target_object()
    object_tracking.save_fps()
    object_tracking.save_bbox()
    print("Tracking was completed, please check the results folder.")
