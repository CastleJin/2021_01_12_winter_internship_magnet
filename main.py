import os
import torch
import argparse
from train import mag

# set the gpu number
# os.environ["CUDA_VISIBLE_DEVICES"] = ''

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train',
                    help='train, play, play_temporal')
parser.add_argument('--checkpoint_path', dest='checkpoint', defualt=None,
                    help='Path of checkpoint file for load model')
parser.add_argument('--data_path', dest='data_path', default=None,
                    help='Path of dataset directory for train model')


# for inference
parser.add_argument('--vid_dir', dest='vid_dir', default=None,
                    help='Video folder to run the network on.')
parser.add_argument('--frame_ext', dest='frame_ext', default='png',
                    help='Video frame file extension.')
parser.add_argument('--out_dir', dest='out_dir', default=None,
                    help='Output folder of the video run.')
parser.add_argument('--amplification_factor', dest='amplification_factor',
                    type=float, default=5,
                    help='Magnification factor for inference.')
parser.add_argument('--velocity_mag', dest='velocity_mag', action='store_true',
                    help='Whether to do velocity magnification.')

# For temporal operation.
parser.add_argument('--fl', dest='fl', type=float, default=0.04
                    help='Low cutoff Frequency.')
parser.add_argument('--fh', dest='fh', type=float, default=0.4
                    help='High cutoff Frequency.')
parser.add_argument('--fs', dest='fs', type=float, default=30
                    help='Sampling rate.')
parser.add_argument('--n_filter_tap', dest='n_filter_tap', type=int, default=2
                    help='Number of filter tap required.')
parser.add_argument('--filter_type', dest='filter_type', type=str, default='differenceOfIIR'
                    help='Type of filter to use, must be Butter or differenceOfIIR.')

def main(args):
    model = mag(args.checkpoint)
    if args.phase == 'train':
        if not os.path.isdir(args.data_path):
            raise ValueError('There is no directory on the target path')
        model.train(args.data_path)
    elif args.phase == 'play':
        model.play(args.vid_dir,
                    args.frame_ext,
                    args.out_dir,
                    args.amplification_factor,
                    args.velocity_mag)
    elif args.phase == 'play_temporal':
        model.play_temporal(args.vid_dir,
                            args.frame_ext,
                            args.out_dir,
                            args.amplification_factor,
                            args.fl,
                            args.fh,
                            args.fs,
                            args.n_filter_tap,
                            args.filter_type)
    else:
        raise ValueError('Invalid phase argument. '
                            'Expected ["train", "play", "play_temporal"], '
                            'got ' + args.phase)

#model.play('/content/drive/MyDrive/video', 'png', '/content/drive/MyDrive/video/filter', 20, velocity_mag=True)
#model.play_temporal(vid_dir= '/content/drive/MyDrive/video', frame_ext='png', out_dir='/content/drive/MyDrive/video/filter')

if __name__ == '__main__':
    main(arguments)