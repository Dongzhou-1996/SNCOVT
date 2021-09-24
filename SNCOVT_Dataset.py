import os
import numpy as np
import glob
import cv2
import imageio
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import sys


parser = argparse.ArgumentParser('SNCOVT Dataset')
parser.add_argument('--dataset_dir', type=str, default='D:\\Dataset\\SNCOVT')
args = parser.parse_args()

'''
SNCOVT dataset format:
-train
|--sequence id {:04d}
|       |--img
|       |   |--left
|       |   |   |--{:06}.jpg
|       |   |   ...
|       |   |--right
|       |   |   |--{:06}.jpg
|       |   |   ...
|       |--calibration.npz
|       |--annotations.csv
-test   
|--sequence id {:04d}
|       |--img
|       |   |--left
|       |   |   |--{:06}.jpg
|       |   |   ...
|       |   |--right
|       |   |   |--{:06}.jpg
|       |   |   ...
|       |--calibration.npz
'''


class SNCOVT(object):
    def __init__(self, root_dir='', subset='train'):
        self.dataset_name = 'SNCOVT'
        self.subset = subset
        self.meta = {'resolution': '(1920, 1080)'}

        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            print('=> The root directory is not existed, it will be created soon...')
            os.makedirs(root_dir)
        else:
            print('=> Root directory: {}'.format(root_dir))

        self.subset_dir = os.path.join(self.root_dir, subset)
        if not os.path.exists(self.subset_dir):
            print('=> The {} dataset directory is not existed, it will be created soon...'.format(subset))
            os.makedirs(self.subset_dir)
        else:
            print('=> {} dataset directory: {}'.format(subset, self.subset_dir))

        self.sequence_dirs = sorted([os.path.join(self.subset_dir, seq_name) for seq_name in os.listdir(self.subset_dir)])
        print('=> {} sequences have been found in {} set, {}'.format(len(self.sequence_dirs), subset, self.dataset_name))

        pass

    def dataset_display(self, subset='train', restore=False):
        if subset in ['test', 'train']:
            print('=> start to display {} set in SNCOVT dataset!'.format(subset))
        else:
            print('=> Error! SNCOVT dataset don\'t contain {} set'.format(subset))
            exit(1)

        subset_dir = os.path.join(self.root_dir, subset)

        if not os.path.exists(subset_dir):
            print('=> Error! the {} set directory ({}) is not existed, stop displaying ...'.format(subset, subset_dir))
            exit(1)
        else:
            print('{} subset directory: {}'.format(subset, subset_dir))

        subset_sequence_dirs = [os.path.join(subset_dir, seq_name) for seq_name in os.listdir(subset_dir)]
        print('=> {} sequences have been found in {} set, {}'.format(len(subset_sequence_dirs), subset, self.dataset_name))

        for s, seq_dir in enumerate(subset_sequence_dirs):
            print('=> start to display sequence {:04d}, in {} subset'.format(s + 1, subset))

            check_file = os.path.join(seq_dir, 'check_file.txt')
            frames = np.loadtxt(check_file, delimiter='\n').astype(np.int)
            seq_len = len(frames)
            print('=> This sequence contain {} frames'.format(seq_len))

            if subset == 'train':
                annotation_file = os.path.join(seq_dir, 'annotations.csv')
                df = pd.read_csv(annotation_file, index_col=False)

            video_output_path = os.path.join(seq_dir, '{:04d}.gif'.format(s+1))
            # width, height = eval(self.meta['resolution'])
            # out_video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'MJPG'),
            #                               30, (width * 2, height), True)

            video = []
            for i, frame in enumerate(frames):
                print('=> frame: {}/{}, sequence {:04d}'.format(i + 1, seq_len, s+1))
                left_image_path = os.path.join(seq_dir, 'img', 'left', '{:06d}.jpg'.format(frame))
                right_image_path = os.path.join(seq_dir, 'img', 'right', '{:06d}.jpg'.format(frame))
                if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
                    print('=> failed to load binocular image, please check the integration of dataset!')
                    exit(1)

                # load bino-image
                left_img = imageio.imread(left_image_path)
                right_img = imageio.imread(right_image_path)
                print('=> successfully load binocular image')

                # load annotations
                if subset == 'train':
                    left_roi = eval(df.loc[i, 'left roi'])
                    right_roi = eval(df.loc[i, 'right roi'])
                    truncation = int(df.loc[i, 'truncation'])
                    if truncation <= 2:
                        print('left roi: {}'.format(left_roi))
                        print('right roi: {}'.format(right_roi))
                        cv2.rectangle(left_img, pt1=(left_roi[0], left_roi[1]),
                                      pt2=(left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]), color=(255, 255, 0),
                                      thickness=4)
                        cv2.rectangle(right_img, pt1=(right_roi[0], right_roi[1]),
                                      pt2=(right_roi[0] + right_roi[2], right_roi[1] + right_roi[3]), color=(255, 255, 0),
                                      thickness=4)
                    else:
                        print('=> no 2D annotations in this frame!')

                display_img = np.hstack((left_img, right_img))
                display_img = cv2.resize(display_img, (640, 256))
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                # if restore:
                #     out_video.write(display_img)
                video.append(display_img)
                cv2.namedWindow('image display', cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow('image display', display_img)

                key = cv2.waitKey(1)

                # if key == ord('n'):
                #     print('=> key interrupted! skip to display next sequence')
                #     out_video.release()
                #     break
                # elif key == 27:
                #     print('=> key interrupted! stop displaying ...')
                #     out_video.release()
                #     exit(1)
            if restore:
                print('=> saving video file in {} ...'.format(video_output_path))
                imageio.mimsave(video_output_path, video, 'GIF', duration=0.04)
            # out_video.release()

    def integrity_check(self, thread_nums=9):
        integrity_flag = True
        print('=> start integrity checking ...')
        num_batches = self.__len__() // thread_nums + 1
        for i in range(num_batches):
            executor = ThreadPoolExecutor(max_workers=thread_nums)
            tasks = []
            for thread in range(thread_nums):
                sequence = self.__getitem__((i * thread_nums + thread) % self.__len__())
                print('=> check sequence {} ...'.format(sequence.sequence_dir))
                task = executor.submit(sequence.integrity_check)
                tasks.append(task)

            print('=> checked sequence list: \n{}'.format(self.sequence_dirs[i * thread_nums:(i + 1) * thread_nums]))
            for future in as_completed(tasks):
                ret = future.result()
                if not ret:
                    integrity_flag = False
                    print('=> dataset integrity error!')
                    return integrity_flag
            executor.shutdown()

        if integrity_flag:
            print('=> the {} set of SNCOVT dataset is integral!'.format(self.subset))

        return integrity_flag

    def __getitem__(self, index):
        """
        Args:
            index (integer): Index of a trajectory.

        Returns:
            single_sequence: SingleSeq class, contains left and right images and corresponding annotations
        """
        single_sequence = SingleSeq(self.sequence_dirs[index], subset=self.subset)

        return single_sequence

    def __len__(self):
        return len(self.sequence_dirs)


class SingleSeq(object):

    def __init__(self, sequence_dir, subset='train'):
        self.sequence_dir = sequence_dir
        if sys.platform == 'win32':
            self.name = sequence_dir.split('\\')[-1]
        elif sys.platform == 'win64':
            self.name = sequence_dir.split('\\')[-1]
        elif sys.platform == 'linux':
            self.name = sequence_dir.split('/')[-1]
        else:
            raise ValueError('Unsupported platform!')
        self.subset = subset
        self.init_pos = 0

        # load binocular images
        self.left_img_dir = os.path.join(self.sequence_dir, 'img/left')
        self.left_img_files = sorted(glob.glob(os.path.join(self.left_img_dir, '*.jpg')))
        self.right_img_dir = os.path.join(self.sequence_dir, 'img/right')
        self.right_img_files = sorted(glob.glob(os.path.join(self.right_img_dir, '*.jpg')))

        if subset == 'train':
            # load 2d annotations
            self.annotation2d_file = os.path.join(sequence_dir, 'annotations.csv')
            self.data_frame = pd.read_csv(self.annotation2d_file)
            self.left_annotation = np.array(self.data_frame['left roi'])
            self.right_annotation = np.array(self.data_frame['right roi'])
            self.truncation = np.array(self.data_frame['truncation'])
        else:
            self.annotation2d_file = os.path.join(sequence_dir, 'annotations.csv')
            if os.path.exists(self.annotation2d_file):
                os.remove(self.annotation2d_file)
                print('=> deleted annotation file in sequence {}, subset {}'.format(self.name, self.subset))
            self.left_annotation = None
            self.right_annotation = None
            self.truncation = None
        pass

    def __getitem__(self, index):
        left_img_file = self.left_img_files[index]
        right_img_file = self.right_img_files[index]

        if self.subset == 'train':
            truncation = int(self.truncation[index])
            left_roi = eval(self.left_annotation[index])
            right_roi = eval(self.right_annotation[index])
        else:
            truncation = None
            left_roi = None
            right_roi = None

        return left_img_file, left_roi, right_img_file, right_roi, truncation

    def __len__(self):
        left_img_length = len(self.left_img_files)
        right_img_length = len(self.right_img_files)
        if left_img_length == right_img_length:
            return left_img_length
        else:
            print("left image length: %d" % left_img_length)
            print("right image length: %d" % right_img_length)
            print("the left image length is not equal to the right")
            return 0

    def inverse(self):
        self.left_img_files = self.left_img_files[::-1]
        self.right_img_files = self.right_img_files[::-1]
        self.left_annotation = self.left_annotation[::-1]
        self.right_annotation = self.right_annotation[::-1]
        self.truncation = self.truncation[::-1]

    def rand_pos(self):
        seq_len = self.__len__()
        init_pos = np.random.randint(0, seq_len)
        self.init_pos = init_pos
        if init_pos > seq_len/2:
            self.left_img_files = self.left_img_files[:init_pos]
            self.right_img_files = self.right_img_files[:init_pos]
            self.left_annotation = self.left_annotation[:init_pos]
            self.right_annotation = self.right_annotation[:init_pos]
            self.truncation = self.truncation[:init_pos]
        else:
            self.left_img_files = self.left_img_files[init_pos:]
            self.right_img_files = self.right_img_files[init_pos:]
            self.left_annotation = self.left_annotation[init_pos:]
            self.right_annotation = self.right_annotation[init_pos:]
            self.truncation = self.truncation[init_pos:]

    def integrity_check(self):
        check_file = os.path.join(self.sequence_dir, 'check_file.txt')
        frames = np.loadtxt(check_file, delimiter='\n').astype(np.int)
        integrity_flag = True
        for frame in frames:
            # print('=> frame: {}, sequence: {}'.format(frame, self.name))
            left_img_file = os.path.join(self.left_img_dir, '{:06d}.jpg'.format(frame))
            if not os.path.exists(left_img_file):
                print('=> Err: {} is not existed!'.format(left_img_file))
                integrity_flag = False
            left_img = cv2.imread(left_img_file, 1)
            if left_img is None:
                print('=> Err: {} is empty!'.format(left_img_file))
                integrity_flag = False

            right_img_file = os.path.join(self.right_img_dir, '{:06d}.jpg'.format(frame))
            if not os.path.exists(right_img_file):
                print('=> Err: {} is not existed!'.format(right_img_file))
                integrity_flag = False
            right_img = cv2.imread(right_img_file, 1)
            if right_img is None:
                print('=> Err: {} is empty!'.format(right_img_file))
                integrity_flag = False

        return integrity_flag


if __name__ == '__main__':
    sncovt = SNCOVT(args.dataset_dir, subset='train')
    sncovt.dataset_display(subset='train', restore=True)
    # sncovt.integrity_check(4)

    # SNCOVT dataset iterative exaples
    # repetition = 1
    # scales = 5
    # scale_ratio = 0.1
    # for s, sequence in enumerate(sncovt):
    #     for r in range(repetition):
    #         print('=> repetition: ', r)
    #         print('=> Sequence {}'.format(sequence.name))
    #         window_name = 'displaying'
    #
    #         for i, (left_img_path, left_roi, right_img_path, right_roi, truncation) in enumerate(sequence):
    #             print('=> Sequence {}, Frame: {}/{}'.format(sequence.name, i+1, sequence.__len__()))
    #             # load bino-image
    #             left_img = cv2.imread(left_img_path, 1)
    #             right_img = cv2.imread(right_img_path, 1)
    #             print('=> successfully load binocular image')
    #             if truncation <= 2:
    #                 # scale down annotations
    #                 left_roi = np.array(left_roi).astype(int)
    #                 right_roi = np.array(right_roi).astype(int)
    #
    #                 print('left roi: {}'.format(left_roi))
    #                 print('right roi: {}'.format(right_roi))
    #                 cv2.rectangle(left_img, pt1=(left_roi[0], left_roi[1]),
    #                               pt2=(left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]), color=(255, 255, 0),
    #                               thickness=2)
    #                 cv2.rectangle(right_img, pt1=(right_roi[0], right_roi[1]),
    #                               pt2=(right_roi[0] + right_roi[2], right_roi[1] + right_roi[3]), color=(255, 255, 0),
    #                               thickness=2)
    #             else:
    #                 print('=> no 2D annotations in this frame!')
    #                 continue
    #
    #             display_img = np.hstack((left_img, right_img))
    #             cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
    #             cv2.imshow(window_name, display_img)
    #             key = cv2.waitKey(1)
    #             if key == ord('n'):
    #                 print('=> key interrupted! skip to display next sequence')
    #                 # cv2.destroyWindow(window_name)
    #                 break
    #             elif key == 27:
    #                 print('=> key interrupted! stop displaying ...')
    #                 exit(1)
    #             else:
    #                 print('=> normal displaying ...')
    #         # cv2.destroyWindow(window_name)



