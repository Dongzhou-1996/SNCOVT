from __future__ import absolute_import, division, print_function
import sys
import os
import numpy as np
import glob
import ast
import json
import time
import matplotlib.pyplot as plt
import matplotlib
import cv2
import argparse

# add sources path
sys.path.append('./trackers')

from SNCOVT_Dataset import SNCOVT
from trackers.SiamFC.siamfc import TrackerSiamFC
from trackers.SiamRPN.siamrpn import TrackerSiamRPN
from trackers.KCF.kcftracker import KCFTracker
from trackers.cftracker.mosse import MOSSE
from trackers.cftracker.csk import CSK
from trackers.cftracker.kcf import KCF
from trackers.cftracker.cn import CN
from trackers.cftracker.dsst import DSST
from trackers.cftracker.staple import Staple
from trackers.cftracker.dat import DAT
from trackers.cftracker.eco import ECO
from trackers.cftracker.bacf import BACF
from trackers.cftracker.csrdcf import CSRDCF
from trackers.cftracker.samf import SAMF
from trackers.cftracker.ldes import LDES
from trackers.cftracker.mkcfup import MKCFup
from trackers.cftracker.strcf import STRCF
from trackers.cftracker.mccth_staple import MCCTHStaple
from trackers.lib.eco.config import otb_deep_config, otb_hc_config
from trackers.cftracker.config import staple_config, ldes_config, dsst_config, csrdcf_config, mkcf_up_config, \
    mccth_staple_config


class ExperimentMonoTracker(object):
    """Experiment pipeline and evaluation toolkit for 3D_SOT dataset.
        Args:
            root_dir (string): Root directory of 3D_SOT dataset where
                ``train``, ``val`` and ``test`` folders exist.
            subset (string): Specify ``train``, ``val`` or ``test``
                subset of 3D_SOT.
            result_dir (string, optional): Directory for storing tracking
                results. Default is ``./results``.
            report_dir (string, optional): Directory for storing performance
                evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, subset='val', result_dir='results',
                 report_dir='reports', use_dataset=True, repetition=3, min_scale_ratio=0.1):
        super(ExperimentMonoTracker, self).__init__()
        assert subset in ['test', 'train']
        self.subset = subset
        if use_dataset:
            self.dataset = SNCOVT(root_dir, subset=subset)
        self.result_dir = os.path.join(result_dir, self.dataset.dataset_name, subset)
        self.report_dir = os.path.join(report_dir, self.dataset.dataset_name, subset)
        self.nbins_iou = 201
        self.repetitions = repetition
        self.random_level = 8
        self.scales = 5
        self.scale_ratio = min_scale_ratio

    def normal_run(self, tracker, save_video=False, overwrite_result=False):
        print('Running tracker %s on %s...' % (tracker.name, self.dataset.dataset_name))
        # loop over the complete dataset
        for s, sequence in enumerate(self.dataset):
            sequence_name = sequence.name
            print('--Sequence %s (%dth in total %d)' % (sequence_name,
                                                        s + 1, len(self.dataset)))
            print('--Sequence length %d' % sequence.__len__())
            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                print(' Repetition: %d' % (r + 1))
                record_file = os.path.join(
                    self.result_dir, tracker.name, sequence_name,
                    '%s_%02d_bbox.txt' % (sequence_name, r + 1))

                if os.path.exists(record_file) and not overwrite_result:
                    print('Found results, skipping', sequence_name)
                else:
                    # tracking loop
                    print("=> monocular tracker {} starting ...".format(tracker.name))
                    tracking_results = []
                    overlaps = []
                    times = []
                    bound = ast.literal_eval(self.dataset.meta['resolution'])
                    restart = 1
                    for f, (left_img_file, left_roi, _, _, truncation) in enumerate(sequence):
                        print('--Frame %d/%d in sequence %s' % (f, sequence.__len__(), sequence_name))
                        left_img = cv2.imread(left_img_file, 1)
                        if left_img is None:
                            print("frame {} failed!".format(f))
                            print("=> skipping to track next sequence ...")
                            break
                        if truncation > 2:
                            print("=> the whole object is truncated in frame{}".format(f))
                            print("=> skipping to the next frame ...")
                            continue

                        if f == 0 or restart:
                            print("=> monocular tracker {} initializing ...".format(tracker.name))
                            print('left initial roi: {}'.format(left_roi))
                            start_time = time.time()
                            tracker.init(left_img, left_roi)
                            end_time = time.time()
                            elapsed_time = (end_time - start_time)
                            print('initial elapsed time: {} s'.format(elapsed_time))
                            times.append(elapsed_time)
                            tracking_results.append(left_roi)
                            overlaps.append(1.0)
                            print("=> The initialization of monocular tracker {} have been done!".format(tracker.name))
                            restart = 0
                        else:
                            print('left roi: {}'.format(left_roi))
                            start_time = time.time()
                            left_bbox2d = tracker.update(left_img)
                            end_time = time.time()
                            elapsed_time = (end_time - start_time)
                            times.append(elapsed_time)
                            print('tracking result: {}'.format(left_bbox2d))
                            # print('tracking elapsed time: {}s'.format(elapsed_time))
                            tracking_results.append(left_bbox2d)
                            left_iou = self.bbox2d_overlap_calc(left_roi, left_bbox2d, bound)
                            overlaps.append(left_iou)
                            print('overlap: {}'.format(left_iou))

                    # record results
                    tracking_results = np.array(tracking_results).reshape(-1, 4)
                    overlaps = np.array(overlaps)

                    # record bounding boxes
                    record_dir = os.path.dirname(record_file)
                    if not os.path.isdir(record_dir):
                        os.makedirs(record_dir)
                    np.savetxt(record_file, tracking_results, fmt='%.3f', delimiter=',')
                    print('Tracking results recorded at', record_file)

                    # record overlaps
                    overlap_file = record_file[:record_file.rfind('_')] + '_overlap.txt'
                    np.savetxt(overlap_file, overlaps, fmt='%.3f', delimiter=',')

                    # record running times
                    time_file = record_file[:record_file.rfind('_')] + '_time.txt'
                    np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

                # save videos
                if save_video:
                    video_file = record_file[:record_file.rfind('_')] + '_video.avi'
                    if os.path.exists(video_file) and not overwrite_result:
                        print("Found video results, skipping %s" % sequence_name)
                        break
                    width, height = eval(self.dataset.meta['resolution'])
                    out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height), True)
                    tracking_results = np.loadtxt(record_file, dtype=float, delimiter=',').astype(int)
                    f = 0
                    for left_img_file, left_roi_gt, _, _, truncation in sequence:
                        print('--Frame %dth in total %d' % (f, sequence.__len__()))
                        if truncation > 2:
                            # skip the frame of which object is truncated
                            continue
                        left_img = cv2.imread(left_img_file, 1)
                        # bbox2d ground-truth
                        left_pt1_gt = [left_roi_gt[0], left_roi_gt[1]]
                        left_pt2_gt = [left_roi_gt[0] + left_roi_gt[2], left_roi_gt[1] + left_roi_gt[3]]

                        # bbox2d result of mono tracker
                        left_roi = tracking_results[f, :4]
                        left_pt1 = [left_roi[0], left_roi[1]]
                        left_pt2 = [left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]]

                        # draw groud-truth bbox2d
                        cv2.rectangle(left_img, pt1=tuple(left_pt1_gt), pt2=tuple(left_pt2_gt),
                                      color=(255, 255, 0), thickness=2)

                        # draw bbox2d of binoSiamfc
                        cv2.rectangle(left_img, pt1=tuple(left_pt1), pt2=tuple(left_pt2),
                                      color=(0, 255, 0), thickness=2)
                        if sys.platform == 'win32' or sys.platform == 'win64':
                            cv2.namedWindow('visualization', cv2.WINDOW_GUI_EXPANDED)
                            cv2.imshow('visualization', left_img)
                            key = cv2.waitKey(1)
                            if key & 0xFF == 27:
                                print('=> tracking process stopped !')
                                exit(1)
                            elif key & 0xFF == ord('n'):
                                print("=> skipping to track next sequence ...")
                                break
                            out_video.write(left_img)
                            f = f + 1
                    out_video.release()
                    print('Videos saved at', video_file)
                    if sys.platform == 'win32' or sys.platform == 'win64':
                        cv2.destroyWindow('visualization')

    def random_run(self, tracker, save_video=False, overwrite_result=False, rand_level=1):
        global out_video
        print('=> Running tracker %s on %s...' % (tracker.name, self.dataset.dataset_name))
        # run multiple repetitions for dataset
        for r in range(self.repetitions):
            print('=> Repetition: %d' % (r + 1))
            # loop over the complete dataset
            for s, sequence in enumerate(self.dataset):
                sequence_name = sequence.name
                print('=> Sequence %s (%dth in total %d)' % (sequence_name, s + 1, len(self.dataset)))
                record_file = os.path.join(
                    self.result_dir, tracker.name, sequence_name,
                    '{}_{:02d}_rand{:d}_bbox.txt'.format(sequence_name, r + 1, rand_level))
                print('record file: {}'.format(record_file))
                record_dir = os.path.dirname(record_file)
                if not os.path.isdir(record_dir):
                    os.makedirs(record_dir)

                if os.path.exists(record_file) and not overwrite_result:
                    print('=> Found results, skipping', sequence_name)
                else:
                    # tracking loop
                    print("=> monocular tracker {} starting ...".format(tracker.name))
                    tracking_results = []
                    overlaps = []
                    times = []
                    bound = ast.literal_eval(self.dataset.meta['resolution'])

                    if save_video:
                        video_file = record_file[:record_file.rfind('_')] + '_video.avi'
                        width, height = eval(self.dataset.meta['resolution'])
                        out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height),
                                                    True)

                    # random initial position
                    if rand_level in [1, 4, 5, 7]:
                        sequence.rand_pos()
                        print('=> random initial frame: {}'.format(sequence.init_pos))
                    # random sequence order
                    if rand_level in [2, 4, 6, 7]:
                        # inverse displaying
                        if np.random.rand() < 0.5:
                            print('=> start inverted traversing ...')
                            sequence.inverse()
                        else:
                            print('=> start sequential traversing ...')
                    initial_flag = True
                    for f, (left_img_file, left_roi, _, _, truncation) in enumerate(sequence):
                        print('=> Frame %d/%d in sequence %s' % (f, sequence.__len__(), sequence_name))
                        left_img = cv2.imread(left_img_file, 1)
                        if left_img is None:
                            print("frame {} failed!".format(f))
                            print("=> skipping to track next sequence ...")
                            break
                        if truncation > 2:
                            print("=> the whole object is truncated in frame{}".format(f))
                            print("=> skipping to the next frame ...")
                            continue

                        if f == 0 or initial_flag:
                            print("=> monocular tracker {} initializing ...".format(tracker.name))
                            print('left initial roi: {}'.format(left_roi))

                            if rand_level in [3, 5, 6, 7]:
                                # random initial bounding box
                                rand_left_roi = [0] * 4
                                # w
                                rand_left_roi[2] = int(left_roi[2] * (0.8 + 0.4 * np.random.rand()))
                                # h
                                rand_left_roi[3] = int(left_roi[3] * (0.8 + 0.4 * np.random.rand()))
                                # x
                                rand_left_roi[0] = int(left_roi[0] + (-1 + 2 * np.random.rand()) * left_roi[2] / 5)
                                # y
                                rand_left_roi[1] = int(left_roi[1] + (-1 + 2 * np.random.rand()) * left_roi[3] / 5)

                            else:
                                # original initial bounding box
                                rand_left_roi = left_roi

                            start_time = time.time()
                            tracker.init(left_img, rand_left_roi)
                            end_time = time.time()
                            elapsed_time = (end_time - start_time)
                            print('=> initial elapsed time: {} s'.format(elapsed_time))
                            left_bbox2d = rand_left_roi
                            times.append(elapsed_time)
                            tracking_results.append(np.concatenate((left_roi, left_bbox2d)))
                            left_iou = self.bbox2d_overlap_calc(left_roi, left_bbox2d, bound)
                            overlaps.append(left_iou)
                            initial_flag = False
                            print("=> The initialization of monocular tracker {} have been done!".format(tracker.name))
                        else:
                            print('left roi: {}'.format(left_roi))
                            start_time = time.time()
                            left_bbox2d = tracker.update(left_img)
                            end_time = time.time()
                            elapsed_time = (end_time - start_time)
                            print('=> tracking elapsed time: {} s'.format(elapsed_time))
                            times.append(elapsed_time)
                            print('tracking result: {}'.format(left_bbox2d))
                            # print('tracking elapsed time: {}s'.format(elapsed_time))
                            tracking_results.append(np.concatenate((left_roi, left_bbox2d)))
                            left_iou = self.bbox2d_overlap_calc(left_roi, left_bbox2d, bound)
                            overlaps.append(left_iou)
                            print('overlap: {}'.format(left_iou))

                        if save_video:
                            # bbox2d ground-truth
                            left_pt1_gt = [left_roi[0], left_roi[1]]
                            left_pt2_gt = [left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]]

                            # bbox2d result of mono tracker
                            left_pt1 = [int(left_bbox2d[0]), int(left_bbox2d[1])]
                            left_pt2 = [int(left_bbox2d[0] + left_bbox2d[2]), int(left_bbox2d[1] + left_bbox2d[3])]

                            # draw groud-truth bbox2d
                            cv2.rectangle(left_img, pt1=tuple(left_pt1_gt), pt2=tuple(left_pt2_gt),
                                          color=(255, 255, 0), thickness=2)

                            # draw bbox2d of tracker
                            cv2.rectangle(left_img, pt1=tuple(left_pt1), pt2=tuple(left_pt2),
                                          color=(0, 255, 0), thickness=2)

                            if sys.platform == 'win32' or sys.platform == 'win64':
                                cv2.namedWindow('visualization', cv2.WINDOW_GUI_EXPANDED)
                                cv2.imshow('visualization', left_img)
                                key = cv2.waitKey(1)
                                if key & 0xFF == 27:
                                    print('=> tracking process stopped !')
                                    out_video.release()
                                    exit(1)
                                elif key & 0xFF == ord('n'):
                                    print("=> skipping to track next sequence ...")
                                    out_video.release()
                                    break
                            out_video.write(left_img)
                    if save_video:
                        out_video.release()
                        print('=> Videos saved at', video_file)
                        if sys.platform == 'win32' or sys.platform == 'win64':
                            cv2.destroyWindow('visualization')

                    # record results
                    tracking_results = np.array(tracking_results).reshape(-1, 4)
                    overlaps = np.array(overlaps)

                    # record bounding boxes
                    np.savetxt(record_file, tracking_results, fmt='%.3f', delimiter=',')
                    print('=> Tracking results recorded at', record_file)

                    # record overlaps
                    overlap_file = record_file[:record_file.rfind('_')] + '_overlap.txt'
                    np.savetxt(overlap_file, overlaps, fmt='%.3f', delimiter=',')

                    # record running times
                    time_file = record_file[:record_file.rfind('_')] + '_time.txt'
                    np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def scaled_run(self, tracker, save_video=False, overwrite_result=False, scales=6):
        print('=> Running tracker %s on %s...' % (tracker.name, self.dataset.dataset_name))
        self.scales = scales
        for scale in range(scales):
            print('=> scale level: {}'.format(scale + 1))
            scale_factor = 1 - scale * (1 - self.scale_ratio) / scales
            img_size = ast.literal_eval(self.dataset.meta['resolution'])
            scaled_size = (int(img_size[0] * scale_factor), int(img_size[1] * scale_factor))
            print('=> scaled img size: {}'.format(scaled_size))

            # loop over the complete dataset
            for s, sequence in enumerate(self.dataset):
                sequence_name = sequence.name
                print('=> Sequence %s (%dth in total %d)' % (sequence_name,
                                                             s + 1, len(self.dataset)))
                print('=> Sequence length %d' % sequence.__len__())
                # run multiple repetitions for each sequence
                for r in range(self.repetitions):
                    print('=> Repetition: %d' % (r + 1))
                    record_file = os.path.join(
                        self.result_dir, tracker.name, sequence_name,
                        '{}_{:02d}_scale{:d}_bbox.txt'.format(sequence_name, r + 1, scale + 1))
                    print('record_file: {}'.format(record_file))
                    record_dir = os.path.dirname(record_file)
                    if not os.path.isdir(record_dir):
                        os.makedirs(record_dir)

                    if os.path.exists(record_file) and not overwrite_result:
                        print('=> Found results, skipping', sequence_name)
                    else:
                        # tracking loop
                        print("=> monocular tracker {} starting ...".format(tracker.name))
                        tracking_results = []
                        overlaps = []
                        times = []
                        bound = ast.literal_eval(self.dataset.meta['resolution'])

                        video_file = record_file[:record_file.rfind('_')] + '_video.avi'
                        # if os.path.exists(video_file) and not overwrite_result:
                        #     print("=> Found video results, skipping %s" % sequence_name)
                        #     continue
                        width, height = scaled_size
                        out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height),
                                                    True)

                        initial_flag = True
                        for f, (left_img_file, left_roi, _, _, truncation) in enumerate(sequence):
                            print('=> Frame %d/%d in sequence %s' % (f, sequence.__len__(), sequence_name))
                            left_img = cv2.imread(left_img_file, 1)
                            if left_img is None:
                                print("frame {} failed!".format(f))
                                print("=> skipping to track next sequence ...")
                                break
                            if truncation > 2:
                                print("=> the whole object is truncated in frame{}".format(f))
                                print("=> skipping to the next frame ...")
                                continue
                            # scale down image
                            left_img = cv2.resize(left_img, scaled_size, cv2.INTER_LINEAR)
                            # scale down annotations
                            left_roi = np.array(left_roi) * scale_factor
                            left_roi = left_roi.astype(int)

                            if f == 0 or initial_flag:
                                print("=> monocular tracker {} initializing ...".format(tracker.name))
                                print('left initial roi: {}'.format(left_roi))
                                start_time = time.time()
                                tracker.init(left_img, left_roi)
                                end_time = time.time()
                                elapsed_time = (end_time - start_time)
                                print('=> initial elapsed time: {} s'.format(elapsed_time))
                                times.append(elapsed_time)
                                tracking_results.append(left_roi)
                                overlaps.append(1.0)
                                print("=> The initialization of monocular tracker {} have been done!".format(
                                    tracker.name))
                                left_bbox2d = left_roi
                                initial_flag = False
                            else:
                                print('left roi: {}'.format(left_roi))
                                start_time = time.time()
                                left_bbox2d = tracker.update(left_img)
                                end_time = time.time()
                                elapsed_time = (end_time - start_time)
                                print('=>tracking elapsed time: {}s'.format(elapsed_time))
                                times.append(elapsed_time)
                                print('tracking result: {}'.format(left_bbox2d))
                                tracking_results.append(left_bbox2d)
                                left_iou = self.bbox2d_overlap_calc(left_roi, left_bbox2d, bound)
                                overlaps.append(left_iou)
                                print('overlap: {}'.format(left_iou))

                            if save_video:
                                # bbox2d ground-truth
                                left_pt1_gt = [left_roi[0], left_roi[1]]
                                left_pt2_gt = [left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]]

                                # bbox2d result of mono tracker
                                left_pt1 = [int(left_bbox2d[0]), int(left_bbox2d[1])]
                                left_pt2 = [int(left_bbox2d[0] + left_bbox2d[2]), int(left_bbox2d[1] + left_bbox2d[3])]

                                # draw groud-truth bbox2d
                                cv2.rectangle(left_img, pt1=tuple(left_pt1_gt), pt2=tuple(left_pt2_gt),
                                              color=(255, 255, 0), thickness=2)

                                # draw bbox2d of tracker
                                cv2.rectangle(left_img, pt1=tuple(left_pt1), pt2=tuple(left_pt2),
                                              color=(0, 255, 0), thickness=2)

                                # if sys.platform == 'win32' or sys.platform == 'win64':
                                #     cv2.namedWindow('visualization', cv2.WINDOW_GUI_EXPANDED)
                                #     cv2.imshow('visualization', left_img)
                                #     key = cv2.waitKey(1)
                                #     if key & 0xFF == 27:
                                #         print('=> tracking process stopped !')
                                #         out_video.release()
                                #         exit(1)
                                #     elif key & 0xFF == ord('n'):
                                #         print("=> skipping to track next sequence ...")
                                #         out_video.release()
                                #         break
                                out_video.write(left_img)

                        out_video.release()
                        print('=> Videos saved at', video_file)
                        # if sys.platform == 'win32' or sys.platform == 'win64':
                        #     cv2.destroyWindow('visualization')

                        # record results
                        tracking_results = np.array(tracking_results).reshape(-1, 4)
                        overlaps = np.array(overlaps)

                        # record bounding boxes
                        np.savetxt(record_file, tracking_results, fmt='%.3f', delimiter=',')
                        print('Tracking results recorded at', record_file)

                        # record overlaps
                        overlap_file = record_file[:record_file.rfind('_')] + '_overlap.txt'
                        np.savetxt(overlap_file, overlaps, fmt='%.3f', delimiter=',')

                        # record running times
                        time_file = record_file[:record_file.rfind('_')] + '_{}_time.txt'.format(sys.platform)
                        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def report(self, tracker_names, plot_curves=True):
        assert isinstance(tracker_names, (list, tuple))
        report_files = []
        for name in tracker_names:
            performance = {}
            report_dir = os.path.join(self.report_dir, name)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            report_file = os.path.join(report_dir, 'performance.json')
            report_files.append(report_file)

            print('=> {} evaluating'.format(name))
            ious = {}
            times = {}

            performance.update({name: {
                'sequence_wise': {}}})

            for s, sequence in enumerate(self.dataset):
                sequence_name = sequence.name
                print('=> Sequence %s (%dth in total %d)' % (sequence_name,
                                                             s + 1, len(self.dataset)))
                iou_files = glob.glob(os.path.join(
                    self.result_dir, name, sequence_name,
                    '*_overlap.txt'))
                if len(iou_files) == 0:
                    raise Exception('=> Results for sequence %s not found.' % sequence_name)

                # read overlaps of all repetitions
                ious[sequence_name] = []
                iou_data = [np.loadtxt(f, delimiter=',') for f in iou_files]
                sequence_ious = np.concatenate(iou_data)
                ious[sequence_name] = sequence_ious

                # stack all tracking times
                times[sequence_name] = []
                time_files = glob.glob(os.path.join(
                    self.result_dir, name, sequence_name,
                    '*_time.txt'))
                time_data = [np.loadtxt(t, delimiter=',') for t in time_files]
                sequence_times = np.concatenate(time_data)
                if len(sequence_times) > 0:
                    times[sequence_name] = sequence_times

                # store sequence-wise performance
                ao, sr, speed, _ = self._evaluate(sequence_ious, sequence_times)

                performance[name]['sequence_wise'].update(
                    {os.path.join(sequence_name): {
                        'ao': ao,
                        'sr': sr,
                        'speed_fps': speed,
                        'length': sequence.__len__() - 1}})

            ious = np.concatenate(list(ious.values()))
            times = np.concatenate(list(times.values()))

            # store overall performance
            ao, sr, speed, succ_curve = self._evaluate(ious, times)

            performance[name].update({'overall': {
                'ao': ao,
                'sr': sr,
                'speed_fps': speed,
                'succ_curve': succ_curve.tolist()}})

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)
        # plot success curves
        if plot_curves:
            keys = ['overall']
            self.plot_curves(report_files, tracker_names)

    def repetition_report(self, tracker_names, plot_curves=True, overwrite=True):
        assert isinstance(tracker_names, (list, tuple))

        report_files = []

        for name in tracker_names:
            performance = {}
            report_dir = os.path.join(self.report_dir, name)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            report_file = os.path.join(report_dir, 'repetitions.json')
            report_files.append(report_file)
            if os.path.exists(report_dir) and not overwrite:
                print('=> Found repetition report file, skipping to next tracker...')
                continue

            print('===================================================')
            print('=> {} evaluating'.format(name))

            performance.update({name: {}})
            for r in range(self.repetitions):
                print('=> repetition: {}'.format(r + 1))
                rep_name = 'rep{:02d}'.format(r + 1)
                performance[name].update({rep_name: {}})

                ious = {}

                performance[name][rep_name].update({'sequence_wise': {}})
                for s, sequence in enumerate(self.dataset):
                    sequence_name = sequence.name
                    print('=> Sequence %s (%dth in total %d)' % (sequence_name,
                                                                 s + 1, len(self.dataset)))

                    iou_files = glob.glob(os.path.join(
                        self.result_dir, name, sequence_name,
                        '*_{:02d}_rand7_overlap.txt'.format(r + 1)))

                    if len(iou_files) == 0:
                        raise Exception('=> Results for sequence %s not found.' % sequence_name)

                    # read all the overlaps in one repetition
                    ious[sequence_name] = []
                    iou_data = [np.loadtxt(f, delimiter=',') for f in iou_files]
                    sequence_ious = np.concatenate(iou_data)
                    ious[sequence_name] = sequence_ious

                    # store sequence-wise performance
                    ao, sr, speed, _ = self._evaluate(sequence_ious)

                    performance[name][rep_name]['sequence_wise'].update(
                        {sequence_name: {
                            'ao': ao,
                            'sr': sr}})

                ious = np.concatenate(list(ious.values()))

                # store overall performance
                ao, sr, speed, _ = self._evaluate(ious)

                performance[name][rep_name].update({'overall': {
                    'ao': ao,
                    'sr': sr}})
            print('===================================================')

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)

        # plot repetition curves
        if plot_curves:
            self.plot_repetition_curves(report_files, tracker_names)

    def random_report(self, tracker_names, plot_curves=True, overwrite=True):
        assert isinstance(tracker_names, (list, tuple))

        report_files = []
        for name in tracker_names:
            performance = {}
            report_dir = os.path.join(self.report_dir, name)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            report_file = os.path.join(report_dir, 'random_initial.json')
            report_files.append(report_file)
            if os.path.exists(report_file) and not overwrite:
                print('=> Found random initial report file. skipping to next tracker {}'.format(name))
                continue

            print('===================================================')
            print('=> {} evaluating'.format(name))

            performance.update({name: {}})
            for l in range(self.random_level):
                print('=> random level: {}'.format(l))
                rand_name = 'rand{:d}'.format(l)
                performance[name].update({rand_name: {}})

                ious = {}

                performance[name][rand_name].update({'sequence_wise': {}})
                for s, sequence in enumerate(self.dataset):
                    sequence_name = sequence.name
                    print('=> Sequence %s (%dth in total %d)' % (sequence_name,
                                                                 s + 1, len(self.dataset)))
                    iou_files = []
                    for r in range(self.repetitions):
                        iou_file = os.path.join(
                            self.result_dir, name, sequence_name,
                            '{}_{:02d}_rand{:d}_overlap.txt'.format(sequence_name, r + 1, l))
                        iou_files.append(iou_file)

                    if len(iou_files) == 0:
                        raise Exception('=> Results for sequence %s not found.' % sequence_name)

                    # read all the overlaps in one random initialization level
                    ious[sequence_name] = []
                    iou_data = [np.loadtxt(f, delimiter=',') for f in iou_files]
                    sequence_ious = np.concatenate(iou_data)
                    ious[sequence_name] = sequence_ious

                    # store sequence-wise performance
                    ao, sr, speed, _ = self._evaluate(sequence_ious)

                    performance[name][rand_name]['sequence_wise'].update(
                        {sequence_name: {
                            'ao': ao,
                            'sr': sr}})

                ious = np.concatenate(list(ious.values()))

                # store overall performance
                ao, sr, speed, _ = self._evaluate(ious)

                performance[name][rand_name].update({'overall': {
                    'ao': ao,
                    'sr': sr}})
            print('===================================================')

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)

        # plot repetition curves
        if plot_curves:
            self.plot_random_initial_curves(report_files, tracker_names)

    def scale_report(self, tracker_names, plot_curves=True, overwrite=True):
        assert isinstance(tracker_names, (list, tuple))

        report_files = []
        for name in tracker_names:
            performance = {}
            report_dir = os.path.join(self.report_dir, name)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            report_file = os.path.join(report_dir, 'multi-scales.json')
            report_files.append(report_file)
            if os.path.exists(report_file) and not overwrite:
                print('=> Found multi-scales report file. skipping to next tracker {}'.format(name))
                continue

            print('===================================================')
            print('=> {} evaluating'.format(name))

            performance.update({name: {}})
            for scale in range(self.scales):
                print('=> scales level: {}'.format(scale + 1))
                scale_name = 'scale{:d}'.format(scale + 1)
                performance[name].update({scale_name: {}})

                ious = {}
                times = {}

                performance[name][scale_name].update({'sequence_wise': {}})
                for s, sequence in enumerate(self.dataset):
                    sequence_name = sequence.name
                    print('=> Sequence %s (%dth in total %d)' % (sequence_name,
                                                                 s + 1, len(self.dataset)))

                    iou_files = glob.glob(os.path.join(
                        self.result_dir, name, sequence_name,
                        '*_*_scale{:d}_overlap.txt'.format(scale + 1)))

                    if len(iou_files) == 0:
                        raise Exception('=> Results for sequence %s not found.' % sequence_name)

                    # read all the overlaps in one random initialization level
                    ious[sequence_name] = []
                    iou_data = [np.loadtxt(f, delimiter=',') for f in iou_files]
                    sequence_ious = np.concatenate(iou_data)
                    ious[sequence_name] = sequence_ious

                    # stack all tracking times
                    times[sequence_name] = []
                    time_files = glob.glob(os.path.join(
                        self.result_dir, name, sequence_name,
                        '*_*_scale{:d}_{}_time.txt'.format(scale + 1, sys.platform)))
                    time_data = [np.loadtxt(t, delimiter=',') for t in time_files]
                    sequence_times = np.concatenate(time_data)
                    if len(sequence_times) > 0:
                        times[sequence_name] = sequence_times
                    else:
                        raise Exception('=> Time record file for sequence {} not found!'.format(sequence_name))

                    # store sequence-wise performance
                    ao, sr, speed, _ = self._evaluate(sequence_ious, sequence_times)

                    performance[name][scale_name]['sequence_wise'].update(
                        {sequence_name: {
                            'ao': ao,
                            'sr': sr,
                            '{}_speed_fps'.format(sys.platform): speed}})

                ious = np.concatenate(list(ious.values()))
                times = np.concatenate(list(times.values()))

                # store overall performance
                ao, sr, speed, _ = self._evaluate(ious, times)

                performance[name][scale_name].update({'overall': {
                    'ao': ao,
                    'sr': sr,
                    '{}_speed_fps'.format(sys.platform): speed}})
            print('===================================================')

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)
                print('=> successfully write down report data in {}'.format(report_file))

        # plot repetition curves
        if plot_curves:
            self.plot_multi_scales_curves(report_files, tracker_names)

    def _check_deterministic(self, tracker_name, sequence_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, sequence_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % sequence_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())

        return len(set(records)) == 1

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('Results recorded at', record_file)

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _evaluate(self, ious, times=np.array([])):
        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)
        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        # thr_iou = np.linspace(0, 1, 101)
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        bin_iou = np.array([[i >= thr for thr in thr_iou] for i in ious])
        # bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        succ_curve = np.mean(bin_iou, axis=0)
        return ao, sr, speed_fps, succ_curve

    def _plot_curves(self, report_files, tracker_name, keys, extension='.png'):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_name)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot' + extension)

        # markers
        markers = ['-', '--', '-.']
        # markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for k, key in enumerate(keys):
            line, = ax.plot(thr_iou,
                            performance[tracker_name][key]['succ_curve'],
                            markers[k % len(markers)])
            lines.append(line)
            legends.append('%s_%s: [%.3f]' % (
                'SOT3D Baseline', key, performance[tracker_name][key]['ao']))

        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

    def plot_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        # assume tracker_names[0] is your tracker
        report_dir = self.report_dir
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot' + extension)
        key = 'overall_bino'

        # filter performance by tracker_names
        performance = {k: v for k, v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

    def plot_repetition_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        report_dir = self.report_dir
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        repetition_fig = os.path.join(report_dir, 'repetition_plot' + extension)

        # # markers
        # markers = ['-', '--', '-.']
        # markers = [c + m for m in markers for c in [''] * 10]

        # colors
        cmap = plt.cm.get_cmap('Set2', len(tracker_names))

        # filter performance by tracker_names
        performance = {k: v for k, v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t['rep01']['overall']['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot repetition plots
        repetitions = np.linspace(1, self.repetitions, self.repetitions)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            print('=> ploting the repetition curve of {}'.format(name))
            # calculate mAO for different repetitions
            mAOs = []
            for r in repetitions:
                aos = []
                for j in range(int(r)):
                    rep_name = 'rep{:02d}'.format(j + 1)
                    aos.append(performance[name][rep_name]['overall']['ao'])
                mAO = np.mean(aos)
                mAOs.append(mAO)

            line, = ax.plot(repetitions, mAOs, color=cmap(i))
            lines.append(line)
            legends.append('%s' % name)

        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='upper right',
                           bbox_to_anchor=(1., 1.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Repetitions',
               ylabel='mAO',
               xlim=(1, self.repetitions), ylim=(0.3, 0.7),
               title='Repetition plots')
        ax.grid(True)
        plt.xticks(range(1, self.repetitions + 1, 1))
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', repetition_fig)
        fig.savefig(repetition_fig,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

    def plot_random_initial_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        report_dir = self.report_dir
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        random_fig = os.path.join(report_dir, 'random_plot' + extension)

        # markers
        # markers = ['-', '--', '-.']
        # markers = [c + m for m in markers for c in [''] * 2]

        # colors
        cmap = plt.cm.get_cmap('Set2', len(tracker_names))

        # filter performance by tracker_names
        performance = {k: v for k, v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t['rand7']['overall']['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot repetition plots
        random_levels = np.linspace(0, self.random_level - 1, self.random_level)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            print('=> ploting the repetition curve of {}'.format(name))
            # calculate mAO for different repetitions
            aos = []
            for l in range(self.random_level):
                rand_name = 'rand{:d}'.format(l)
                aos.append(performance[name][rand_name]['overall']['ao'])

            line, = ax.plot(random_levels, aos, color=cmap(i))
            lines.append(line)
            legends.append('%s' % name)
            ax.scatter(random_levels, aos, color=cmap(i), marker='d', s=14)
            # ax.plot(random_levels,
            #         [performance[name]['rand0']['overall']['ao']] * self.random_level,
            #         '--', color=cmap(i), alpha=0.6)

        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Random level',
               ylabel='mAO',
               xlim=(0, self.random_level - 1), ylim=(0.2, 0.8),
               title='Random plots')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', random_fig)
        fig.savefig(random_fig,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

    def plot_multi_scales_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        report_dir = self.report_dir
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        multi_scales_fig = os.path.join(report_dir, 'multi_scale_plot' + extension)

        # markers
        # markers = ['-', '--', '-.']
        # markers = [c + m for m in markers for c in [''] * 10]

        # colors
        cmap = plt.cm.get_cmap('Set2', len(tracker_names))

        # filter performance by tracker_names
        performance = {k: v for k, v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t['scale1']['overall']['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot repetition plots
        scales = np.linspace(1, self.scales, self.scales)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            print('=> ploting the multi-scales curve of {}'.format(name))
            # calculate mAO for different repetitions
            aos = []
            for scale in range(self.scales):
                scale_name = 'scale{:d}'.format(scale + 1)
                aos.append(performance[name][scale_name]['overall']['ao'])
            # mAO of tracker in multi-scales
            mAO = np.mean(aos)
            std_dev = np.std(aos, ddof=1)
            line, = ax.plot(scales,
                            aos,
                            color=cmap(i))
            lines.append(line)
            ax.scatter(scales, aos, color=cmap(i), marker='*', s=20)
            legends.append('{}: [{:0.3f} ± {:0.3f}]'.format(name, mAO, std_dev))

        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Scales',
               ylabel='AO',
               xlim=(1, self.scales), ylim=(0.2, 0.8),
               title='Multi-scales plots')
        plt.xticks(range(1, self.scales + 1, 1))
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', multi_scales_fig)
        fig.savefig(multi_scales_fig,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

    def bbox2d_overlap_calc(self, anno, bbox2d, bound=(1920, 1080)):
        rects1 = np.array(anno).reshape(-1, 4)
        rects2 = np.array(bbox2d).reshape(-1, 4)
        assert rects1.shape == rects2.shape
        if bound is not None:
            # bounded rects1
            rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
            rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
            rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
            rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
            # bounded rects2
            rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
            rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
            rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
            rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

        rects_inter = self._intersection(rects1, rects2)
        areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

        areas1 = np.prod(rects1[..., 2:], axis=-1)
        areas2 = np.prod(rects2[..., 2:], axis=-1)
        areas_union = areas1 + areas2 - areas_inter

        eps = np.finfo(float).eps
        ious = areas_inter / (areas_union + eps)
        ious = np.clip(ious, 0.0, 1.0)

        return ious

    def _intersection(self, rects1, rects2):
        r"""Rectangle intersection.

        Args:
            rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
            rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
        """
        assert rects1.shape == rects2.shape
        x1 = np.maximum(rects1[..., 0], rects2[..., 0])
        y1 = np.maximum(rects1[..., 1], rects2[..., 1])
        x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                        rects2[..., 0] + rects2[..., 2])
        y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                        rects2[..., 1] + rects2[..., 3])

        w = np.maximum(x2 - x1, 0)
        h = np.maximum(y2 - y1, 0)

        return np.stack([x1, y1, w, h]).T


class MonoTracker(object):

    def __init__(self, mono_tracker_name='SiamRPN'):
        support_trackers = ['SiamRPN', 'SiamFC', 'KCF', 'KCF_CN', 'KCF_GRAY', 'KCF_HOG',
                            'MOSSE', 'CSK', 'CN', 'DCF_GRAY', 'DCF_HOG', 'DAT',
                            'DSST', 'DSST-LP', 'MKCFup', 'MKCFup-LP', 'STRCF', 'LDES',
                            'ECO-HC', 'ECO', 'CSRDCF', 'CSRDCF-LP', 'BACF', 'SAMF',
                            'Staple', 'Staple-CA', 'MCCTH-Staple']
        assert mono_tracker_name in support_trackers
        self.name = mono_tracker_name

        if mono_tracker_name == 'SiamRPN':
            self.tracker = TrackerSiamRPN('trackers/SiamRPN/model.pth')

        elif mono_tracker_name == 'SiamFC':
            self.tracker = TrackerSiamFC('trackers/SiamFC/model.pth')

        elif mono_tracker_name == 'KCF':
            self.tracker = KCFTracker(True, True, True)

        elif mono_tracker_name == 'MOSSE':
            self.tracker = MOSSE()

        elif mono_tracker_name == 'CSK':
            self.tracker = CSK()

        elif mono_tracker_name == 'CN':
            self.tracker = CN()

        elif mono_tracker_name == 'DSST':
            self.tracker = DSST(dsst_config.DSSTConfig())

        elif mono_tracker_name == 'Staple':
            self.tracker = Staple(config=staple_config.StapleConfig())

        elif mono_tracker_name == 'Staple-CA':
            self.tracker = Staple(config=staple_config.StapleCAConfig())

        elif mono_tracker_name == 'KCF_CN':
            self.tracker = KCF(features='cn', kernel='gaussian')

        elif mono_tracker_name == 'KCF_GRAY':
            self.tracker = KCF(features='gray', kernel='gaussian')

        elif mono_tracker_name == 'KCF_HOG':
            self.left_tracker = KCF(features='hog', kernel='gaussian')

        elif mono_tracker_name == 'DCF_GRAY':
            self.tracker = KCF(features='gray', kernel='linear')

        elif mono_tracker_name == 'DCF_HOG':
            self.tracker = KCF(features='hog', kernel='linear')

        elif mono_tracker_name == 'DAT':
            self.tracker = DAT()

        elif mono_tracker_name == 'ECO-HC':
            self.tracker = ECO(config=otb_hc_config.OTBHCConfig())

        elif mono_tracker_name == 'ECO':
            self.tracker = ECO(config=otb_deep_config.OTBDeepConfig())

        elif mono_tracker_name == 'BACF':
            self.tracker = BACF()

        elif mono_tracker_name == 'CSRDCF':
            self.tracker = CSRDCF(config=csrdcf_config.CSRDCFConfig())

        elif mono_tracker_name == 'CSRDCF-LP':
            self.tracker = CSRDCF(config=csrdcf_config.CSRDCFLPConfig())

        elif mono_tracker_name == 'SAMF':
            self.tracker = SAMF()

        elif mono_tracker_name == 'LDES':
            self.tracker = LDES(ldes_config.LDESDemoLinearConfig())

        elif mono_tracker_name == 'DSST-LP':
            self.tracker = DSST(dsst_config.DSSTLPConfig())

        elif mono_tracker_name == 'MKCFup':
            self.tracker = MKCFup(config=mkcf_up_config.MKCFupConfig())

        elif mono_tracker_name == 'MKCFup-LP':
            self.tracker = MKCFup(config=mkcf_up_config.MKCFupLPConfig())

        elif mono_tracker_name == 'STRCF':
            self.tracker = STRCF()

        elif mono_tracker_name == 'MCCTH-Staple':
            self.tracker = MCCTHStaple(config=mccth_staple_config.MCCTHOTBConfig())

        else:
            print('=> mono-tracker {} is not supported!'.format(mono_tracker_name))
            print('=> supported trackers list: \n{}'.format(support_trackers))
        pass

    def init(self, img, roi):
        print('=> start to initialize monocular tracker ...')
        self.tracker.init(img, roi)
        print('=> monocular tracker have been initialized successfully!')

    def update(self, img):
        bbox2d = self.tracker.update(img)
        return bbox2d


parser = argparse.ArgumentParser("Mono evaluation")
parser.add_argument('--dataset_root', default='/home/group1/dzhou/dataset/NEAT_Supplies/Dataset', type=str,
                    help='the root directory of SNCOVT dataset')

args = parser.parse_args()

if __name__ == '__main__':
    dataset_root = args.dataset_root
    mono_trackers = [
        MonoTracker('SiamFC'),
        MonoTracker('SiamRPN'),
        MonoTracker('ECO'),
        MonoTracker('DAT'),
        MonoTracker('MOSSE'),
        MonoTracker('Staple'),
        MonoTracker('KCF'),
        MonoTracker('CSK'),
        MonoTracker('Staple-CA'),
        MonoTracker('KCF_CN'),
        MonoTracker('KCF_GRAY'),
        MonoTracker('KCF_HOG'),
        MonoTracker('DCF_GRAY'),
        MonoTracker('DCF_HOG'),
        MonoTracker('ECO-HC'),
        MonoTracker('BACF'),
        MonoTracker('CSRDCF'),
        MonoTracker('CSRDCF-LP'),
        MonoTracker('SAMF'),
        MonoTracker('LDES'),
        MonoTracker('MKCFup'),
        MonoTracker('MKCFup-LP'),
        MonoTracker('STRCF'),
        MonoTracker('MCCTH-Staple'),
        MonoTracker('CN'),
        MonoTracker('DSST'),
        MonoTracker('DSST-LP')
    ]
    mono_evaluation = ExperimentMonoTracker(dataset_root, subset='test', use_dataset=True, repetition=3, min_scale_ratio=0.05)
    tracker_names = []
    for tracker in mono_trackers:
        tracker_names.append(tracker.name)
        mono_evaluation.random_run(tracker, save_video=True, overwrite_result=False, rand_level=7)

        # different image scales
        # mono_evaluation.scaled_run(tracker, save_video=True, overwrite_result=True, scales=5)
        # mono_evaluation.report([tracker.name], plot_curves=True)

    # mono_evaluation.random_report(tracker_names, plot_curves=True, overwrite=False)
    mono_evaluation.repetition_report(tracker_names, plot_curves=True, overwrite=True)
    # mono_evaluation.scale_report(tracker_names, plot_curves=True, overwrite=False)
