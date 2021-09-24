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
from trackers.cftracker.config import staple_config, ldes_config, dsst_config, csrdcf_config, mkcf_up_config, mccth_staple_config


class ExperimetBinoTracker(object):
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

    def __init__(self, root_dir, subset='val', result_dir='./bino_results',
                 report_dir='./bino_reports', use_dataset=True):
        super(ExperimetBinoTracker, self).__init__()
        assert subset in ['test', 'train']
        self.subset = subset
        if use_dataset:
            self.dataset = SNCOVT(root_dir, subset=subset)
        self.result_dir = os.path.join(result_dir, self.dataset.dataset_name, subset)
        self.report_dir = os.path.join(report_dir, self.dataset.dataset_name, subset)
        print('evaluation result directory: {}'.format(self.result_dir))
        self.nbins_iou = 201
        self.repetitions = 1

    def normal_run(self, bino_tracker, save_video=False, overwrite_result=False):
        print('================================================')
        print('=> Running bino tracker %s on %s...' % (bino_tracker.name, self.dataset.dataset_name))
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
                    self.result_dir, bino_tracker.name, sequence_name,
                    '%s_%02d_bbox.txt' % (sequence_name, r + 1))
                # print('=> recording file path: {}'.format(record_file))

                if os.path.exists(record_file) and not overwrite_result:
                    print('=> Found results, skipping', sequence_name)
                else:
                    # tracking loop
                    print("=> binocular tracker {} starting ...".format(bino_tracker.name))
                    tracking_results = []
                    overlaps = []
                    times = []
                    bound = ast.literal_eval(self.dataset.meta['resolution'])
                    for f, (left_img_file, left_roi, right_img_file, right_roi, truncation) in enumerate(sequence):
                        print('=> Frame %d/%d in sequence %s' % (f, sequence.__len__(), sequence_name))
                        left_img = cv2.imread(left_img_file, 1)
                        right_img = cv2.imread(right_img_file, 1)
                        if left_img is None and right_img_file is None:
                            print("frame {} failed!".format(f))
                            print("=> skipping to track next sequence ...")
                            break
                        if truncation > 2:
                            print("=> the whole object is truncated in frame {}".format(f))
                            print("=> skipping to the next frame ...")
                            continue

                        if f == 0 :
                            print("=> binocular tracker {} initializing ...".format(bino_tracker.name))
                            start_time = time.time()
                            bino_tracker.bino_init(left_img, left_roi, right_img, right_roi)
                            end_time = time.time()
                            elapsed_time = (end_time - start_time)
                            print('initial elapsed time: {} s'.format(elapsed_time))
                            times.append(elapsed_time)
                            tracking_results.append(np.array(left_roi + right_roi))
                            overlaps.append([1.0, 1.0])
                            print("=> The initialization of binocular tracker {} have been done!".format(bino_tracker.name))
                        else:
                            start_time = time.time()
                            left_bbox2d, right_bbox2d = bino_tracker.bino_update(left_img, right_img)
                            end_time = time.time()
                            elapsed_time = (end_time - start_time)
                            times.append(elapsed_time)
                            print('tracking result: {}'.format(left_bbox2d))
                            print('tracking result: {}'.format(right_bbox2d))
                            print('tracking elapsed time: {}s'.format(elapsed_time))
                            tracking_results.append(np.concatenate((left_bbox2d, right_bbox2d)))
                            left_iou = self.bbox2d_overlap_calc(left_roi, left_bbox2d, bound=bound)
                            right_iou = self.bbox2d_overlap_calc(right_roi, right_bbox2d, bound=bound)
                            overlaps.append(np.concatenate((left_iou, right_iou)))
                            print('left overlap: {}, right overlap: {}'.format(left_iou, right_iou))

                    # record results
                    tracking_results = np.array(tracking_results).reshape(-1, 8)
                    overlaps = np.array(overlaps).reshape(-1, 2)

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
                        print("=> Found video results, skipping %s" % sequence_name)
                        continue
                    width, height = eval(self.dataset.meta['resolution'])
                    out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width*2, height), True)
                    tracking_results = np.loadtxt(record_file, dtype=float, delimiter=',').astype(int)
                    f = 0
                    for left_img_file, left_roi_gt, right_img_file, right_roi_gt, truncation in sequence:
                        print('=> Frame %dth in total %d' % (f, sequence.__len__()))
                        if truncation > 2:
                            # skip the frame of which object is truncated
                            continue
                        left_img = cv2.imread(left_img_file, 1)
                        right_img = cv2.imread(right_img_file, 1)
                        # bbox2d ground-truth
                        left_pt1_gt = [left_roi_gt[0], left_roi_gt[1]]
                        left_pt2_gt = [left_roi_gt[0] + left_roi_gt[2], left_roi_gt[1] + left_roi_gt[3]]
                        right_pt1_gt = [right_roi_gt[0], right_roi_gt[1]]
                        right_pt2_gt = [right_roi_gt[0] + right_roi_gt[2], right_roi_gt[1] + right_roi_gt[3]]
                        # bbox2d result of bino tracker
                        left_roi = tracking_results[f, :4]
                        right_roi = tracking_results[f, 4:8]
                        left_pt1 = [left_roi[0], left_roi[1]]
                        left_pt2 = [left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]]
                        right_pt1 = [right_roi[0], right_roi[1]]
                        right_pt2 = [right_roi[0] + right_roi[2], right_roi[1] + right_roi[3]]
                        # draw groud-truth bbox2d
                        cv2.rectangle(left_img, pt1=tuple(left_pt1_gt), pt2=tuple(left_pt2_gt),
                                      color=(255, 255, 0), thickness=2)
                        cv2.rectangle(right_img, pt1=tuple(right_pt1_gt), pt2=tuple(right_pt2_gt),
                                      color=(255, 255, 0), thickness=2)
                        # draw bbox2d of bino tracker
                        cv2.rectangle(left_img, pt1=tuple(left_pt1), pt2=tuple(left_pt2),
                                      color=(0, 255, 0), thickness=2)
                        cv2.rectangle(right_img, pt1=tuple(right_pt1), pt2=tuple(right_pt2),
                                      color=(0, 255, 0), thickness=2)
                        visualization = np.hstack((left_img, right_img))
                        if sys.platform == 'win32' or 'win64':
                            cv2.namedWindow('visualization', cv2.WINDOW_GUI_EXPANDED)
                            cv2.imshow('visualization', visualization)
                            key = cv2.waitKey(1)
                            if key & 0xFF == 27:
                                print('=> tracking process stopped !')
                                exit(1)
                            elif key & 0xFF == ord('n'):
                                print("=> skipping to track next sequence ...")
                                break
                        out_video.write(visualization)
                        f = f + 1
                    out_video.release()
                    print('Videos saved at', video_file)
                    if sys.platform == 'win32' or 'win64':
                        cv2.destroyWindow('visualization')
        print('================================================')

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
            bino_ious = {}
            left_ious = {}
            right_ious = {}
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
                    raise Exception('Results for sequence %s not found.' % sequence_name)

                # read overlaps of all repetitions
                bino_ious[sequence_name] = []
                left_ious[sequence_name] = []
                right_ious[sequence_name] = []
                iou_data = [np.loadtxt(f, delimiter=',') for f in iou_files]
                sequence_ious = np.concatenate(iou_data)
                left_sequence_ious = sequence_ious[:, 0]
                right_sequence_ious = sequence_ious[:, 1]
                bino_sequence_ious = (left_sequence_ious + right_sequence_ious) / 2
                if len(bino_sequence_ious) > 0 and len(left_sequence_ious) > 0 and len(right_sequence_ious) > 0:
                    bino_ious[sequence_name] = bino_sequence_ious
                    left_ious[sequence_name] = left_sequence_ious
                    right_ious[sequence_name] = right_sequence_ious

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
                bino_ao, bino_sr, bino_speed, _ = self._evaluate(bino_sequence_ious, sequence_times)
                left_ao, left_sr, left_speed, _ = self._evaluate(left_sequence_ious, sequence_times)
                right_ao, right_sr, right_speed, _ = self._evaluate(right_sequence_ious, sequence_times)

                performance[name]['sequence_wise'].update(
                    {os.path.join(sequence_name, 'left'): {
                    'ao': left_ao,
                    'sr': left_sr,
                    'speed_fps': left_speed,
                    'length': sequence.__len__() - 1}})

                performance[name]['sequence_wise'].update(
                    {os.path.join(sequence_name, 'right'): {
                        'ao': right_ao,
                        'sr': right_sr,
                        'speed_fps': right_speed,
                        'length': sequence.__len__() - 1}})

                performance[name]['sequence_wise'].update(
                    {os.path.join(sequence_name, 'bino'): {
                        'ao': bino_ao,
                        'sr': bino_sr,
                        'speed_fps': bino_speed,
                        'length': sequence.__len__() - 1}})

            left_ious = np.concatenate(list(left_ious.values()))
            right_ious = np.concatenate(list(right_ious.values()))
            bino_ious = np.concatenate(list(bino_ious.values()))
            times = np.concatenate(list(times.values()))

            # store overall performance
            bino_ao, bino_sr, bino_speed, bino_success_curve = self._evaluate(bino_ious, times)
            left_ao, left_sr, left_speed, left_success_curve = self._evaluate(left_ious, times)
            right_ao, right_sr, right_speed, right_success_curve = self._evaluate(right_ious, times)

            performance[name].update({'overall_left': {
                'ao': left_ao,
                'sr': left_sr,
                'speed_fps': left_speed,
                'succ_curve': left_success_curve.tolist()}})

            performance[name].update({'overall_right': {
                'ao': right_ao,
                'sr': right_sr,
                'speed_fps': right_speed,
                'succ_curve': right_success_curve.tolist()}})

            performance[name].update({'overall_bino': {
                'ao': bino_ao,
                'sr': bino_sr,
                'speed_fps': bino_speed,
                'succ_curve': bino_success_curve.tolist()}})

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)
                print('=> saving performance file to {}'.format(report_file))

        # plot success curves
        if plot_curves:
            # keys = ['overall_left', 'overall_right', 'overall_bino']
            self.plot_curves(report_files, tracker_names)

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

    def _evaluate(self, ious, times):
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
                tracker_name, key, performance[tracker_name][key]['ao']))

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
                           bbox_to_anchor=(1.05, 0.), borderaxespad=0)

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


class PseudoBinoTracker(object):

    def __init__(self, mono_tracker_name='SiamRPN'):
        support_trackers = ['SiamRPN', 'SiamFC', 'KCF', 'KCF_CN', 'KCF_GRAY', 'KCF_HOG',
                            'MOSSE', 'CSK', 'CN', 'DCF_GRAY', 'DCF_HOG', 'DAT',
                            'DSST', 'DSST-LP', 'MKCFup', 'MKCFup-LP', 'STRCF', 'LDES',
                            'ECO-HC', 'ECO', 'CSRDCF', 'CSRDCF-LP', 'BACF', 'SAMF',
                            'Staple', 'Staple-CA', 'MCCTH-Staple']
        assert mono_tracker_name in support_trackers
        self.name = 'Bino_' + mono_tracker_name
        if mono_tracker_name == 'SiamRPN':
            self.left_tracker = TrackerSiamRPN('trackers/SiamRPN/model.pth')
            self.right_tracker = TrackerSiamRPN('trackers/SiamRPN/model.pth')
        elif mono_tracker_name == 'SiamFC':
            self.left_tracker = TrackerSiamFC('trackers/SiamFC/model.pth')
            self.right_tracker = TrackerSiamFC('trackers/SiamFC/model.pth')
        elif mono_tracker_name == 'KCF':
            self.left_tracker = KCFTracker(True, True, True)
            self.right_tracker = KCFTracker(True, True, True)
        elif mono_tracker_name == 'MOSSE':
            self.left_tracker = MOSSE()
            self.right_tracker = MOSSE()
        elif mono_tracker_name == 'CSK':
            self.left_tracker = CSK()
            self.right_tracker = CSK()
        elif mono_tracker_name == 'CN':
            self.left_tracker = CN()
            self.right_tracker = CN()
        elif mono_tracker_name == 'DSST':
            self.left_tracker = DSST(dsst_config.DSSTConfig())
            self.right_tracker = DSST(dsst_config.DSSTConfig())
        elif mono_tracker_name == 'Staple':
            self.left_tracker = Staple(config=staple_config.StapleConfig())
            self.right_tracker = Staple(config=staple_config.StapleConfig())
        elif mono_tracker_name == 'Staple-CA':
            self.left_tracker = Staple(config=staple_config.StapleCAConfig())
            self.right_tracker = Staple(config=staple_config.StapleCAConfig())
        elif mono_tracker_name == 'KCF_CN':
            self.left_tracker = KCF(features='cn', kernel='gaussian')
            self.right_tracker = KCF(features='cn', kernel='gaussian')
        elif mono_tracker_name == 'KCF_GRAY':
            self.left_tracker = KCF(features='gray', kernel='gaussian')
            self.right_tracker = KCF(features='gray', kernel='gaussian')
        elif mono_tracker_name == 'KCF_HOG':
            self.left_tracker = KCF(features='hog', kernel='gaussian')
            self.right_tracker = KCF(features='hog', kernel='gaussian')
        elif mono_tracker_name == 'DCF_GRAY':
            self.left_tracker = KCF(features='gray', kernel='linear')
            self.right_tracker = KCF(features='gray', kernel='linear')
        elif mono_tracker_name == 'DCF_HOG':
            self.left_tracker = KCF(features='hog', kernel='linear')
            self.right_tracker = KCF(features='hog', kernel='linear')
        elif mono_tracker_name == 'DAT':
            self.left_tracker = DAT()
            self.right_tracker = DAT()
        elif mono_tracker_name == 'ECO-HC':
            self.left_tracker = ECO(config=otb_hc_config.OTBHCConfig())
            self.right_tracker = ECO(config=otb_hc_config.OTBHCConfig())
        elif mono_tracker_name == 'ECO':
            self.left_tracker = ECO(config=otb_deep_config.OTBDeepConfig())
            self.right_tracker = ECO(config=otb_deep_config.OTBDeepConfig())
        elif mono_tracker_name == 'BACF':
            self.left_tracker = BACF()
            self.right_tracker = BACF()
        elif mono_tracker_name == 'CSRDCF':
            self.left_tracker = CSRDCF(config=csrdcf_config.CSRDCFConfig())
            self.right_tracker = CSRDCF(config=csrdcf_config.CSRDCFConfig())
        elif mono_tracker_name == 'CSRDCF-LP':
            self.left_tracker = CSRDCF(config=csrdcf_config.CSRDCFLPConfig())
            self.right_tracker = CSRDCF(config=csrdcf_config.CSRDCFLPConfig())
        elif mono_tracker_name == 'SAMF':
            self.left_tracker = SAMF()
            self.right_tracker = SAMF()
        elif mono_tracker_name == 'LDES':
            self.left_tracker = LDES(ldes_config.LDESDemoLinearConfig())
            self.right_tracker = LDES(ldes_config.LDESDemoLinearConfig())
        elif mono_tracker_name == 'DSST-LP':
            self.left_tracker = DSST(dsst_config.DSSTLPConfig())
            self.right_tracker = DSST(dsst_config.DSSTLPConfig())
        elif mono_tracker_name == 'MKCFup':
            self.left_tracker = MKCFup(config=mkcf_up_config.MKCFupConfig())
            self.right_tracker = MKCFup(config=mkcf_up_config.MKCFupConfig())
        elif mono_tracker_name == 'MKCFup-LP':
            self.left_tracker = MKCFup(config=mkcf_up_config.MKCFupLPConfig())
            self.right_tracker = MKCFup(config=mkcf_up_config.MKCFupLPConfig())
        elif mono_tracker_name == 'STRCF':
            self.left_tracker = STRCF()
            self.right_tracker = STRCF()
        elif mono_tracker_name == 'MCCTH-Staple':
            self.left_tracker = MCCTHStaple(config=mccth_staple_config.MCCTHOTBConfig())
            self.right_tracker = MCCTHStaple(config=mccth_staple_config.MCCTHOTBConfig())
        else:
            print('=> mono-tracker {} is not supported!'.format(mono_tracker_name))
            print('=> supported trackers list: \n{}'.format(support_trackers))
        pass

    def bino_init(self, left_img, left_roi, right_img, right_roi):
        print('=> start to initialize binocular tracker ...')
        self.left_tracker.init(left_img, left_roi)
        self.right_tracker.init(right_img, right_roi)
        print('=> binocular tracker have been initialized successfully!')

    def bino_update(self, left_img, right_img):
        left_bbox2d = self.left_tracker.update(left_img)
        right_bbox2d = self.right_tracker.update(right_img)
        return left_bbox2d, right_bbox2d


parser = argparse.ArgumentParser("Binocular tracking")
parser.add_argument('--dataset_root', default='E:/dataset/NEAT_Supplies/Dataset', type=str, help='the root directory of SNCOVT dataset')
args = parser.parse_args()

if __name__ == '__main__':
    bino_trackers = [
        PseudoBinoTracker('SiamFC', mono_tracker_path='trackers/SiamFC/model.pth'),
        PseudoBinoTracker('SiamRPN', mono_tracker_path='trackers/SiamRPN/model.pth'),
        # PseudoBinoTracker('KCF'),
        PseudoBinoTracker('MOSSE'),
        PseudoBinoTracker('CSK'),
        PseudoBinoTracker('Staple'),
        PseudoBinoTracker('Staple-CA'),
        PseudoBinoTracker('KCF_GRAY'),
        PseudoBinoTracker('KCF_HOG'),
        PseudoBinoTracker('DCF_GRAY'),
        PseudoBinoTracker('DCF_HOG'),
        PseudoBinoTracker('DAT'),
        PseudoBinoTracker('ECO-HC'),
        PseudoBinoTracker('ECO'),
        PseudoBinoTracker('BACF'),
        PseudoBinoTracker('CSRDCF'),
        PseudoBinoTracker('CSRDCF-LP'),
        PseudoBinoTracker('SAMF'),
        PseudoBinoTracker('MKCFup'),
        PseudoBinoTracker('MKCFup-LP'),
        PseudoBinoTracker('STRCF'),
        PseudoBinoTracker('MCCTH-Staple'),
        # # PseudoBinoTracker('KCF_CN'),
        # # PseudoBinoTracker('CN'),
        # # PseudoBinoTracker('DSST'),
        # # PseudoBinoTracker('DSST-LP')
        # PseudoBinoTracker('LDES'),
    ]
    bino_evaluation = ExperimetBinoTracker(args.dataset_root,
                                           report_dir='bino_reports',
                                           result_dir='bino_results',
                                           subset='test', use_dataset=True)
    tracker_names = []
    for tracker in bino_trackers:
        tracker_names.append(tracker.name)
        bino_evaluation.normal_run(tracker, save_video=True, overwrite_result=False)
    bino_evaluation.report(tracker_names, plot_curves=True)