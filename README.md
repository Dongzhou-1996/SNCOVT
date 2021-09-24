# SNCOVT
`Space non-cooperative object visual tracking (SNCOVT)` with monocular or binocular camera is one of the most significant tasks for on-orbit services. To verify the availability of those generic trackers in aerospace domain, we present a moderate and simulated space non-cooperative object visual tracking dataset, which contains 60 binocular video sequences with manual annotations, `54742 frames` in total. Meanwhile, a new evaluation protocol that is more appropriate to the practical scenario and some novel metrics are adopted into our SNCOVT evaluation toolkit. 


<img src="https://github.com/Dongzhou-1996/SNCOVT/blob/master/screenshots/c-asteroid.gif" width="400px" height="150px"> <img src="https://github.com/Dongzhou-1996/SNCOVT/blob/master/screenshots/s-asteroid.gif" width="400px" height="150px">
<img src="https://github.com/Dongzhou-1996/SNCOVT/blob/master/screenshots/satellite.gif" width="400px" height="150px"> <img src="https://github.com/Dongzhou-1996/SNCOVT/blob/master/screenshots/space_debris.gif" width="400px" height="150px">

See more details about SNCOVT dataset in [2D vision-based tracking algorithm for general space non-cooperative objects](https://doi.org/10.1016/j.actaastro.2021.07.023).

If you use this code or our dataset please cite:
```
@article{ZHOU2021193,
title = {2D vision-based tracking algorithm for general space non-cooperative objects},
journal = {Acta Astronautica},
volume = {188},
pages = {193-202},
year = {2021},
issn = {0094-5765},
doi = {https://doi.org/10.1016/j.actaastro.2021.07.023},
author = {Dong Zhou and Guanghui Sun and Jialin Song and Weiran Yao}
}
```

# Dataset Download
![dataset](./SNCOVT_Dataset.bmp)

**Note that: Training set of SNCOVT included 20 binocular video sequences is totally public, while test set is still preserved**.
- Baidu Netdisk: *available soon* ...
- Google Drive: *available soon* ...

Once download our SNCOVT dataset, please run following commands to check the integrity of dataset with multi-threads:
```
$: git clone https://github.com/Dongzhou-1996/SNCOVT.git
$: cd {repository dir}
$: python
>>> from SNCOVT_Dataset import SNCOVT
>>> sncovt = SNCOVT('{dataset_dir}', subset='train')
>>> sncovt.integrity(thread_nums=4)
```
if return `True`, the dataset is integral. Furhtermore, you can see all the binocular image with `SNCOVT.dataset_display()`, for example:
```
>>> from SNCOVT_Dataset import SNCOVT
>>> sncovt = SNCOVT('{dataset_dir}', subset='train')
>>> sncovt.dataset_display()
```
During dataset displaying, you can press `ESC` key to stop and `n` to visualize next sequence.

The normal method provided by us to access data is as follows:
```
sncovt = SNCOVT(args.dataset_dir, subset='train')
for s, sequence in enumerate(sncovt):
	for i, (left_img_path, left_roi, right_img_path, right_roi, truncation) in enumerate(sequence):
		print('=> Sequence {}, Frame: {}/{}'.format(sequence.name, i+1, sequence.__len__()))
```

# Preparing Trackers
In this repository, we have provided plenty of monocular trackers like SiamRPN, SiamFC, ECO, BACF, MCCTH, SAMF, CSRDCF, STRCF, MKCFup, KCF, DAT, DCF, CSRDCF, CSK, MOSSE, Staple, which are also motioned in our paper.

The summary lists of requirement for above trackers:
- pytorch >= 1.4
- numpy
- matplotlib
- opencv-python
- got10k
- cupy-cuda110 or other version 101, 100, relies on the cuda version on your PC
- mxnet-cu110 or other version, the same as cupy
- numba
- scipy
- h5py
- colorama
- Cython
- tqdm
- Pillow
- pandas

You can also install all the requirements with following command:
```
pip install -r requirements.txt
```
Most of CF trackers are derived from [PyTrackers](https://github.com/Dongzhou-1996/pyCFTrackers). If you want to use these CF based trackers, please implement the install instructions as follows:
```
$: cd {repository path}
$: cd trackers/lib/eco/features
$: python setup.py build_ext --inplace
```

The other trackers to be adopted in our monocular or binocular evaluation algorithm, should wrapper two methods `tracker.init()` & `tracker.update()`.

Pre-trained SiamRPN models:
- [Baidu Netdisk](https://pan.baidu.com/s/1k-J63jnTktWB-qB4fpm_HQ) (code: 1111)
- [Google Drive](https://drive.google.com/drive/folders/1zUIb3-V3Ao5xgeCU9945Bd_E06nXLE91?usp=sharing)

# Monocular evaluation
To evaluate diverse monocular tracking method mentioned above, we create a new class `MonoTracker` to re-wrap them. It mainly consits of three method `MonoTracker.__init()__`, `MonoTracker.init()`, and `MonoTracker.update()`. You can simply evaluate those trackers on our SNCOVT dataset as follows:
```
$: cd {repository path}
$: python
>>> from mono_evaluation import ExperimentMonoTracker, MonoTracker
>>> mono_tracker = MonoTracker('DAT')
>>> mono_evaluation = ExperimentMonoTracker('{dataset_root}', subset='train', use_dataset=True, repetition=3)
>>> mono_evaluation.normal_run(mono_tracker, save_video=True, overwrite_result=True)
>>> mono_evaluation.report([mono_tracker.name], plot_curves=True, overwrite=True)
```

# Binocular evaluation
There is few binocular tracker, we therefore utilize a new class `PseudoBinoTracker` to re-wrap classic monocular tracker for binocular tracking. So you can simpy evaluate those pseudo binocular tracker on our SNCOVT dataset as follows:
```
$: cd {repository path}
$: python
>>> from bino_evaluation import ExperimentBinoTracker, BinoTracker
>>> bino_tracker = PseudoBinoTracker('DAT')
>>> bino_evaluation = ExperimentBinoTracker('{dataset_root}', subset='train', use_dataset=True, repetition=3)
>>> bino_evaluation.normal_run(mono_tracker, save_video=True, overwrite_result=True)
>>> bino_evaluation.report([bino_tracker.name], plot_curves=True, overwrite=True)
```

