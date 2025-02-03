## Exploratory write-up
This following file serves as an **informal** exploratory write-up on why I thought this method would improve performance in cross-modality single-source domain generalization for medical segmentation tasks. Take a look if you want to see my **thought process, related works, results, and evaluations** for this whole experiment. 

[Download pdf](https://github.com/DillanImans/brats_grl_vffc/blob/raw/main/Exploratory%20write-up.pdf)


## Code Usage
This section will only cover the usage for configuring, training, visualizing, and testing for a DDP & GRL-based single-source domain segmentation on the BraTS 2021 dataset. If you want to do all these things using all 4 modalities from the BraTS dataset, refer to the [original repository](https://github.com/faizan1234567/Brain-Tumors-Segmentation/tree/main).

I will say that while we have a lot of "config options", most other big feature edits such as augmentations, preprocessing, or model changes need to be manually edited in the code. If you read the code, you'll easily understand what you need to change. Though I'll try to explain the most important changes that you can make.


##### Data Preproc + Augment changes
All changes are to be made in `brats.py` or `augment.py`.

`augment.py` have *DataAugmenter* classes that are used to augment the data whilst IN TRAINING. I don't usually use this as I augment the data immediately before creating the data loader, hence I always turn off augment in the actual `train_grl.py`. Though if you want, you can make the data non-augmented and use the *DataAugmenter* class instead.

`brats.py` is the main area to edit data augments, transformations, etc. 
1. The *BraTSIndependent* class is to preprocess the BraTS dataset for a single modality, and should not be changed.
2. The *BraTSBezierAug3DDataset* class is to augment the data with the bezier intensity transformations and to give it a domain label. It also subsets datasets. You can uncomment out the "self.monai_3d_augment" stuff if you want flip and rotates. You can also add or delete augments from the "monai_3d_augment function". Otherwise, currently it only creates two subsets: domain label 1 with 500 patients with the positive gradient intensity transformation & domain label 2 with another 500 patients with the negative gradient intensity transformation.
You want to edit the *BraTSBezierAug3DDataset* if you want to add preprocessing, augmentations, etc. You should only touch *BraTSIndependent*if you know what you're doing.



##### Dataset Visualization
All changes are to be made in *visualizeDatasetTesting.py*

Basically, this is to visualize the augmentations and transformations to the data. All you need to edit is in *inspect_and_visualize_dataset* at the bottom.
- use_bezier: bezier transformation activate
- domain_label: set domain label, important for checking the specific augmentation made to the specific domain
- use_augmenter: this is to use the augment in `augment.py`. This should be false if use_bezier is and augment is true. This should be true if use_bezier and augment is false since you are just passing the un-bezier'd data.
- sample_id: this just forces a specific patient to be visualized.

It'll then output you a visualization_output.png which has a single slice of the data and the 3 labels (et, tc, wt).

##### Training
All changes are to be made in `train_grl.py`

1. *def main* is where you wanna edit a couple things. First of all, there is a section of cfg.model.architecture where it's all different model architectures with their own hyperparameters. You can change a lot of their hyperparameters here, mainly "in_channels" if you want to change different number of channels that you give (if you want to train more than one modality) as well as "num_domains" for the number of domains you want to give (right now its 1 in channel as its a single modality and 2 num_domains as its domain_label 1 and domain_label 2). There is also a section for the different datasets to use, most of them commented out. You can uncomment and comment out the one you want to use, but make sure the "in_channels" is changed appropriately depending on the number of modalities you pass. The current uncommented one is the GRL based one.
2. The command you want to run is:
`torchrun --nproc_per_node=4 train_grl.py     dataset.dataset_folder=/home/monetai/Desktop/dillan/dataBratsFormatted training.new_max_epochs=100 training.batch_size=1 training.val_every=5 training.learning_rate=1e-4 model.architecture=grlsegres_net training.schedule_alpha=True training.lambda_domain=0 training.resume=False`
Which I think is self explanatory on what you need to change depending on your preferences. To clarify further, "schedule_alpha" is the lambda scheduler, which starts at "lambda_domain = 0". If you change this to False, then the lambda will be constantly the one you set in "lambda_domain". If it's false and "lambda_domain = 0", then technically you are fully neglecting the GRL and only pass segmentation loss, making it a normal training WITHOUT GRL.

For the dataset folder, you do need to make it organized correctly. Check the main github link [here](https://github.com/faizan1234567/Brain-Tumors-Segmentation/tree/main) to confirm your dataset's organization.

After training, this will output a "segres_net_runs" folder that has the checkpoint and the best model you can use to test on.


##### Testing and Evaluating
All changes are to be made in `test.py` and `averageDiceTest.py`

`test.py`
1. Not much to change here. There is "in_channels" at line 156 to change if you want to use different number of modalities. There is also network architecture hyperparameters to change if you want. And also the dataset thing similar to the ones in training. For more details just check the training explanations above.
2. The command you want to run is:
`python test.py test.weights=/home/monetai/Desktop/dillan/code/brrr/Brain-Tumors-Segmentation/archiveModels/SEGRESNET_t2noaug/best-model/best-model.pkl dataset.dataset_folder=/home/monetai/Desktop/dillan/dataBratsFormatted test.batch=1 model.architecture=grlsegres_net`
You want to make sure that the .pkl points to a trained model, and the model.architecture uses the same exact architecture. The .pkl is obtained after you finish all training epochs, in the new outputted folder.

`averageDiceTest.py`
After you do the test run, you'll get a csv. You can then run this .py file to get the average dice scores for all wt (whole tumor), et (enhancing tumor), and tc (tumor core), and the mean from all 3. You should change the file_path here 
by editing the file_path correctly.


##### Ratio of GIN and GOUT + Check Model
All changes to be made in `segresnet.py` and `check_model.py`

`segresnet.py`
This is where the GRL based SegResNet is called. Don't change anything if you don't know what you're doing. However there is one thing to be examined here, that is in line 417-420 for "ratio_gin" and "ratio_gout". If you want to go into specifics what this is, you'll have to check the vffc paper (check the acknowledgements section below), but essentially it's how much of the features you want the frequency module to learn more on "local patches" vs to learn more on "global patches". Read the paper if you want to see how this works. Or else just leave it to 0.5, 0.5.

`check_model.py`
This is just code to check the shapes and channels of the model. You can run it to see an overview.



## Acknowledgements
This repository is an extension from [this BraTS pipeline](https://github.com/faizan1234567/Brain-Tumors-Segmentation/tree/main). The VFFC method is based on [this repository](https://github.com/aimagelab/vffc/tree/main?tab=readme-ov-file), which was introduced in the following paper:
*Quattrini, Fabio & Pippi, Vittorio & Cascianelli, Silvia & Cucchiara, Rita. (2023). Volumetric Fast Fourier Convolution for Detecting Ink on the Carbonized Herculaneum Papyri. 10.48550/arXiv.2308.05070.* 

