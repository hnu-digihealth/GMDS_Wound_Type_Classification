#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2025 AG-RAIMIA-Müller, University of Augsburg                 #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import argparse
import pandas as pd
from collections import defaultdict
import random
import numpy as np
# TensorFlow libraries
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       EarlyStopping
from tensorflow.keras.metrics import AUC
# AUCMEDI libraries
from aucmedi import *
from aucmedi.utils.class_weights import compute_class_weights
from aucmedi.sampling import sampling_split
from aucmedi.neural_network.loss_functions import categorical_focal_loss
from aucmedi.evaluation import *
from aucmedi.ensemble import Bagging

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="AUCMEDI Training for Wound Type Classification")
parser.add_argument("--input", help="Path to imaging dataset",
                    required=True, type=str, dest="path_input")
parser.add_argument("--output", help="Path to output directory (will be created)",
                    required=False, type=str, dest="path_output", default="models")
parser.add_argument("--eval", help="Path to evaluation directory (will be created)",
                    required=False, type=str, dest="path_evaluate", default="evaluate")
parser.add_argument("--gpu", help="GPU assignment",
                    required=False, type=int, dest="gpu", default=0)
args = parser.parse_args()

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

# Provide path to imaging data
path_input = args.path_input
# Results directory
path_output = args.path_output
if not os.path.exists(path_output) : os.mkdir(path_output)
path_evaluate = args.path_evaluate
if not os.path.exists(path_evaluate) : os.mkdir(path_evaluate)

# Seed
seed = 1234
# Training paras
batch_size = 48
epochs = 1000
iterations = 180

#-----------------------------------------------------#
#                       Sampling                      #
#-----------------------------------------------------#
def prepare_data():
    # Pillar #1: Initialize input data reader
    ds = input_interface(interface="directory",
                         path_imagedir=path_input,
                         training=True)
    (index_list, class_ohe, nclasses, class_names, image_format) = ds

    # Step 1: Map image path to index in list
    path_to_index = {path: idx for idx, path in enumerate(index_list)}

    # Step 2: Extract patient IDs
    def get_patient_id(path):
        return path.split('/')[1].split('_')[0]  # e.g., 'ulc011'

    # Step 3: Group paths by patient
    patient_to_images = defaultdict(list)
    for path in index_list:
        patient_id = get_patient_id(path)
        patient_to_images[patient_id].append(path)

    # Step 4: Shuffle and split patient IDs
    patient_ids = list(patient_to_images.keys())
    random.shuffle(patient_ids)

    split_ratio = 0.8
    split_point = int(len(patient_ids) * split_ratio)

    train_patients = patient_ids[:split_point]
    test_patients = patient_ids[split_point:]

    # Step 5: Collect paths and indices
    train_paths = [img for patient in train_patients for img in patient_to_images[patient]]
    test_paths = [img for patient in test_patients for img in patient_to_images[patient]]

    train_indices = [path_to_index[img] for img in train_paths]
    test_indices = [path_to_index[img] for img in test_paths]

    # Step 6: Split class_ohe
    class_ohe_train = class_ohe[train_indices]
    class_ohe_test = class_ohe[test_indices]

    # === Output ===
    print(f"Train set: {len(train_paths)} images, {len(train_patients)} patients")
    print(f"Test set: {len(test_paths)} images, {len(test_patients)} patients")

    # Perform dataset exploration - training
    ds_peak = evaluate_dataset(
        samples=train_paths,
        labels=class_ohe_train,
        out_path=path_evaluate,
        class_names=class_names,
        suffix="training",
    )
    print(ds_peak)

    # Perform dataset exploration - testing
    ds_peak = evaluate_dataset(
        samples=test_paths,
        labels=class_ohe_test,
        out_path=path_evaluate,
        class_names=class_names,
        suffix="testing",
    )
    print(ds_peak)

    # return sampling map
    return ((train_paths, class_ohe_train), (test_paths, class_ohe_test)), (nclasses, class_names, image_format)

#-----------------------------------------------------#
#                   AUCMEDI Pipeline                  #
#-----------------------------------------------------#
def run_aucmedi(train_x, train_y, test_x, test_y, ds_meta):
    # Unpack dataset meta information
    (nclasses, class_names, image_format) = ds_meta

    # Initialize Image Augmentation
    aug = ImageAugmentation(flip=True, rotate=True, brightness=True, contrast=True,
                             saturation=True, hue=True, scale=True, crop=False,
                             grid_distortion=False, compression=False, gamma=False,
                             gaussian_noise=True, gaussian_blur=False,
                             downscaling=False, elastic_transform=True)

    # Compute class weights
    cw_loss, _ = compute_class_weights(ohe_array=train_y)

    # Initialize model
    model = NeuralNetwork(nclasses, channels=3, architecture="2D.ConvNeXtBase",
                           loss=categorical_focal_loss(cw_loss),
                           metrics=[AUC(100)],
                           pretrained_weights=True)
    model.tf_epochs = 16

    # Initialize Bagging object for 3-fold cross-validation
    el = Bagging(model, k_fold=3)


    # Pillar #3: Initialize bagging Data Generator
    datagen = DataGenerator(train_x, path_input, 
                            labels=train_y,
                            batch_size=batch_size, 
                            data_aug=aug, 
                            shuffle=True,
                            workers=16,
                            use_multiprocessing=True,
                            resize=model.meta_input,
                            standardize_mode=model.meta_standardize,
                            seed=seed,
                            image_format=image_format)
    
    # Define callbacks
    cb_es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    callbacks = [cb_es]

    # Train model
    el.train(datagen,
             epochs=epochs, iterations=iterations,
             callbacks=callbacks, transfer_learning=True)

    # Dump latest model
    el.dump(path_output)

    # Load model
    el.load(path_output)

    # Pillar #3: Initialize testing Data Generators
    test_gen = DataGenerator(test_x, path_input, 
                            labels=None,
                            batch_size=batch_size, 
                            workers=16,
                            use_multiprocessing=True,
                            data_aug=None, 
                            shuffle=False,
                            resize=model.meta_input,
                            standardize_mode=model.meta_standardize,
                            seed=seed,
                            image_format=image_format)

    # Make predictions
    preds = el.predict(test_gen, aggregate="mean")

    # Store predictions to disk
    df_index = pd.DataFrame(data={"index": test_x})
    df_pd = pd.DataFrame(data=preds, columns=["pd_" + i for i in class_names])
    df_gt = pd.DataFrame(data=test_y, columns=["gt_" + i for i in class_names])
    
    df_merged = pd.concat([df_index, df_pd, df_gt], axis=1, sort=False)
    path_preds = os.path.join(path_evaluate, "preds.ensemble.csv")
    df_merged.to_csv(path_preds, index=False)

    # Evaluate performance
    evaluate_performance(
        preds=preds,
        labels=test_y,
        out_path=path_evaluate,
        class_names=class_names
    )

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
if __name__ == '__main__':
    # Prepare data
    subsets_map, ds_meta = prepare_data()
    train_x, train_y = subsets_map[0]
    test_x, test_y = subsets_map[1]
    # Run AUCMEDI pipeline
    run_aucmedi(train_x, train_y, test_x, test_y, ds_meta)
