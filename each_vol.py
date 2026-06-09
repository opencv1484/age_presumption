import nibabel as nib
import numpy as np
import os
import pandas as pd


def save_each_val():
    # FreeSurfer出力
    input_file = "aseg.nii.gz"

    # 出力先
    save_dir = "brain_regions"
    os.makedirs(save_dir, exist_ok=True)

    # ラベル定義    
    
    regions = {
        "hippocampus":[17,53],
        "amygdala":[18,54],
        "thalamus":[10,49],
        "caudate":[11,50],
        "putamen":[12,51],
        "pallidum":[13,52],
        "lateral_ventricle":[4,43],
        "inferior_lateral_ventricle":[5,44],
        "cerebellum_white":[7,46],
        "cerebellum_cortex":[8,47],
        "brainstem":[16]
    }

    img = nib.load(input_file)
    data = img.get_fdata()

    for region_name, labels in regions.items():

        mask = np.isin(data, labels)

        out_img = nib.Nifti1Image(
            mask.astype(np.uint8),
            img.affine,
            img.header
        )

        save_path = os.path.join(
            save_dir,
            f"{region_name}.nii.gz"
        )

        nib.save(out_img, save_path)

        print(save_path)




def cal_each_val():


    input_file = "aseg.nii.gz"


    regions = {
        "hippocampus":[17,53],
        "amygdala":[18,54],
        "thalamus":[10,49],
        "caudate":[11,50],
        "putamen":[12,51],
        "pallidum":[13,52],
        "lateral_ventricle":[4,43],
        "inferior_lateral_ventricle":[5,44],
        "cerebellum_white":[7,46],
        "cerebellum_cortex":[8,47],
        "brainstem":[16]
    }


    img = nib.load(input_file)
    data = img.get_fdata()



    results = []

    voxel_volume = np.prod(img.header.get_zooms())

    for region_name, labels in regions.items():

        mask = np.isin(data, labels)

        volume_mm3 = mask.sum() * voxel_volume

        results.append(
            [region_name, volume_mm3]
        )

    df = pd.DataFrame(
        results,
        columns=["Region", "Volume_mm3"]
    )

    df.to_csv(
        "brain_volumes.csv",
        index=False
    )

    print(df)
    
    


def cortex():
    import nibabel as nib
    import numpy as np

    img = nib.load("aparc+aseg.mgz")

    data = img.get_fdata()
    cortical_regions = {

        "frontal": [
            1028,2028, # superior frontal
            1003,2003, # caudal middle frontal
            1027,2027, # rostral middle frontal
            1018,2018  # pars opercularis
        ],

        "temporal": [
            1009,2009, # inferior temporal
            1015,2015, # middle temporal
            1030,2030, # superior temporal
            1001,2001, # entorhinal
            1016,2016  # parahippocampal
        ],

        "parietal": [
            1029,2029, # superior parietal
            1008,2008, # inferior parietal
            1025,2025, # precuneus
            1031,2031  # supramarginal
        ],

        "occipital": [
            1011,2011, # lateral occipital
            1005,2005, # cuneus
            1013,2013, # lingual
            1021,2021  # pericalcarine
        ],

        "insula": [
            1035,2035
        ]
    }





#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=========================================================
MRI Feature Visualization Pipeline
DeepWMH + FreeSurfer

Outputs
-------
features.csv
feature_heatmap.nii.gz
overlay.nii.gz
overlay_preview.png

Author:
=========================================================
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


# =====================================================
# Utility
# =====================================================

def ensure_dir(directory):
    """
    出力ディレクトリ作成

    Parameters
    ----------
    directory : str

    Returns
    -------
    None
    """

    os.makedirs(directory, exist_ok=True)


# =====================================================
# NIFTI Loader
# =====================================================

def load_nifti(filepath):
    """
    NIFTI読込

    Parameters
    ----------
    filepath : str

    Returns
    -------
    img : nibabel image
    data : ndarray
    """

    img = nib.load(filepath)
    data = img.get_fdata()

    return img, data


# =====================================================
# FreeSurfer Loader
# =====================================================

def load_freesurfer_segmentation(seg_path):
    """
    aparc+aseg.mgz 読込

    Parameters
    ----------
    seg_path : str

    Returns
    -------
    seg_img
    seg_data
    """

    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    return seg_img, seg_data


# =====================================================
# WMH Loader
# =====================================================

def load_wmh_mask(mask_path):
    """
    DeepWMH出力マスク読込

    Parameters
    ----------
    mask_path : str

    Returns
    -------
    mask_img
    mask_data
    """

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    return mask_img, mask_data


# =====================================================
# Voxel Volume
# =====================================================

def get_voxel_volume(img):
    """
    ボクセル容積(mm3)

    Parameters
    ----------
    img

    Returns
    -------
    voxel_volume
    """

    zooms = img.header.get_zooms()[:3]

    voxel_volume = (
        zooms[0]
        * zooms[1]
        * zooms[2]
    )

    return voxel_volume


# =====================================================
# ROI Labels
# =====================================================

def get_roi_dictionary():
    """
    FreeSurfer ROIラベル辞書

    Returns
    -------
    dict
    """

    return {

        "Left_Thalamus": 10,
        "Left_Caudate": 11,
        "Left_Putamen": 12,
        "Left_Pallidum": 13,
        "Brain_Stem": 16,

        "Left_Hippocampus": 17,
        "Left_Amygdala": 18,

        "Right_Thalamus": 49,
        "Right_Caudate": 50,
        "Right_Putamen": 51,
        "Right_Pallidum": 52,

        "Right_Hippocampus": 53,
        "Right_Amygdala": 54
    }


# =====================================================
# FreeSurfer Features
# =====================================================

def extract_freesurfer_features(
        seg_img,
        seg_data):
    """
    ROI容積抽出(mm3)

    Parameters
    ----------
    seg_img
    seg_data

    Returns
    -------
    features : dict
    """

    voxel_volume = get_voxel_volume(seg_img)

    roi_dict = get_roi_dictionary()

    features = {}

    for roi_name, label in roi_dict.items():

        voxel_count = np.sum(
            seg_data == label
        )

        volume_mm3 = (
            voxel_count
            * voxel_volume
        )

        features[roi_name] = volume_mm3

    return features


# =====================================================
# WMH Features
# =====================================================

def extract_wmh_features(
        mask_img,
        mask_data):
    """
    WMH容積(mm3)

    Parameters
    ----------
    mask_img
    mask_data

    Returns
    -------
    dict
    """

    voxel_volume = get_voxel_volume(
        mask_img
    )

    voxel_count = np.sum(
        mask_data > 0
    )

    volume_mm3 = (
        voxel_count
        * voxel_volume
    )

    return {
        "WMH_Volume_mm3":
            volume_mm3
    }


# =====================================================
# Save CSV
# =====================================================

def save_features_csv(
        features,
        output_csv):
    """
    CSV保存

    Parameters
    ----------
    features
    output_csv
    """

    df = pd.DataFrame([features])

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"[Saved] {output_csv}"
    )


# =====================================================
# ROI Heatmap
# =====================================================

def create_roi_heatmap(
        seg_data,
        features):
    """
    ROIへ特徴量値を投影

    Parameters
    ----------
    seg_data
    features

    Returns
    -------
    heatmap
    """

    heatmap = np.zeros_like(
        seg_data,
        dtype=np.float32
    )

    roi_dict = get_roi_dictionary()

    for roi_name, label in roi_dict.items():

        if roi_name not in features:
            continue

        value = features[roi_name]

        heatmap[
            seg_data == label
        ] = value

    return heatmap


# =====================================================
# WMH Density Heatmap
# =====================================================

def create_wmh_heatmap(
        wmh_mask,
        sigma=2):
    """
    WMH密度ヒートマップ

    Parameters
    ----------
    wmh_mask
    sigma

    Returns
    -------
    density
    """

    density = gaussian_filter(
        wmh_mask.astype(
            np.float32
        ),
        sigma=sigma
    )

    return density


# =====================================================
# Normalize Heatmap
# =====================================================

def normalize_heatmap(
        heatmap):
    """
    0-1正規化

    Parameters
    ----------
    heatmap

    Returns
    -------
    normalized
    """

    mn = heatmap.min()
    mx = heatmap.max()

    if mx == mn:
        return heatmap

    return (
        heatmap - mn
    ) / (
        mx - mn
    )


# =====================================================
# Combine Heatmaps
# =====================================================

def combine_heatmaps(
        roi_heatmap,
        wmh_heatmap,
        roi_weight=1.0,
        wmh_weight=1.0):
    """
    ROI + WMH統合

    Parameters
    ----------
    roi_heatmap
    wmh_heatmap
    roi_weight
    wmh_weight

    Returns
    -------
    combined
    """

    roi_heatmap = normalize_heatmap(
        roi_heatmap
    )

    wmh_heatmap = normalize_heatmap(
        wmh_heatmap
    )

    combined = (
        roi_weight
        * roi_heatmap
        +
        wmh_weight
        * wmh_heatmap
    )

    return combined


# =====================================================
# Overlay
# =====================================================

def create_overlay(
        mri,
        heatmap,
        alpha=0.35):
    """
    MRI+Heatmap

    Parameters
    ----------
    mri
    heatmap
    alpha

    Returns
    -------
    overlay
    """

    heatmap = normalize_heatmap(
        heatmap
    )

    overlay = (
        mri
        +
        alpha * heatmap
    )

    return overlay


# =====================================================
# Save NIFTI
# =====================================================

def save_nifti(
        data,
        reference_img,
        output_path):
    """
    NIFTI保存

    Parameters
    ----------
    data
    reference_img
    output_path
    """

    output_img = nib.Nifti1Image(
        data.astype(
            np.float32
        ),
        reference_img.affine,
        reference_img.header
    )

    nib.save(
        output_img,
        output_path
    )

    print(
        f"[Saved] {output_path}"
    )


# =====================================================
# PNG Preview
# =====================================================


# =====================================================
# Main
# =====================================================

def __main():

    T1_PATH = (
        "input/T1.nii.gz"
    )

    SEG_PATH = (
        "freesurfer/subject1/mri/aparc+aseg.mgz"
    )

    WMH_PATH = (
        "deepwmh/WMH_mask.nii.gz"
    )

    OUTPUT_DIR = (
        "output"
    )

    ensure_dir(
        OUTPUT_DIR
    )

    print(
        "Loading MRI..."
    )

    mri_img, mri = load_nifti(
        T1_PATH
    )

    print(
        "Loading FreeSurfer..."
    )

    seg_img, seg = \
        load_freesurfer_segmentation(
            SEG_PATH
        )

    print(
        "Loading WMH..."
    )

    _, wmh = \
        load_wmh_mask(
            WMH_PATH
        )

    print(
        "Extracting features..."
    )

    fs_features = \
        extract_freesurfer_features(
            seg_img,
            seg
        )

    wmh_features = \
        extract_wmh_features(
            mri_img,
            wmh
        )

    features = {}

    features.update(
        fs_features
    )

    features.update(
        wmh_features
    )

    save_features_csv(
        features,
        os.path.join(
            OUTPUT_DIR,
            "features.csv"
        )
    )

    print(
        "Generating heatmap..."
    )

    roi_heatmap = \
        create_roi_heatmap(
            seg,
            features
        )

    wmh_heatmap = \
        create_wmh_heatmap(
            wmh
        )

    final_heatmap = \
        combine_heatmaps(
            roi_heatmap,
            wmh_heatmap
        )

    save_nifti(
        final_heatmap,
        mri_img,
        os.path.join(
            OUTPUT_DIR,
            "feature_heatmap.nii.gz"
        )
    )

    overlay = \
        create_overlay(
            mri,
            final_heatmap
        )

    save_nifti(
        overlay,
        mri_img,
        os.path.join(
            OUTPUT_DIR,
            "overlay.nii.gz"
        )
    )

    print(
        "\nPipeline Completed."
    )


if __name__ == "__main__":
    __main()





#容積の正規化
#Brain Age Gap 実年齢との差
#SHAP Heatmapはかなり強い
#Radiomics
"""
候補
FirstOrder
GLCM
GLRLM
GLSZM
NGTDM
"""

#eTIV