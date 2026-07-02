#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import subprocess

import boto3
import nibabel as nib
import numpy as np
import pandas as pd

# =====================================
# AWS
# =====================================

S3_BUCKET = "brain-age"

INPUT_PREFIX = "input/"
FS_PREFIX = "freesurfer/"
MASK_PREFIX = "masks/"
OVERLAY_PREFIX = "overlays/"
FEATURE_PREFIX = "features/"

s3 = boto3.client("s3")

# =====================================
# FreeSurfer Label Mapping
# aparc+aseg.mgz
# =====================================

LOBES = {

    "Frontal": [
        1002,1003,1012,1014,1017,1018,
        1019,1020,1024,1027,1028,1032,
        2002,2003,2012,2014,2017,2018,
        2019,2020,2024,2027,2028,2032
    ],

    "Temporal": [
        1001,1006,1007,1009,
        1015,1030,1033,1034,
        2001,2006,2007,2009,
        2015,2030,2033,2034
    ],

    "Parietal": [
        1008,1022,1025,1029,1031,
        2008,2022,2025,2029,2031
    ],

    "Occipital": [
        1005,1011,1013,1021,
        2005,2011,2013,2021
    ],

    "Insula": [
        1035,
        2035
    ]
}

LEFT_FRONTAL = [
    1002,1003,1012,1014,1017,1018,
    1019,1020,1024,1027,1028,1032
]

RIGHT_FRONTAL = [
    2002,2003,2012,2014,2017,2018,
    2019,2020,2024,2027,2028,2032
]

LEFT_TEMPORAL = [
    1001,1006,1007,1009,
    1015,1030,1033,1034
]

RIGHT_TEMPORAL = [
    2001,2006,2007,2009,
    2015,2030,2033,2034
]

LEFT_PARIETAL = [
    1008,1022,1025,1029,1031
]

RIGHT_PARIETAL = [
    2008,2022,2025,2029,2031
]

# =====================================
# Utility
# =====================================

def download_file(bucket, key, local_file):
    s3.download_file(bucket, key, local_file)

def upload_file(local_file, bucket, key):
    s3.upload_file(local_file, bucket, key)

def voxel_volume(img):
    z = img.header.get_zooms()[:3]
    return z[0] * z[1] * z[2]

def save_nifti(data, ref_img, out_path):

    out = nib.Nifti1Image(
        data.astype(np.float32),
        ref_img.affine,
        ref_img.header
    )

    nib.save(out, out_path)

def create_mask(seg, labels):
    return np.isin(seg, labels).astype(np.uint8)

def create_overlay(mri, mask):

    overlay = np.zeros_like(
        mri,
        dtype=np.float32
    )

    overlay[mask > 0] = mri[mask > 0]

    return overlay

def volume_mm3(mask, voxel_mm3):
    return float(np.sum(mask) * voxel_mm3)

def asymmetry_ratio(left_v, right_v):

    denom = left_v + right_v

    if denom == 0:
        return 0.0

    return abs(left_v - right_v) / denom

# =====================================
# FreeSurfer
# =====================================

def run_freesurfer(input_nifti,
                   subject_id,
                   subjects_dir):

    cmd = [

        "recon-all",

        "-i",
        input_nifti,

        "-s",
        subject_id,

        "-all"
    ]

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = subjects_dir

    subprocess.run(
        cmd,
        check=True,
        env=env
    )

# =====================================
# mgz → nii
# =====================================

def convert_aparc_to_nifti(
        aparc_mgz,
        output_nifti):

    subprocess.run(
        [
            "mri_convert",
            aparc_mgz,
            output_nifti
        ],
        check=True
    )

# =====================================
# Feature Extraction
# =====================================

def extract_features(
        subject_id,
        t1_img,
        aparc_img,
        output_dir):

    t1 = t1_img.get_fdata()
    seg = aparc_img.get_fdata()

    voxel_mm3 = voxel_volume(aparc_img)

    features = {}

    features["SubjectID"] = subject_id

    total_mask = seg > 0

    features["TotalBrainVolume_mm3"] = \
        volume_mm3(
            total_mask,
            voxel_mm3
        )

    left_frontal = volume_mm3(
        create_mask(seg, LEFT_FRONTAL),
        voxel_mm3
    )

    right_frontal = volume_mm3(
        create_mask(seg, RIGHT_FRONTAL),
        voxel_mm3
    )

    left_temporal = volume_mm3(
        create_mask(seg, LEFT_TEMPORAL),
        voxel_mm3
    )

    right_temporal = volume_mm3(
        create_mask(seg, RIGHT_TEMPORAL),
        voxel_mm3
    )

    left_parietal = volume_mm3(
        create_mask(seg, LEFT_PARIETAL),
        voxel_mm3
    )

    right_parietal = volume_mm3(
        create_mask(seg, RIGHT_PARIETAL),
        voxel_mm3
    )

    features["Left_Frontal_Volume"] = left_frontal
    features["Right_Frontal_Volume"] = right_frontal

    features["Left_Temporal_Volume"] = left_temporal
    features["Right_Temporal_Volume"] = right_temporal

    features["Left_Parietal_Volume"] = left_parietal
    features["Right_Parietal_Volume"] = right_parietal

    features["Frontal_Ratio"] = \
        asymmetry_ratio(
            left_frontal,
            right_frontal
        )

    features["Temporal_Ratio"] = \
        asymmetry_ratio(
            left_temporal,
            right_temporal
        )

    for lobe, labels in LOBES.items():

        mask = create_mask(
            seg,
            labels
        )

        overlay = create_overlay(
            t1,
            mask
        )

        features[
            f"{lobe}_Volume_mm3"
        ] = volume_mm3(
            mask,
            voxel_mm3
        )

        mask_file = os.path.join(
            output_dir,
            f"{lobe}.nii.gz"
        )

        overlay_file = os.path.join(
            output_dir,
            f"{lobe}_overlay.nii.gz"
        )

        save_nifti(
            mask,
            t1_img,
            mask_file
        )

        save_nifti(
            overlay,
            t1_img,
            overlay_file
        )

    csv_file = os.path.join(
        output_dir,
        f"{subject_id}_features.csv"
    )

    pd.DataFrame(
        [features]
    ).to_csv(
        csv_file,
        index=False
    )

    return csv_file

# =====================================
# Subject
# =====================================

def process_subject(input_key):

    subject_id = \
        os.path.basename(
            input_key
        ).replace(
            ".nii.gz",
            ""
        )

    with tempfile.TemporaryDirectory() as tmp:

        t1_file = os.path.join(
            tmp,
            "t1.nii.gz"
        )

        download_file(
            S3_BUCKET,
            input_key,
            t1_file
        )

        subjects_dir = os.path.join(
            tmp,
            "subjects"
        )

        os.makedirs(
            subjects_dir,
            exist_ok=True
        )

        run_freesurfer(
            t1_file,
            subject_id,
            subjects_dir
        )

        fs_subject = os.path.join(
            subjects_dir,
            subject_id
        )

        aparc_mgz = os.path.join(
            fs_subject,
            "mri",
            "aparc+aseg.mgz"
        )

        aparc_nii = os.path.join(
            tmp,
            "aparc+aseg.nii.gz"
        )

        convert_aparc_to_nifti(
            aparc_mgz,
            aparc_nii
        )

        upload_file(
            aparc_nii,
            S3_BUCKET,
            f"{FS_PREFIX}{subject_id}/aparc+aseg.nii.gz"
        )

        output_dir = os.path.join(
            tmp,
            "output"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        t1_img = nib.load(
            t1_file
        )

        aparc_img = nib.load(
            aparc_nii
        )

        extract_features(
            subject_id,
            t1_img,
            aparc_img,
            output_dir
        )

        for f in os.listdir(output_dir):

            full = os.path.join(
                output_dir,
                f
            )

            if f.endswith("_overlay.nii.gz"):

                upload_file(
                    full,
                    S3_BUCKET,
                    f"{OVERLAY_PREFIX}{subject_id}/{f}"
                )

            elif f.endswith("_features.csv"):

                upload_file(
                    full,
                    S3_BUCKET,
                    f"{FEATURE_PREFIX}{f}"
                )

            elif f.endswith(".nii.gz"):

                upload_file(
                    full,
                    S3_BUCKET,
                    f"{MASK_PREFIX}{subject_id}/{f}"
                )

# =====================================
# Main
# =====================================

def main():

    resp = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=INPUT_PREFIX
    )

    if "Contents" not in resp:
        return

    for obj in resp["Contents"]:

        key = obj["Key"]

        if not key.endswith(".nii.gz"):
            continue

        print("Processing:", key)

        process_subject(key)

if __name__ == "__main__":
    main()
    #https://surfer.nmr.mgh.harvard.edu/registration.html
    #https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads?utm_source=chatgpt.com