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