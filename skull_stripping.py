def ras_nifti(in_gz,out_gz):
    import nibabel as nib
    from nibabel.orientations import aff2axcodes
    img = nib.load(in_gz)
    print(aff2axcodes(img.affine))
    img_ras = nib.as_closest_canonical(img)
    #nib.save(img_ras, out_gz)
    



def resample_voxel(in_gz):
    from nibabel.processing import resample_to_output
    import nibabel as nib
    img = nib.load(in_gz)
    voxel_size = img.header.get_zooms()[:3]
    print(voxel_size)
    #img_resampled = resample_to_output(img_ras, voxel_sizes=(1,1,1))


def skull_stripping():
    import subprocess

    subprocess.run([
        "hd-bet",
        "-i", "input.nii.gz",
        "-o", "output_dir"
    ])




def crop():
    import numpy as np

    img = "" #img_data
    coords = np.where(img > 0)

    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()

    cropped = img[x_min:x_max, y_min:y_max, z_min:z_max]


def __bias_field_correction():
    """skull stripping 未使用の場合"""
    import ants
    #pip install antspyx
    
    img = ants.image_read("input.nii.gz")
    # N4 bias field correction
    corrected = ants.n4_bias_field_correction(img)
    ants.image_write(corrected, "output_corrected.nii.gz")


def bias_field_correction():
    """skull stripping 後に使用"""
    import ants
    #pip install antspyx
    img = ants.image_read("input.nii.gz")
    mask = ants.image_read("mask.nii.gz")
    corrected = ants.n4_bias_field_correction(img, mask=mask)
    ants.image_write(corrected, "output_corrected.nii.gz")

def intensity_norm():
    #z-score
    import numpy as np

    img = "" #corrected_img  # numpy配列

    brain = img[img > 0]  # 背景除外
    mean = brain.mean()
    std = brain.std()

    norm = (img - mean) / std


