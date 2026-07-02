

def file_operation():
    import os
    import shutil


    mode = "T2"
    phase = "test"

    root = f"E:/mri_age/dataset/data/{phase}" #train #test
    folders = os.listdir(root)
    for folder in folders:
        img_path = f"E:/mri_age/dataset/data/{phase}/{folder}/{mode}/NIfTI"
        try:
            for file in os.listdir(img_path):
                    if file.endswith(".gz"):
                        #print(file)
                        src = f"{img_path}/{file}"
                        dst = f"E:/mri_age/dataset/3D_{mode}/{file}"
                        shutil.move(src, dst)
                        print(dst)
                        print(src)
                        break
        except:
            continue




def skull_stripping():

    import subprocess
    import os
    path = f"E:/mri_age/dataset/3D_T2/test" #f"E:/mri_age/dataset/3D_{mode}/{file}"
    

    files = os.listdir(path)
    for file in files:
        in_path = f"{path}/{file}"
        out_path = f"E:/mri_age/dataset/3D_T2_skstp/test/{file}"

        subprocess.run([
            "hd-bet",
            "-i", in_path,
            "-o", out_path
        ])

if __name__=='__main__': 
    skull_stripping()