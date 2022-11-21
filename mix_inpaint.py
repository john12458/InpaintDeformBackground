import shutil
import os
src_dir = "/workspace/data/inpaint/Inpaint_dataset1/"
dst_dir = "/workspace/data/inpaint/mix/"

src_origin_dir = f"{src_dir}/origin/"
src_warpped_dir = f"{src_dir}/warpped/"
src_mask_dir = f"{src_dir}/mask/"

dst_origin_dir = f"{dst_dir}/origin/"
dst_warpped_dir = f"{dst_dir}/warpped/"
dst_mask_dir = f"{dst_dir}/mask/"

def copy(srcDir,dstDir):
    print("start copy...")
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)
    for fname in os.listdir(srcDir):
        src = srcDir + fname
        dst = dstDir + "I1_" + fname 
        shutil.copyfile(src, dst)
    print("done.")
copy(src_origin_dir, dst_origin_dir)
copy(src_warpped_dir, dst_warpped_dir)
copy(src_mask_dir,dst_mask_dir)