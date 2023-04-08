# 该.py文件用于删除保存的训练权重.ckpt文件，以方便重新训练：
# 可选删除目标目录或当前终端运行目录；
# 可选由个人决定删除或全部删除；

import os
import glob

# 目标目录
# directory = "/path/to/directory"

# 当前目录
directory = os.getcwd()

# 查找目标目录下后缀为.ckpt的文件
ckpt_files = glob.glob(os.path.join(directory, "*.ckpt"))

def all_delete():
    """
    全部删除
    """
    if not ckpt_files:
        print("The files do not exist in the target directory.")
    else:  
        # 遍历找到的文件并全部删除
        for file in ckpt_files:
            os.remove(file)
        print("Done.")
    
    
def choose_delete():
    """
    选择删除
    """
    # 遍历找到的文件并显示，根据yes/no删除
    if not ckpt_files:
        print("The files do not exist in the target directory.")
    else:
        deleted_files = []
        for file in ckpt_files:
            yes_no = input(f"Found:\n{file}, Delete it?(y/n)")
            if yes_no == 'y':
                deleted_files.append(file)
                os.remove(file)
            else:
                continue
        print("Done.\n")
        print("Delete list:\n")
        print(deleted_files)



all_delete()

    