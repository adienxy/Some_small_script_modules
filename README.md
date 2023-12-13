# 一些学习过程中用到的脚本

* ## datasets_convert_csv_to_txt.py
    ### 该.py文件用于数据集文件转换，包含以下功能：
    1. 将原列标签为label,text样式的.csv数据集转换为text,label样式的.txt数据集
    2. 将转换为.txt的数据集shuffle，将以0,1,2,3...形式按label标号排列的数据集打散，例如：

    |  text   | label  |
    |  ----  | ----  |
    | AAA  | 0 |
    | BBB  | 1 |
    | CCC  | 2 |  

    随机打散为

    |  text   | label  |
    |  ----  | ----  |
    | BBB  | 1 |
    | CCC  | 2 |
    | AAA  | 0 |  

    3. 将shuffle后的数据集split为train.txt, test.txt, dev.txt

* ## Delete_ckpt.py
    1. 该.py文件用于删除保存的训练权重.ckpt文件，以方便重新训练：

    2. 可选删除目标目录或当前终端运行目录；

    3. 可选由个人决定删除或全部删除；
* ## a_build_my_data.py
      建立特定格式的数据集
