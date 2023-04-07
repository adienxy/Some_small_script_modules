# Datasets_Convert
# 该.py文件用于数据集文件转换，包含以下功能：
1. 将原列标签为label,text样式的.csv数据集转换为text,label样式的.txt数据集
2. 将转换为.txt的数据集shuffle，将以0,1,2,3...形式按label标号排列的数据集打散，例如：
|  label   | text  |
|  ----  | ----  |
| 0  | AAA |
| 1  | BBB |
| 2  | CCC |  

随机打散为
|  label   | text  |
|  ----  | ----  |
| 1  | BBB |
| 3  | CCC |
| 0  | AAA |  
3. 将shuffle后的数据集split为train.txt, test.txt, dev.txt
