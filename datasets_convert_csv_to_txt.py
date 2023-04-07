# 该.py文件用于数据集文件转换，包含以下功能：
# 1. 将原列标签为label,text样式的.csv数据集转换为text,label样式的.txt数据集
# 2. 将转换为.txt的数据集shuffle，将以0,1,2,3...形式按label标号排列的数据集打散
# 3. 将shuffle后的数据集split为train.txt, test.txt, dev.txt
 

import pandas as pd
from tqdm import *

def csv_to_txt(csv_dir, label1, label2):
    """
    将csv文件转换为txt文件,并交换label1和label2的先后顺序,中间用制表符隔开;
    label1和label2分别为先后标签顺序,
    例如原文件有两列,第一列为label,第二列为review,则该函数转换后的结果为review,label,中间以制表符隔开
    """
    # 读取csv文件
    df = pd.read_csv(csv_dir) # label,review

    print("Starting convert datasets to txt...\n")
    # 将数据转换为指定格式的字符串
    data_str = ''
    print("Starting...")
    for index, row in tqdm(df.iterrows()):
        # 将原来label,review形式转换为review\tlabel
        data_str += f"{row[label2]}\t{row[label1]}\n"

    # 将字符串写入txt文件
    print("Writing...")
    with open('csv_to_txt.txt', 'w') as file:
        file.write(data_str)
    print("Done.")

def split_txt_to_3(original_file, train_end_value, test_end_value):
    """
    将已经转换为txt文件的数据切分为train.txt,test.txt,dev.txt;
    original_file为原始总数据集txt文件的路径,train_end_value, test_end_value, dev_end_value分别为数据集划分的标签数,
    例如有一个约12w数据的数据集,已经转换为txt格式,
    train_end_value=100000,test_end_value=110000,剩下的数据全划给dev.txt,
    即想让train.txt有10w条数据,test.txt有1w条数据,dev.txt剩下的约1w条数据
    """
    counter = 0
    print("Starting split datasets...\n")
    with open(original_file) as f, open('train.txt', 'w') as file_A, open('test.txt', 'w') as file_B, open('dev.txt', 'w') as file_C:
        print("Start writing...")
        for line in tqdm(f):
            if counter < train_end_value:
                # writing data for tarin.txt
                file_A.write(line)
            elif counter >= train_end_value and counter < test_end_value:
                # writing data for test.txt
                file_B.write(line)
            else:
                # writing data for dev.txt
                file_C.write(line)
            counter += 1
    print("Done.")

def shuffle_datasets(input_file_dir, output_file_dir):
    """
    """
    print("Starting shuffle datasets...\n")
    print("Starting loading...")
    # 读取数据集
    df = pd.read_csv(input_file_dir, sep='\t', header=None, names=['text', 'label'])
    # 打乱数据集
    df = df.sample(frac=1).reset_index(drop=True)
    # 保存打乱后的数据集
    df.to_csv(output_file_dir, sep='\t', header=False, index=False)
    print("Done.")



# csv文件路径
csv_data_dir = 'weibo_senti_100k.csv'


# 将csv转换为txt文件，并交换原csv文件的标签顺序，中间以制表符隔开
csv_to_txt(csv_dir=csv_data_dir, label1='label', label2='review')


txt_data_dir= 'weibo_senti_100k/csv_to_txt.txt'

# 将数据集打乱,并将打乱后的数据集输出为txt格式
shuffled_txt = 'shuffled_txt.txt'
shuffle_datasets(txt_data_dir, shuffled_txt)


# 将数据集划分为train.txt,test.txt,dev.txt,并且train.txt有前10w条数据,text.txt有第100001-110000条数据,dev.txt有剩下的1w条数据
split_txt_to_3(txt_data_dir, 100000, 110000)


