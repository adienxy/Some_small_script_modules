import  json
from tqdm import tqdm
import os
from nltk.tokenize import word_tokenize
import stanza
from pyabsa import AspectSentimentTripletExtraction as ASTE
from pyabsa import available_checkpoints


### 更改此处适配自身数据集
def extract_datasets_to_build_text_and_label(input_data_dir, output_data_dir):
    """
    提取数据集中每个样本的text和label
    
    ### example:
    [
        {
        "text": "in his first stab at the form , jacquot takes a slightly anarchic approach that works only sporadically .",
        "label": 2
        },
        {
            ...
        }
    ]
    ###
    
    """
    
    with open(input_data_dir, 'r', encoding='UTF-8') as current_file:
        
        print("extracting the data...")
        output_list = []    # 保存提取的数据
        for line in tqdm(current_file):
            data = json.loads(line)
            text_value = data.get('text')   # 提取特定键的值
            label_value = data.get('label')
            
            current_data_dict = {'text': text_value, 'label': label_value}
            output_list.append(current_data_dict)
            # print(current_data_dict)
            
    with open(output_data_dir, 'w', encoding='UTF-8') as current_file:
        # 将提取的数据写入新文件
        print("writing the data to new files...")
        json.dump(output_list, current_file, ensure_ascii=False, indent=2)

    return_parameter_name = os.path.basename(output_data_dir)
    
    print("\nAll rows have been processed.")
    print(f"\nOutput file name: {return_parameter_name}")


def extract_aspect_term(words_list, aspect_file_output):
    """
    抽取句子的aspect term, 并标明from, to, 和sentiment polarity.
    
    ### example:
    [
        {
            "sentence_id": 656,
            "sentence": "a delightful coming of age story .",
            "Aspect": [
            "coming of age story"
            ],
            "polarity": [
            "Positive"
            ]
        },
        {
            ...
        }
    ]
    ###
    
    """
    
    ckpts = available_checkpoints(show_ckpts=True) 
    triplet_extractor = ASTE.AspectSentimentTripletExtractor(checkpoint="english")
    output_extract_data = []
    j_flag = 0  # 记录当前sentence id
    for k in tqdm(words_list):
        current_extract_data_list = []
        current_sentence = ' '.join(k) # 将已分词的句子转换为字符串
        current_extract_output = triplet_extractor.predict(current_sentence)    # run extract
        current_dict = current_extract_output
        dict_sentence = current_dict.get('sentence', 'Null')
        dict_triplets = current_dict.get('Triplets', 'Null')
        if dict_sentence != 'Null': # 当前句子不为空时对aspect, polarity进行extract, 如果为空则置为Null
            aspects = [triplet.get('Aspect', '') for triplet in dict_triplets]
            polarity = [triplet.get('Polarity', '') for triplet in dict_triplets]
            current_data_dict = {'sentence_id': j_flag, 'sentence': dict_sentence, 'Aspect': aspects, 'polarity': polarity}
        else:
            current_data_dict = {'sentence_id': j_flag, 'sentence': 'Null'}
        j_flag = j_flag + 1
        current_extract_data_list.append(current_data_dict)
        output_extract_data.append(current_data_dict)

    with open(aspect_file_output, 'w', encoding='UTF-8') as output_file:
        json.dump(output_extract_data, output_file, ensure_ascii=False, indent=2)


def extract_headid_pos_deprel(words_list, pos_deprel_output_dir, save_or_not=True):
    """
    提取数据集中每个样本的head_id, upos, deprel.
    每个word均有一个字典, 
    [
        [
            sentence1:
            [
                {
                    word1
                },
                {
                    word2
                },
                {
                    ...
                }
            ]
        ],
        [
            sentence2:
            [
                {
                    word1
                },
                {
                    word2
                },
                {
                    ...
                }
            ]
        ]
    ]
    
    ### example:
    {
      "word_id": 5,
      "word": "entertained",
      "xpos": "VBN",
      "head_id": 23,
      "deprel": "advcl"
    },
    {
        ...
    }
    ###
    
    ### parameter:
        words_list:
            type: list
            description: 函数spilt_sentece_to_word的返回值
        pos_deprel_output_dir:
            type: str
            description: 函数处理后数据的输出路径
        save_or_not:
            type: bool
            description: 是否保存结果, if save_or_not: pos_deprel_output_dir, else: return
        return:
            type: list
            description: function result
    
    """
      
    output_sentence_list = []   # 保存word_list抽取所有句子相应成分的结果
    extract_current_sentece = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    for k in tqdm(words_list):
        current_output_list = []    # 存储当前句子的抽取结果
        current_sentence = ' '.join(k)  # 将已分词的字符串转换为正常形态
        doc = extract_current_sentece(current_sentence)
        for sent in doc.sentences:
            for word in sent.words:
                current_data_dict = {'word_id': word.id, 'word': word.text, 'xpos': word.xpos, 'head_id': word.head, 'deprel': word.deprel}
                current_output_list.append(current_data_dict)
        output_sentence_list.append(current_output_list)
        # print(output_sentence_list)
    print("has been extracted head_id, upos, deprel.")
    if save_or_not:
        with open(pos_deprel_output_dir, 'w', encoding='UTF-8') as output_file:
            json.dump(output_sentence_list, output_file, ensure_ascii=False, indent=2)
    
    return output_sentence_list


def spilt_sentece_to_word(input_file_dir, output_dir):
    """
    将句子分词, 返回结果为一个列表, 其包含多个已被分词后的句子列表
    
    ### examples:
    [
        ['smart', ',', 'provocative', 'and', 'blisteringly', 'funny', '.'],
        [...]
    ]
    ###
    
    ### parameter:
        input_file_dir: 
            type: str
            description: 经函数extract_datasets_to_build_text_and_label处理后的数据的路径
        return: 
            type: list
            description: 一个包含多个被分词后句子列表的列表
        
    """
    
    tokenizered_list = []   # 所有句子分词的结果
    with open(input_file_dir, 'r', encoding='UTF-8') as current_file:
        data_list = json.load(current_file)
        for dictionary in tqdm(data_list):
            text_value = dictionary.get("text") # 获取目标键的值
            tokens = word_tokenize(text_value) # 使用nltk对句子进行分词
            tokenizered_list.append(tokens)
        with open(output_dir, 'w', encoding='UTF-8') as output_file:
            json.dump(tokenizered_list, output_file, ensure_ascii=False, indent=2)
    return tokenizered_list

def judge_term_and_polarity(terms_file_dir, output_dir, word_list=[]):
    """
    1. 将pyabsa抽取的aspect terms, from, to清洗, 去除重复的aspect, porlarity;
    2. 匹配aspect对应的porlarity;
    3. 通过分词后的sentence list获取aspect在原句子中的位置索引
    4. 最终创建一个dict:
    ### example:
    final_list = 
    [
        {
            "sentence_id": 0,
            "aspects": [
                {
                    "term": [
                    "approach"
                    ],
                    "from": 13,
                    "to": 14,
                    "porlarity": "neutral"
                }
            ]
        }
    ]   
    ###
    
    ### parameter:
        terms_file_dir: 
            type: str
            description: 经函数extract_aspect_term处理后的数据的路径 
        output_dir: 
            type: str
            description: 经该函数处理后输出文件时的路径
        word_list: 
            type: list
            description: 经函数spilt_sentece_to_word分词后的数据
    """
    with open(terms_file_dir, 'r', encoding='UTF-8') as current_file:
        data_list = json.load(current_file)
        out_data_list = []
        count_flag = 0
        word_list_len = len(word_list)
        for k in tqdm(data_list):
            sentence_id = k.get('sentence_id')  # 读取对应的键的值
            sentence = k.get('sentence')
            if sentence != 'Null':
                aspect_list = list(dict.fromkeys(k.get('Aspect')))  # 列表去重
                polarity_list = list(dict.fromkeys(k.get('polarity')))
                current_sentence_output_dict_temp = {
                    "sentence_id": sentence_id,
                    "aspects": []
                }
                for current_index, asp_ in enumerate(aspect_list):
                    try:
                        asp_from = word_list[sentence_id].index(asp_)   # aspect_from
                        asp_to = asp_from + 1   # aspect_to
                        if len(aspect_list) > len(polarity_list):   # 当同一porlarity的aspect出现多次时取其一
                            correp_porlarity = polarity_list[0]
                        elif len(aspect_list) < len(polarity_list): # 当同一aspect存在不同porlarity时令其porlarity = neutral
                            correp_porlarity = 'neutral'
                        else:
                            correp_porlarity = polarity_list[current_index] # term corresponding porlartiy
                        temp_dict = {
                            "term":[
                                asp_
                            ],
                            "from": asp_from,
                            "to": asp_to,
                            "porlarity": correp_porlarity.lower()
                        }
                        current_sentence_output_dict_temp["aspects"].append(temp_dict)
                    except ValueError:
                        # pyabsa 有时将多个词结合起来赋予一个aspect, 此时通过nltk.tokenize将该aspect_term tokenize,
                        # 并将该aspect_term均令为同一porlarity
                        tokens = word_tokenize(asp_)
                        for tokend_index, tokend_asp_ in enumerate(tokens):
                            try:
                                asp_from = word_list[sentence_id].index(tokend_asp_)
                                asp_to = asp_from + 1
                                if len(aspect_list) > len(polarity_list):
                                    correp_porlarity = polarity_list[0]
                                elif len(aspect_list) < len(polarity_list):
                                    correp_porlarity = 'neutral'
                                else:
                                    correp_porlarity = polarity_list[current_index]
                                temp_dict = {
                                    "term":[
                                        tokend_asp_
                                    ],
                                    "from": asp_from,
                                    "to": asp_to,
                                    "porlarity": correp_porlarity.lower()
                                }
                                current_sentence_output_dict_temp["aspects"].append(temp_dict)
                            except ValueError:
                                continue
                    count_flag = count_flag + 1
            
            out_data_list.append(current_sentence_output_dict_temp)         
        
        print(f"all data: {word_list_len}, successed: {count_flag}\n")
              
        with open(output_dir, 'w', encoding='UTF-8') as out_current_file:
            print("writing the data to new files...\n")
            json.dump(out_data_list, out_current_file, ensure_ascii=False, indent=2)
            print("successed.")
                    
def find_all_duplicates(input_list):
    """
    如果列表中有重复元素, 返回存在重复元素的列表; 否则返回一个空列表, 表示没有找到重复元素.
    ### parameter:
        input_list:
            type: list
            description: 要进行判断的list
        return:
            type: list
            description: empty list or duplicates list
    """
    seen = set()
    duplicates = []
    for item in input_list:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    return duplicates

def build_final_dataset(splited_token_list, aspect_terms_file, xpos_deprel_head_id_file, output_dir):
    """
    输出最终数据.
    
    ### example: 
    [
        {
            "token": [
                "it",
                "'s",
                "played",
                "in",
                "the",
                "most",
                "straight",
                "-",
                "faced",
                "fashion",
                ",",
                "with",
                "little",
                "humor",
                "to",
                "lighten",
                "things",
                "up",
                "."
            ],
            "pos": [
                ...
            ],
            "head": [
                ...
            ],
            "deprel": [
                ...
            ],
            "aspects": [
            {
                "term": [
                "humor"
                ],
                "from": 11,
                "to": 12,
                "porlarity": "negative"
            },
            {
                ...
            }
            ]
        },
        {
            ...
        }
    ]
    ###
    
    ### parameter:
        splited_token_list: 
            type: list
            description: 经函数spilt_sentece_to_word分词后的数据
        aspect_terms_file:
            type: str
            description: 经函数judge_term_and_polarity处理后的数据的路径
        xpos_deprel_head_id_file:
            type: str
            description: 经函数extract_headid_pos_deprel处理后的数据的路径
        output_dir:
            type: str
            description: 最终数据输出路径
            
    """
    with open(aspect_terms_file, 'r', encoding='UTF-8') as current_aspect_terms_file:
        with open(xpos_deprel_head_id_file, 'r', encoding='UTF-8') as current_xpos_deprel_head_id_file:
            aspect_terms_list = json.load(current_aspect_terms_file)
            xpos_head_id_deprel_list = json.load(current_xpos_deprel_head_id_file)
            token_list = splited_token_list
            out_list = []
            out_index = 0   # 用于更新out_list中对应sentence的 aspects
            for index_, k in enumerate(aspect_terms_list): 
                aspects_term = k.get('aspects', None)   
                if aspects_term:    # 部分sentence的 aspect为空列表, 若为空则跳过
                    current_sentence_id = k.get('sentence_id', None)
                    temp_dict = {
                            "token": [],
                            "pos": [],
                            "head": [],
                            "deprel": [],
                            "aspects": []
                        }
                    for word_data in xpos_head_id_deprel_list[current_sentence_id]: # 获取当前句子每个词的相关数据
                        temp_dict["token"].append(word_data.get('word'))
                        temp_dict["pos"].append(word_data.get('xpos'))
                        temp_dict["head"].append(word_data.get('head_id'))
                        temp_dict["deprel"].append(word_data.get('deprel'))
                    out_list.append(temp_dict)
                    out_list[out_index].update({"aspects": aspects_term})
                    out_index = out_index + 1
            with open(output_dir, 'w', encoding='UTF-8') as out_current_file:
                print("writing the Final_data to new files...\n")
                json.dump(out_list, out_current_file, ensure_ascii=False, indent=2)
                print("successed.") 

def main():
    # 抽取原数据集, 获取text和对应label
    extract_datasets_to_build_text_and_label(
        input_data_dir=config['original_data_dir'],
        output_data_dir=config['text_label_data_dir']
        )
    # 将获取的text进行分词
    splited_sentence_list = spilt_sentece_to_word(
        input_file_dir=config['text_label_data_dir'],
        output_dir=config['splited_sentence_data_dir']
        )
    # 抽取句子的aspect term, 并标明from, to, 和sentiment polarity.
    extract_aspect_term(
        words_list=splited_sentence_list,
        aspect_file_output=config['extracted_aspect_term_data_dir']
        )
    # 提取数据集中每个样本的head_id, upos, deprel.
    extract_headid_pos_deprel_ = extract_headid_pos_deprel(
        words_list=splited_sentence_list,
        pos_deprel_output_dir=config['pos_deprel_output_data_dir'],
        save_or_not=True)
    # 清洗pyabsa抽取的aspect terms, 匹配对应aspect的porlarity, 获取aspect的from和to索引
    judge_term_and_polarity(
        terms_file_dir=config['extracted_aspect_term_data_dir'],
        output_dir=config['judge_term_and_polarity_out_data_dir'],
        word_list=splited_sentence_list)
    # 整合最终数据
    build_final_dataset(splited_token_list=splited_sentence_list,
                        aspect_terms_file=config['judge_term_and_polarity_out_data_dir'],
                        xpos_deprel_head_id_file=config['pos_deprel_output_data_dir'],
                        output_dir=config['final_data_out_dir'])
    
if __name__ == '__main__':
    str1 = 'F:\\1aA_Wxy_WorkStation\\Datasets\\sentiment_datasets\\build_data\\final\\'
    config = {
        "original_data_dir": str1 + 'original.jsonl',
        "text_label_data_dir": str1 + 'text_label_data.json',
        "splited_sentence_data_dir": str1 + 'splited_sentence_data.json',
        "extracted_aspect_term_data_dir": str1 + 'splited_sentence_data.json',
        "pos_deprel_output_data_dir": str1 + 'pos_deprel_output_data.json',
        "judge_term_and_polarity_out_data_dir": str1 + 'judge_term_and_polarity_out_data.json',
        "final_data_out_dir": str1 + 'final_data.json'
    }
    
    main()