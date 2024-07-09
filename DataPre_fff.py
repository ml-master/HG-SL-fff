import json
import re
from transformers import BertTokenizer, BertModel
import torch
import shutil
import pickle
import torch.nn as nn

# 寻找不同的数据
def SearchDiffD():
    # 读取txt文件
    with open('data\gos\gos_news_list.txt', 'r', encoding='utf-8') as txt_file:
        txt_lines = txt_file.read().splitlines()

    # 读取json文件
    with open('MLData\megafake-1_style_based_fake.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    i = 0
    error = []

    for key, value in json_data.items():
        match = re.search(r'\d+', key)
        if match:
            new_key = f"gossipcop-{match.group()}"
        print(new_key)
        if new_key in txt_lines:
            i += 1
        else:
            error.append(new_key)

    print(error)
    print(i)

# 寻找相同的数据并保存
def SearchSaveSameD():

    NewData = {}

    with open('data\gos\gos_news_list.txt', 'r', encoding='utf-8') as txt_file:
        txt_lines = txt_file.read().splitlines()

    # 读取json文件
    with open('MLData\megafake-1_style_based_fake.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    for key, value in json_data.items():
        match = re.search(r'\d+', key)
        if match:
            new_key = f"gossipcop-{match.group()}"
        if new_key in txt_lines:
            tempData = {new_key:value['generated_text']}
            NewData.update(tempData)
        print(NewData)

    with open('MLData\matched_data.json', 'w', encoding='utf-8') as output_file:
        json.dump(NewData, output_file, ensure_ascii=False, indent=4)

# Bert处理文本
def TransformerText1(text):
    target_dim = 500
    # print("flag1")
    # 初始化BERT模型和分词器
    model_name = 'bert-base-uncased'  # 对于中文文本，我们使用中文BERT模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 编写你的文本
    # text = "今天天气不错"

    # 对文本进行编码，添加必要的BERT tokens
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # print("flag2")
    # 获取模型的输出，包括所有隐藏层的输出和池化层的输出
    with torch.no_grad():  # 在不需要计算梯度的情况下执行，以减少计算资源消耗
        outputs = model(**encoded_input)

    # 获取最后一层的隐藏状态（CLS token的输出）
    # last_hidden_state = outputs.last_hidden_state

    
    # 获取池化层的输出
    pooler_output = outputs.pooler_output

    # 定义线性层以调整特征向量的维度
    linear_layer = nn.Linear(pooler_output.size(-1), target_dim)

    

    # 通常我们使用CLS token的输出作为文本的特征向量
    # 它是一个可以通过target_dim = 128条件大小的向量
    feature_vector = linear_layer(pooler_output)

    # 将浮点数向量转换为整数向量
    integer_feature_vector = (feature_vector * 10000).int()

    # print(integer_feature_vector)  # 打印特征向量
    return integer_feature_vector



# 利用Bert处理文本方法处理文本
def TransformerText2():
    source_file = 'MLData\matched_data.json'
    destination_file = 'MLData\matched_data_copy.json'
    file_path1 = 'MLData/TextVector.pkl'

    shutil.copyfile(source_file, destination_file)

    with open(destination_file, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    i = 0
    textVectorSet= {}
    for key, value in json_data.items():
        tempVector = TransformerText1(value)
        textVectorSet[key] = tempVector
        i += 1
        print("i:" + str(i))
        # print(i)
        '''
        print(tempVector)
        
        break
        if i == 3:
           break
        '''
    with open(file_path1, 'wb') as output_file:
        pickle.dump(textVectorSet, output_file)
    



    

# 查看生成的文件
def ViewData():
    # 定义文件路径
    file_path1 = 'MLData/TextVector.pkl'
    file_path2 = 'data/gos/Processed/news_centered.pickle'
    file_path3 = 'data/gos/news_mapping.pickle'

    # 加载pickle文件
    with open(file_path1, 'rb') as file:  # 确保使用 'rb' 模式
        data = pickle.load(file)
    
    """把数据保存在文件中查看
    with open("000test_data.txt", 'w') as file:
        for item in test_data:
        # 将元素转换为字符串并写入文件，后面加上换行符
            file.write(str(item) + '\n')
    """
    

    # 打印加载的数据
    print(len(data))
    print(type(data))
    print(data["gossipcop-905229"])

# 利用map将标签和向量对应
def id2Vector():
    file_path1 = 'MLData/TextVector.pkl'
    file_path2 = 'data/gos/news_mapping.pickle'
    file_path3 = 'data/gos/id2Vector.pickle'

    with open(file_path1, 'rb') as file:
        vector_data = pickle.load(file)
    
    with open(file_path2, 'rb') as file:
        map_data = pickle.load(file)

    id2vector = {}
    # print(map_data)

    for key in map_data:
        if key in vector_data:
            print(map_data[key])
            id2vector[map_data[key]] = vector_data[key]
    
    print(id2vector[100])
    with open(file_path3, 'wb') as output_file:
        pickle.dump(id2vector, output_file)

def gen_new_gos_news_list():
    with open('MLData\matched_data.json', 'r', encoding='utf-8') as f:
        matched_data = json.load(f)

    with open("del_user.pkl", 'rb') as file:
        del_user = pickle.load(file)
        print('读取的列表内容：')
        print(del_user)
        print("jieshu")
        
    gos_news_list_dict = {}
    with open('data/gos/news_centered_data.txt', 'r', encoding='utf-8') as txt_file:
            # 逐行读取
            for line in txt_file:
                # 去除行尾的换行符
                line = line.strip()
                # 以第一个逗号为分界点分割键和值
                key, value = line.split(',', 1)
                # 将键值对添加到字典中
                
                x = value.split()
                f = 1
                for y in x:
                    if y in del_user:
                        f = 0
                if f:
                    gos_news_list_dict[key] = value
                # print(x)
    
    # 创建一个新字典来存储结果
    new_dict = {}
    for key in gos_news_list_dict:
        # 检查键是否在原始字典中
        if key in matched_data:
            # 如果是，将键和值添加到新字典中
            
            new_dict[key] = gos_news_list_dict[key]
    
    print(new_dict["gossipcop-945700"])

    # 指定要写入的文件名
    filename = 'output.txt'

    # 打开文件，如果文件不存在将会被创建
    with open(filename, 'w', encoding='utf-8') as file:
        # 遍历字典中的每个键值对
        for key, value in new_dict.items():
            # 将键和值连接成一个字符串，中间用逗号分隔
            line = f"{key},{value}\n"
            # 写入文件
            file.write(line)

    print(f"数据已写入到 {filename} 文件中。")
    
    
    
def from_matched_data_extract_name():
    with open('MLData\matched_data.json', 'r', encoding='utf-8') as f:
        matched_data = json.load(f)
    # 指定要写入的文件名
    filename = 'keys.txt'

    # 打开文件，如果文件不存在将会被创建
    with open(filename, 'w', encoding='utf-8') as file:
        # 遍历字典中的每个键
        for key in matched_data.keys():
            # 将键写入文件，后面跟一个换行符
            file.write(f"{key}\n")

    print(f"所有键已保存到 {filename} 文件中。")

def gen_new_label():
    # 创建一个空列表来存储文件内容
    lines_list = []

    # 指定要读取的文件名
    
    filename1 = 'data/gos/general/gos_news_list.txt'

    # 打开文件
    with open(filename1, 'r', encoding='utf-8') as file:
        # 逐行读取
        for line in file:
            # 去除行尾的换行符
            line = line.strip()
            # 将行添加到列表中
            lines_list.append(line)

    # 打印列表内容
    # for line in lines_list:
        # print(line)

    print(f"文件内容已读取并存入列表，共 {len(lines_list)} 行。")
    # 创建一个空字典来存储文本和数字
    data_dict = {}

    filename2 = 'data/gos/label.txt'
    # 打开文件
    with open(filename2, 'r', encoding='utf-8') as file:
        # 逐行读取
        for line in file:
            # 去除行尾的换行符
            line = line.strip()
            # 以空格为分隔符分割文本和数字
            parts = line.split()
            # 确保分割后有两个部分
            if len(parts) == 2:
                text = parts[0]
                number = parts[1]
                # 将文本和数字存入字典
                data_dict[text] = number

    # 创建一个新字典来存储匹配的键值对
    new_dict = {key: data_dict[key] for key in lines_list if key in data_dict}

    # 指定要写入的文件名
    filename = 'output.txt'

    # 打开文件，如果文件不存在将会被创建
    with open(filename, 'w', encoding='utf-8') as file:
        # 遍历新字典中的每个键值对
        for key, value in new_dict.items():
            # 将键和值连接成一个字符串，中间用空格分隔
            line = f"{key} {value}\n"
            # 写入文件
            file.write(line)

    print(f"匹配的键值对已保存到 {filename} 文件中。")

def gen_new_user_centered_data():
    # 创建一个空列表来存储文件内容
    lines_list = []

    # 指定要读取的文件名
    
    filename1 = 'data/gen/gen_news_list.txt'

    # 打开文件
    with open(filename1, 'r', encoding='utf-8') as file:
        # 逐行读取
        for line in file:
            # 去除行尾的换行符
            line = line.strip()
            # 将行添加到列表中
            lines_list.append(line)

    # 打印列表内容
    # for line in lines_list:
        # print(line)

    print(f"文件内容已读取并存入列表，共 {len(lines_list)} 行。")

    # 指定要读取的文件名
    input_filename = 'data/gos/user_centered_data.txt'
    # 指定要写入的文件名
    output_filename = 'output.txt'

    del_user = []
    # 打开输入文件
    with open(input_filename, 'r', encoding='utf-8') as input_file:
        # 打开输出文件，如果文件不存在将会被创建
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            # 逐行读取输入文件
            for line in input_file:
                # 去除行尾的换行符
                line = line.strip()
                # 分割行内容
                parts = line.split(',')
                # parts = parts.split()
                # 检查第二个部分是否在列表中
                key = parts[1].split(' ', 1)[0]
                # print(key)
                if key in lines_list:
                    # 如果是，将这行内容写入输出文件
                    output_file.write(line + '\n')
                else:
                    del_user.append(key)

    
    with open("1del_user.pkl", 'wb') as file:
        pickle.dump(del_user, file)
        print(f'列表已存储到 {"del_user.pkl"}')
    
    print(f"包含指定内容的行已保存到 {output_filename} 文件中。")
    

class test():
    pass

if __name__ == "__main__": 
    # SearchDiffD()
    # SearchSaveSameD()
    # TransformerText2()
    # ViewData()
    # id2Vector()
    # test()
    gen_new_gos_news_list()
    # from_matched_data_extract_name()
    # gen_new_user_centered_data()