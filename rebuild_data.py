import time
import json
import os
import faiss
import requests
import numpy as np
import ollama
from ollama import embed
import math
import msgpack

# 重建数据，主要的原因是因为 embedding 的时间比较久，
# 本身的代码写的有问题，先缓存 embedding 的数据到一个文件里
# 这样用 faiss 重建时，以及迁移到 c# 时，可以节约一个步骤

# 重建的方法，读取 data/LuXunWorks.json_1.json 文件
# 拆分出1w条数据，并且按照 1w 条数据 重新写入一个新的文件
# 在这个过程中，调用 ollama 的 embed 方法，将数据转换为 embedding

# 读取文件
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_files_from_dir(input_dir_path):
    file_paths = []
    for root, dirs, files in os.walk(input_dir_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

"""创建向量数据库"""
start_time = time.time()
batch_size = 5000
field_name = "chunk"
input_dir_path = 'data'
input_file_paths = get_files_from_dir(input_dir_path)
for input_file_path in input_file_paths:
    data_list = []
    
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
    # 按照1w条数据为一个批次，分别生成embedding
    total_queries = len(data_list)
    num_batches = math.ceil(total_queries / batch_size)
    print(f"共有 {total_queries} 条数据，将分为 {num_batches} 批次入库。")
    for batch_num in range(num_batches):
        # new_file_path = input_file_path.replace('.json', f'_embedding_{batch_num}.json')
        msgpack_file_path = input_file_path.replace('.json', f'_embedding_{batch_num}.msgpack')        
        if os.path.exists(msgpack_file_path):
            print(f"第 {batch_num + 1} 批次的嵌入已经生成，跳过。")
            continue
        
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size, total_queries)
        batch_datas = data_list[start_index:end_index]
        
        queryBatchSize = 50
        
        def split_batch_datas(batch_datas, queryBatchSize=50):
            return [batch_datas[i:i + queryBatchSize] for i in range(0, len(batch_datas), queryBatchSize)]

        split_batches = split_batch_datas(batch_datas, queryBatchSize)

        for index,split_batch in enumerate(split_batches):
            batch_queries = [data['chunk'] for data in split_batch]
                        
            # 向量化当前批次的查询
            response = embed(model='bge-m3', input=batch_queries)
            vectors = response.get('embeddings', [])
            # 回写到 batch_datas 里的 embedding 字段
            for i, data in enumerate(split_batch):
                data['chunk_embedding'] = vectors[i]
            
            batch_queries = [data['window'] for data in split_batch]
                        
            # 向量化当前批次的查询
            response = embed(model='bge-m3', input=batch_queries)
            vectors = response.get('embeddings', [])
            # 回写到 batch_datas 里的 embedding 字段
            for i, data in enumerate(split_batch):
                data['window_embedding'] = vectors[i]
                
            print(f"第 {batch_num + 1} 批次的嵌入生成成功，共 {len(vectors)} 条数据。 剩余: {len(split_batches) - index} 批次")
                
        # 写入到新的文件里
        # with open(new_file_path, 'w', encoding='utf-8') as file:
        #     json.dump(batch_datas, file, ensure_ascii=False, indent=4)
        

        with open(msgpack_file_path, 'wb') as file:
            packed = msgpack.packb(batch_datas, use_bin_type=True)
            file.write(packed)
            
        print(f"第 {batch_num + 1} 批次的嵌入生成成功，共 {len(vectors)} 条数据。")
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"数据入库耗时：{elapsed_time:.2f} 秒")
# 数据入库耗时：22449.67 秒

