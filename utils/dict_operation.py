

def create_new_dict_based_on_values(original_dict):
    new_dict = {}
    value_seen = set()  # 用于记录已经添加过的新字典的值

    for key, value in original_dict.items():
        if value not in value_seen:
            new_dict[value] = key
            value_seen.add(value)
    
    return new_dict

def replace_dict_keys(d, key_mapping, subkey_mapping):
    new_dict = {}
    for k, v in d.items():
        new_key = key_mapping.get(k, k)  # 获取映射值，如果不存在则使用原值
        new_dict[new_key] = {subkey_mapping.get(sk, sk): sv for sk, sv in v.items()}  # 修改子字典中的子键
    return new_dict
