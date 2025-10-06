"""
Data processing utilities.
"""
import numpy as np

def flatten_dict(d, parent_key='', sep='.'):
    """
    将一个嵌套字典展平。
    例如: {'a': {'b': 1}} -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d, sep='.'):
    """
    将一个扁平的、用分隔符表示层级的字典恢复成嵌套字典。
    例如: {'a.b': 1} -> {'a': {'b': 1}}
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_nested = result
        for part in parts[:-1]:
            if part not in d_nested:
                d_nested[part] = {}
            d_nested = d_nested[part]

        # 1. 首先检查值是否是 NumPy 数组
        if isinstance(value, np.ndarray):
            # 2. 如果是，再检查它是否是 0 维数组 (单个值)
            if value.shape == ():
                # 如果是，用 .item() 提取出 Python 纯量值
                d_nested[parts[-1]] = value.item()
            else:
                # 否则，它是一个正常的数组，直接赋值
                d_nested[parts[-1]] = value
        else:
            # 3. 如果不是 NumPy 数组 (例如 int, float, str, bool), 直接赋值
            d_nested[parts[-1]] = value
            
    return result


if __name__ == "__main__":
    # 测试代码
    nested_dict = {'a': {'b': 1, 'c': 2}, 'd': 3}
    flat_dict = flatten_dict(nested_dict)
    print("Flattened:", flat_dict)

    restored_dict = unflatten_dict(flat_dict)
    print("Restored:", restored_dict)