# data: https://download.pytorch.org/tutorial/data.zip
import io
import os
import unicodedata
import string
import glob
from icecream import ic
import torch
import random

# alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def load_data():
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    def find_files(path):
        # ic(glob.glob(path)) # data文件夹开始的
        return glob.glob(path)

    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n') # 读入文件的每行
        # ic(lines)
        # ic([unicode_to_ascii(line) for line in lines])
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files('data/names/*.txt'):
        # ic(os.path.basename(filename)) # 文件名
        # ic(os.path.splitext(os.path.basename(filename))) # 获得文件名主体和扩展名
        category = os.path.splitext(os.path.basename(filename))[0]
        # ic(category)
        all_categories.append(category) # 对于一个文件添加
        # ic(all_categories) # 字符串类型

        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories


"""
To represent a single letter, we use a “one-hot vector” of 
size <1 x n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.
To make a word we join a bunch of those into a
2D matrix <line_length x 1 x n_letters>.
That extra 1 dimension is because PyTorch assumes
everything is in batches - we’re just using a batch size of 1 here.
字母->向量 <1 x n_letters>
单词->矩阵 <line_length x 1 x n_letters>, batch size of 1
文档->张量
"""


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
# 一行是一个单词，是一个序列
def line_to_tensor(line): # 转换成张量
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def random_training_example(category_lines, all_categories): # 随机取一个训练样本
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]

    category = random_choice(all_categories)
    line = random_choice(category_lines[category]) #从该category中取
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    ic(category_tensor)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


if __name__ == '__main__':
    ic(ALL_LETTERS)
    ic(N_LETTERS) # 57
    
    print(unicode_to_ascii('Ślusàrski'))

    category_lines, all_categories = load_data()
    ic(category_lines['Italian'][:5])

    ic(letter_to_tensor('J'))  # [1, 57]
    ic(line_to_tensor('Jones').size())  # [5, 1, 57]