# Lecture-09: CNN Principles how computer recognize images.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

# 卷积
def conv(image, filters, stride=1):
    # 遍历滤波器 filters
    conv_results = [conv_(image, f, stride) for f in filters] # 列表生成式

    return np.array(conv_results) # 转换成 numpy 数组

# 真正的卷积函数
# 对一个滤波器 filter
def conv_(image, filter, stride=1):
    # 5 * 5, 3 * 3: 3 * 3(stride=1), 2 * 2(stride=2)
    # 做了 Padding 吗？？？
    height = image.shape[0] - filter.shape[0] + 1
    width = image.shape[1] - filter.shape[1] + 1

    conv_result = np.zeros(shape=(height // stride, width // stride)) # 得到卷积后的图像尺寸

    for h in range(0, height, stride):
        for w in range(0, width, stride):
            # 截取图像的对应区域。
            window = image[h: h + filter.shape[0], w: w + filter.shape[1], :]
            # 可以验证一下，待验证。

            # ic(window.shape)
            # ic(filter.shape)

            # window和滤波器对应位置相乘后相加。
            conv_result[h][w] = np.sum(np.multiply(window, filter)) # 广播
            # ic(conv_result[h][w].shape)
            
            # np.max(window) => max pooling
            # np.mean(window) => mean pooling

    return conv_result


if __name__ == '__main__':
    dog = Image.open('dog.jpg') # 打开图像

    dog = np.array(dog) # 把 PIL 数据转化成 numpy 数组

    # plt.imshow(dog) # Matplotlib 显示图像

    filter_ = np.array([ # 滤波器
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])

    filter_2 = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    ic(dog.shape) # (250, 400, 3) (y, x, channel)

    # 滤波器列表。
    dog_convs = conv(dog, [filter_, filter_2])

    fig, axes = plt.subplots(1, 2)
    ic(axes)

    # 别再纠结这两个滤波器代表什么意思了，要了解的话去看图像处理的课。
    axes[0].imshow(dog_convs[0])
    axes[1].imshow(dog_convs[1])
    plt.show()

    ic(dog_convs.shape) # (2, 248, 398)，原因在列表生成式

    flatten = dog_convs.reshape(1, -1) # (1, 2 * 248 * 398)
    ic(flatten.shape)

    # FC层输入是铺平的向量长度，输出为5维。

    outputs = np.matmul(flatten, np.random.random(size=(flatten.shape[1], 5)))

    print(outputs)


