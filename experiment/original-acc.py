import matplotlib.pyplot as plt

# 层数（Iteration）
layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 对应的精度值
accuracy = [
    0.8175287356321839,
    0.8577586206896551,
    0.8433908045977011,
    0.8721264367816092,
    0.8879310344827587,
    0.8663793103448276,
    0.8951149425287356,
    0.8793103448275862,
    0.8864942528735632,
    0.8419540229885057
]

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(layers, accuracy, marker='o', color='b', label='Accuracy')

# 添加标题和标签
plt.title('Accuracy per Layer')
plt.xlabel('Layer')
plt.ylabel('Accuracy')

# 显示网格
plt.grid(True)

# 显示图例
plt.legend()

# 展示图形
plt.show()
