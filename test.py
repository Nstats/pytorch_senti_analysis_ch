import matplotlib.pyplot as plt
x1 = ['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08',      '2017-09', '2017-10', '2017-11', '2017-12']
y1 = [86, 85, 84, 80, 75, 70, 70, 74, 78, 70, 74, 80]
plt.figure(figsize=(16, 4))
plt.title("my weight")
plt.plot(x1, y1, label='weight changes', linewidth=3, color='r', marker='o', markerfacecolor='blue', markersize=2) # 横坐标描述
plt.xlabel('month')
plt.ylabel('weight')
for a, b in zip(x1, y1):
    plt.text(a, b, b, fontsize=15)
plt.legend()
plt.show()
