import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()
x = np.arange(10000)
y = np.random.standard_normal(10000)
plt.plot(x, y, label='loss_evar', linewidth=1, color='b', marker='o', markerfacecolor='green', markersize=2)
plt.savefig('./test.jpg')
print('time_used=', time.time()-start_time)
