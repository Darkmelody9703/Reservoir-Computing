import numpy as np
import matplotlib.pyplot as plt

with open('2022.12.05-01-33-25.log', 'r') as f:
    log = f.readlines()

train_acc, test_acc, train_loss, test_loss = [], [], [], []
for i in log:
    # acc.append(float(i.split(',')[0][-6:]))
    # loss.append(float(i.split(',')[1].split('\') ')[1].split('\n')[0]))
    train_acc.append(float(i.split(',')[0]))
    test_acc.append(float(i.split(',')[1]))
    train_loss.append(float(i.split(',')[2]))
    test_loss.append(float(i.split(',')[3]))
plt.subplot(121)
plt.plot(train_acc)
plt.plot(test_acc)
# plt.axis([-5, 505, -0.05, 1.05])
plt.grid()

plt.subplot(122)
plt.plot(train_loss)
plt.plot(test_loss)
# plt.axis([-5, 505, -0.05, 3])
plt.grid()
plt.show()
