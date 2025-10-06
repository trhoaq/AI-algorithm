import matplotlib.pyplot as plt
import numpy as np
import random as rand


x = rand.randint(20, 50)
learning_rate = 0.001
y = lambda x: x**2
dy = lambda x: 2*x
count = 0

X1 = [x]
Y1 = [y(x)]


while x>0.05:
    count +=1
    x -= learning_rate*dy(x)
    gradient_desent = y(x)

    X1.append(x)
    Y1.append(gradient_desent)

    if gradient_desent<0.05:
        print(f"result: {gradient_desent}\nstep: {count}")
        break

X2 = np.array(X1)
Y2 = np.array(Y1)
Count = np.array([x for x in range(0, count+1)])

plt.figure(figsize=(10, 6))
plt.plot( Count, Y2, marker="o", label="f(x)", color="red" )
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Biểu diễn gradient desent theo x")
plt.legend()
plt.grid(True)
plt.show()