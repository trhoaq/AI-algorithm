import matplotlib.pyplot as plt
import numpy as np
import random as rand


# x = rand.randint(20, 50)
x= 5
learning_rate = 0.01
y = lambda x: x**2 +5*np.sin(x)
dy = lambda x: 2*x+5*np.cos(x)
count = 0

X1 = [x]
Y1 = [y(x)]


while abs(dy(x)) > 1e-3 :
    count +=1
    x -= learning_rate*dy(x)
    gradient_descent = y(x)

    X1.append(x)
    Y1.append(gradient_descent)

print(f"Result: f(x) = {gradient_descent:.4f} tại x = {x:.4f}, bước = {count}")

X2 = np.array(X1)
Y2 = np.array(Y1)
step = np.arange(len(Y2))

plt.figure(figsize=(10, 6))
plt.plot( step, Y2, marker="o", label="f(x)", color="red" )
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Biểu diễn gradient desent theo x")
plt.legend()
plt.grid(True)
plt.show()