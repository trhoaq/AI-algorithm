import matplotlib.pyplot as plt
import numpy as np
import random as rand


x = rand.randint(5, 10)
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

x_plot = np.linspace(-5, 5, 400)
y_plot = y(x_plot)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))  

axs[0].plot(step, Y2, marker="o", color="red", label="f(x)")
axs[0].set_xlabel("Iteration (Step)")
axs[0].set_ylabel("f(x)")
axs[0].set_title("Gradient Descent theo Step")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(x_plot, y_plot, label='f(x)')
axs[1].scatter(X1, Y1, color='red', label='Steps', zorder=5)
axs[1].plot(X1, Y1, '--', color='red', alpha=0.6)
axs[1].set_title("Gradient Descent trên đồ thị f(x)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("f(x)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()