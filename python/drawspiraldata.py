import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./python/spiral_data.csv")

plt.figure(figsize=(6,6))

# scatter plot, color by class
for class_num in df['class'].unique():
    subset = df[df['class'] == class_num]
    plt.scatter(subset['x'], subset['y'], label=f'Class {class_num}', alpha=0.7)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Spiral Data')
plt.legend()
plt.axis('equal')  # ensures circles look circular
plt.show()