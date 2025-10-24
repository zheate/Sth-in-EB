# pip install tensorflow
import numpy as np
import tensorflow as tf

# 可复现
np.random.seed(0)
tf.random.set_seed(0)

# 1) 造点数据：y = 2x + 0.5 + 噪声
X = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)
y = 2 * X + 0.5 + 0.1 * np.random.randn(*X.shape).astype(np.float32)

# 2) 最小 MLP：Input -> Dense(8, ReLU) -> Dense(1)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='mse'
)

# 3) 训练
hist = model.fit(X, y, epochs=300, batch_size=32, verbose=0)
print(f"final loss: {hist.history['loss'][-1]:.4f}")

# 4) 预测
x_test = np.array([[-0.5], [0.0], [0.5]], dtype=np.float32)
y_pred = model.predict(x_test, verbose=0)
for x_i, y_i in zip(x_test.ravel(), y_pred.ravel()):
    print(f"x={x_i:+.2f} -> y_pred={y_i:+.3f}")
