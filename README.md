Deep Learning Experiment — Fashion-MNIST with a 10-Layer MLP
I ran 3 experiments comparing optimizers, activation functions, and batch normalization on Fashion-MNIST using TensorFlow/Keras:
⚡ Optimizers: Adam & RMSProp hit 88% accuracy; plain SGD stalled at 83%
🔥 ReLU vs Sigmoid: Sigmoid collapsed to ~10% in a 10-layer net — vanishing gradients in action
📊 Batch Norm: +5% boost for SGD, marginal gains for Adam
