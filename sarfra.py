

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

print("TensorFlow version:", tf.__version__)
print("=" * 60)


print("Loading Fashion-MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values from [0, 255] → [0.0, 1.0]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

# Flatten 28x28 images into 784-dimensional vectors
X_train = X_train.reshape(-1, 784)   # shape: (60000, 784)
X_test  = X_test.reshape(-1, 784)    # shape: (10000, 784)

# One-Hot Encode labels
# Example: class 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# This is needed for categorical_crossentropy loss
y_train_ohe = to_categorical(y_train, 10)
y_test_ohe  = to_categorical(y_test,  10)

print(f"Training data shape : {X_train.shape}")
print(f"Test data shape     : {X_test.shape}")
print(f"Training labels     : {y_train_ohe.shape}")

# Class names for display
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Show sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Fashion-MNIST Sample Images", fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(class_names[y_train[i]])
    ax.axis('off')
plt.tight_layout()
plt.savefig("sample_images.png", dpi=100, bbox_inches='tight')
plt.show()
print("Sample images saved as 'sample_images.png'")
print()

# ============================================================
# SECTION 3: BUILD THE DEEP MLP MODEL
# ============================================

def build_model(activation='relu', use_batchnorm=False, optimizer='adam'):
   
    model = keras.Sequential(name=f"MLP_{activation}_BN{use_batchnorm}")

    # Input Layer
    model.add(layers.Input(shape=(784,)))

    # 10 Hidden Layers (each with 256 neurons)
    for i in range(10):
        model.add(layers.Dense(256))           # Fully connected layer

        if use_batchnorm:
            # Batch Normalization: normalizes activations before activation function
            # Applied BEFORE activation for better gradient flow
            model.add(layers.BatchNormalization())

        # Activation Function
        if activation == 'relu':
            model.add(layers.Activation('relu'))
        else:
            model.add(layers.Activation('sigmoid'))

        model.add(layers.Dropout(0.2))         # Dropout: randomly turns off 20% neurons
                                                # to prevent overfitting

    # Output Layer: 10 neurons (one per class) + Softmax
    # Softmax converts raw scores → probabilities that sum to 1
    model.add(layers.Dense(10, activation='softmax'))

    # Compile: specify optimizer, loss function, and metric
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',   # Best loss for multi-class classification
        metrics=['accuracy']
    )

    return model

# Print model summary to understand the architecture
sample_model = build_model()
print("\nMODEL ARCHITECTURE (10 hidden layers, ReLU, no BatchNorm):")
print("=" * 60)
sample_model.summary()
print()

# ============================================================
# SECTION 4: COMPARE 4 OPTIMIZERS
# ============================================================

# Define the 4 optimizers
optimizers_config = {
    "SGD":           keras.optimizers.SGD(learning_rate=0.01),
    "SGD+Momentum":  keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSProp":       keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam":          keras.optimizers.Adam(learning_rate=0.001),
}

EPOCHS    = 20      # Number of times we go through the entire training set
BATCH_SIZE = 128    # Number of samples per gradient update

histories_optimizer = {}   # Store training history (loss, accuracy per epoch)
convergence_table   = {}   # Store time-to-convergence

TARGET_ACCURACY = 0.85     # We define "converged" as reaching 85% accuracy

print("=" * 60)
print("TRAINING WITH 4 DIFFERENT OPTIMIZERS")
print("(Each model: 10 hidden layers, ReLU, no BatchNorm)")
print("=" * 60)

for opt_name, opt in optimizers_config.items():
    print(f"\n[{opt_name}] Training started...")

    # Build a fresh model for each optimizer
    model = build_model(activation='relu', use_batchnorm=False, optimizer=opt)

    start_time = time.time()

    history = model.fit(
        X_train, y_train_ohe,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,   # Use 10% of training data for validation
        verbose=0                # Suppress per-epoch output for cleaner display
    )

    elapsed = time.time() - start_time

    histories_optimizer[opt_name] = history.history

    # Find which epoch first crossed TARGET_ACCURACY
    val_acc = history.history['val_accuracy']
    converge_epoch = next((i+1 for i, a in enumerate(val_acc) if a >= TARGET_ACCURACY), None)
    if converge_epoch is None:
        converge_epoch = f">  {EPOCHS}"

    convergence_table[opt_name] = {
        'converge_epoch': converge_epoch,
        'final_val_acc' : round(max(val_acc) * 100, 2),
        'train_time_sec': round(elapsed, 1)
    }

    print(f"  [{opt_name}] Done! Final Val Accuracy: {max(val_acc)*100:.2f}% | Time: {elapsed:.1f}s")

# ============================================================
# PLOT 1: Optimizer Comparison — Loss and Accuracy Curves
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Optimizer Comparison on Fashion-MNIST\n(10-Layer MLP, ReLU, No BatchNorm)",
             fontsize=14, fontweight='bold')

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for (opt_name, hist), color in zip(histories_optimizer.items(), colors):
    axes[0].plot(hist['val_loss'],     label=opt_name, color=color, linewidth=2)
    axes[1].plot(hist['val_accuracy'], label=opt_name, color=color, linewidth=2)

axes[0].set_title("Validation Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Validation Accuracy per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].axhline(y=TARGET_ACCURACY, color='gray', linestyle='--',
                label=f'Target ({TARGET_ACCURACY*100:.0f}%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("optimizer_comparison.png", dpi=120, bbox_inches='tight')
plt.show()
print("\nPlot saved as 'optimizer_comparison.png'")

# ============================================================
# PRINT CONVERGENCE TABLE
# ============================================================
print("\n" + "=" * 65)
print("  CONVERGENCE TABLE (Target Accuracy: 85%)")
print("=" * 65)
print(f"  {'Optimizer':<18} {'Converge Epoch':<20} {'Final Val Acc':<20} {'Time (s)'}")
print("-" * 65)
for opt_name, info in convergence_table.items():
    print(f"  {opt_name:<18} {str(info['converge_epoch']):<20} {info['final_val_acc']:<20} {info['train_time_sec']}")
print("=" * 65)

# ============================================================
# SECTION 5: VANISHING GRADIENT — ReLU vs Sigmoid
# ============================================================


print("\n" + "=" * 60)
print("VANISHING GRADIENT: ReLU vs Sigmoid (10-Layer MLP)")
print("=" * 60)

histories_activation = {}

for activation in ['relu', 'sigmoid']:
    print(f"\nTraining with {activation.upper()} activation...")

    model = build_model(activation=activation, use_batchnorm=False,
                        optimizer=keras.optimizers.Adam(learning_rate=0.001))

    history = model.fit(
        X_train, y_train_ohe,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0
    )

    histories_activation[activation] = history.history
    final_acc = max(history.history['val_accuracy']) * 100
    print(f"  {activation.upper()} Final Val Accuracy: {final_acc:.2f}%")

# Plot ReLU vs Sigmoid
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Vanishing Gradient: ReLU vs Sigmoid\n(10-Layer MLP, Adam Optimizer)",
             fontsize=14, fontweight='bold')

axes[0].plot(histories_activation['relu']['val_loss'],    color='#2ecc71',
             linewidth=2, label='ReLU')
axes[0].plot(histories_activation['sigmoid']['val_loss'], color='#e74c3c',
             linewidth=2, label='Sigmoid')
axes[0].set_title("Validation Loss\n(Lower is better → ReLU wins)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(histories_activation['relu']['val_accuracy'],    color='#2ecc71',
             linewidth=2, label='ReLU')
axes[1].plot(histories_activation['sigmoid']['val_accuracy'], color='#e74c3c',
             linewidth=2, label='Sigmoid')
axes[1].set_title("Validation Accuracy\n(Higher is better → ReLU wins)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Annotate the gap
relu_final    = max(histories_activation['relu']['val_accuracy']) * 100
sigmoid_final = max(histories_activation['sigmoid']['val_accuracy']) * 100
axes[1].annotate(f"ReLU: {relu_final:.1f}%",
                 xy=(EPOCHS-1, histories_activation['relu']['val_accuracy'][-1]),
                 xytext=(EPOCHS-6, relu_final/100 - 0.05),
                 color='#2ecc71', fontweight='bold')
axes[1].annotate(f"Sigmoid: {sigmoid_final:.1f}%",
                 xy=(EPOCHS-1, histories_activation['sigmoid']['val_accuracy'][-1]),
                 xytext=(EPOCHS-6, sigmoid_final/100 - 0.1),
                 color='#e74c3c', fontweight='bold')

plt.tight_layout()
plt.savefig("vanishing_gradient.png", dpi=120, bbox_inches='tight')
plt.show()
print("\nPlot saved as 'vanishing_gradient.png'")

# ============================================================
# SECTION 6: BATCH NORMALIZATION EFFECT
# ============================================================


print("\n" + "=" * 60)
print("BATCH NORMALIZATION EFFECT on each Optimizer")
print("=" * 60)

histories_bn = {}

for opt_name, opt_class in [
    ("SGD",          lambda: keras.optimizers.SGD(learning_rate=0.01)),
    ("SGD+Momentum", lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
    ("RMSProp",      lambda: keras.optimizers.RMSprop(learning_rate=0.001)),
    ("Adam",         lambda: keras.optimizers.Adam(learning_rate=0.001)),
]:
    for use_bn in [False, True]:
        label = f"{opt_name} {'+ BN' if use_bn else '(no BN)'}"
        print(f"Training: {label}...")

        model = build_model(activation='relu', use_batchnorm=use_bn,
                            optimizer=opt_class())

        history = model.fit(
            X_train, y_train_ohe,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=0
        )

        histories_bn[label] = history.history
        final_acc = max(history.history['val_accuracy']) * 100
        print(f"  → Final Val Accuracy: {final_acc:.2f}%")

# Plot BN comparison for each optimizer
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Effect of Batch Normalization on Each Optimizer\n(10-Layer MLP, ReLU)",
             fontsize=14, fontweight='bold')

opt_names_list = ["SGD", "SGD+Momentum", "RMSProp", "Adam"]

for col, opt_name in enumerate(opt_names_list):
    no_bn_hist = histories_bn[f"{opt_name} (no BN)"]
    bn_hist    = histories_bn[f"{opt_name} + BN"]

    # Loss plot (top row)
    axes[0, col].plot(no_bn_hist['val_loss'], color='#e74c3c',
                      linewidth=2, label='No BN')
    axes[0, col].plot(bn_hist['val_loss'],    color='#3498db',
                      linewidth=2, label='With BN')
    axes[0, col].set_title(f"{opt_name}\nValidation Loss")
    axes[0, col].set_xlabel("Epoch")
    axes[0, col].set_ylabel("Loss")
    axes[0, col].legend()
    axes[0, col].grid(True, alpha=0.3)

    # Accuracy plot (bottom row)
    axes[1, col].plot(no_bn_hist['val_accuracy'], color='#e74c3c',
                      linewidth=2, label='No BN')
    axes[1, col].plot(bn_hist['val_accuracy'],    color='#3498db',
                      linewidth=2, label='With BN')
    axes[1, col].set_title(f"{opt_name}\nValidation Accuracy")
    axes[1, col].set_xlabel("Epoch")
    axes[1, col].set_ylabel("Accuracy")
    axes[1, col].legend()
    axes[1, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("batchnorm_effect.png", dpi=120, bbox_inches='tight')
plt.show()
print("\nPlot saved as 'batchnorm_effect.png'")

# ============================================================
# SECTION 7: FINAL SUMMARY TABLE
# ============================================================
print("\n" + "=" * 75)
print("  FINAL SUMMARY — All Experiments")
print("=" * 75)
print(f"  {'Experiment':<35} {'Final Val Acc (%)':<20} {'Note'}")
print("-" * 75)

for opt_name in ["SGD", "SGD+Momentum", "RMSProp", "Adam"]:
    acc = max(histories_optimizer[opt_name]['val_accuracy']) * 100
    print(f"  {opt_name + ' (no BN, ReLU)':<35} {acc:<20.2f} Optimizer comparison")

print()
relu_acc    = max(histories_activation['relu']['val_accuracy']) * 100
sigmoid_acc = max(histories_activation['sigmoid']['val_accuracy']) * 100
print(f"  {'ReLU (Adam, 10 layers)':<35} {relu_acc:<20.2f} Vanishing Gradient test")
print(f"  {'Sigmoid (Adam, 10 layers)':<35} {sigmoid_acc:<20.2f} Shows vanishing gradient")

print()
for opt_name in ["SGD", "SGD+Momentum", "RMSProp", "Adam"]:
    acc_no_bn = max(histories_bn[f"{opt_name} (no BN)"]['val_accuracy']) * 100
    acc_bn    = max(histories_bn[f"{opt_name} + BN"]['val_accuracy']) * 100
    gain      = acc_bn - acc_no_bn
    print(f"  {opt_name + ' + BatchNorm':<35} {acc_bn:<20.2f} BN gain: +{gain:.2f}%")

print("=" * 75)
print("\nAll plots saved! Files:")
print("  1. sample_images.png      — Fashion-MNIST samples")
print("  2. optimizer_comparison.png — SGD vs SGD+M vs RMSProp vs Adam")
print("  3. vanishing_gradient.png   — ReLU vs Sigmoid")
print("  4. batchnorm_effect.png     — Effect of Batch Normalization")
print("\nAssignment Complete!")