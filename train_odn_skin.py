import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ========== КОНФИГУРАЦИЯ ==========
DATA_DIR = r"dno_dataset"
IMG_SIZE = (300, 300)          # для EfficientNetB3
BATCH_SIZE = 24
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 5e-5
SEED = 42

# Этапы: (начало, конец, lr, сколько слоёв разморозить (0 - заморожена, -1 - вся))
STAGES = [
    (0,   15, 1e-3, 0),
    (15,  28, 1e-4, 40),
    (28,  45, 1e-5, 100),
    (45,  60, 1e-6, -1),
]

BASE_MODEL_DIR = "./trained_models"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = os.path.join(BASE_MODEL_DIR, f"efficient_b3_{timestamp}")
os.makedirs(MODEL_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ========== МОДЕЛЬ ==========
def create_model():
    base = EfficientNetB3(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base

# ========== ГЕНЕРАТОРЫ ==========
def get_generators():
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=25.0,
        fill_mode='reflect'
    )
    val_test_gen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True, seed=SEED
    )
    val_gen = val_test_gen.flow_from_directory(
        os.path.join(DATA_DIR, 'valid'), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False, seed=SEED
    )
    test_gen = val_test_gen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False, seed=SEED
    )
    class_names = {v: k for k, v in train_gen.class_indices.items()}
    return train_gen, val_gen, test_gen, class_names

# ========== ВЕСА КЛАССОВ ==========
def get_class_weights(train_gen):
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

# ========== CALLBACK ДЛЯ ПОЭТАПНОЙ РАЗМОРОЗКИ (без ошибок) ==========
class StageUnfreeze(callbacks.Callback):
    def __init__(self, base_model, stages):
        super().__init__()
        self.base_model = base_model
        self.stages = stages
        self.current_stage = 0

    def on_epoch_begin(self, epoch, logs=None):
        if self.current_stage >= len(self.stages):
            return
        start_ep, end_ep, lr, unfreeze_layers = self.stages[self.current_stage]
        if epoch == start_ep:
            print(f"\n{'='*50}")
            print(f"▶ Этап {self.current_stage+1}: эпохи {start_ep}–{end_ep}, LR = {lr:.0e}")
            total = len(self.base_model.layers)
            if unfreeze_layers > 0:
                freeze_until = total - unfreeze_layers
                for i, layer in enumerate(self.base_model.layers):
                    layer.trainable = (i >= freeze_until)
                print(f"   Разморожено последних {unfreeze_layers} слоёв")
            elif unfreeze_layers == -1:
                self.base_model.trainable = True
                print("   Полностью разморожена вся базовая модель")
            else:
                self.base_model.trainable = False
                print("   Базовая модель полностью заморожена")

            # Безопасное изменение learning rate (прямое присваивание)
            self.model.optimizer.learning_rate = lr
            print(f"   Learning rate → {lr}")
            self.current_stage += 1

# ========== TEST TIME AUGMENTATION ==========
def predict_with_tta(model, generator):
    generator.reset()
    images = []
    labels = []
    for i in range(len(generator)):
        x, y = generator[i]
        images.append(x)
        labels.append(y)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    preds = []
    # Оригинал
    preds.append(model.predict(images, verbose=0))
    # Отражение
    flipped = np.flip(images, axis=2)
    preds.append(model.predict(flipped, verbose=0))
    # Поворот 90
    rot90 = np.rot90(images, k=1, axes=(1,2))
    preds.append(model.predict(rot90, verbose=0))
    # Поворот 180
    rot180 = np.rot90(images, k=2, axes=(1,2))
    preds.append(model.predict(rot180, verbose=0))
    # Поворот 270
    rot270 = np.rot90(images, k=3, axes=(1,2))
    preds.append(model.predict(rot270, verbose=0))

    avg_preds = np.mean(preds, axis=0)
    return avg_preds, labels

# ========== ОБУЧЕНИЕ ==========
def train():
    train_gen, val_gen, test_gen, class_names = get_generators()
    class_weights = get_class_weights(train_gen)
    print("Веса классов:", class_weights)

    model, base_model = create_model()

    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_model.keras'),
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=12, restore_best_weights=True, verbose=1
    )
    unfreeze_cb = StageUnfreeze(base_model, STAGES)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=STAGES[0][2]),
        loss=loss,
        metrics=['accuracy']
    )

    total_epochs = STAGES[-1][1]
    print(f"\n🚀 Запуск обучения на {total_epochs} эпох (EfficientNetB3, {IMG_SIZE[0]}x{IMG_SIZE[1]})\n")
    history = model.fit(
        train_gen,
        epochs=total_epochs,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, unfreeze_cb],
        class_weight=class_weights,
        verbose=1
    )

    best_path = os.path.join(MODEL_DIR, 'best_model.keras')
    if os.path.exists(best_path):
        final_model = tf.keras.models.load_model(best_path)
        print(f"\n✅ Загружена лучшая модель (val_accuracy = {max(history.history['val_accuracy']):.4f})")
    else:
        final_model = model

    return final_model, history.history, test_gen, class_names

# ========== ОЦЕНКА ==========
def evaluate_with_tta(model, test_gen, class_names, history):
    print("\n🔍 Оценка с Test Time Augmentation (TTA)...")
    avg_preds, y_true = predict_with_tta(model, test_gen)
    y_pred = np.argmax(avg_preds, axis=1)
    tta_acc = np.mean(y_pred == y_true)
    print(f"📊 TTA test accuracy: {tta_acc:.4f}")

    with open(os.path.join(MODEL_DIR, 'test_results_tta.txt'), 'w') as f:
        f.write(f'TTA Test accuracy: {tta_acc:.4f}\n\n')
        f.write(classification_report(y_true, y_pred, target_names=list(class_names.values())))

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names.values(), yticklabels=class_names.values())
    plt.title('Confusion Matrix (TTA)')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix_tta.png'))
    plt.close()

    # Стандартная оценка без TTA
    test_gen.reset()
    y_pred_no_tta = np.argmax(model.predict(test_gen), axis=1)
    acc_no_tta = np.mean(y_pred_no_tta == test_gen.classes)
    print(f"📊 Без TTA test accuracy: {acc_no_tta:.4f}")

    # Графики
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')
    plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(history['accuracy'], label='Train acc')
    plt.plot(history['val_accuracy'], label='Val acc')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
    plt.close()

    print(f"\n📈 Максимальная val_accuracy: {max(history['val_accuracy']):.4f}")

# ========== MAIN ==========
if __name__ == '__main__':
    print(f"📁 Результаты будут сохранены в: {MODEL_DIR}")
    model, history, test_gen, class_names = train()
    evaluate_with_tta(model, test_gen, class_names, history)
    model.save(os.path.join(MODEL_DIR, 'final_model.keras'))
    with open(os.path.join(MODEL_DIR, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)
    print(f"\n🎉 Готово! Файлы в {MODEL_DIR}")