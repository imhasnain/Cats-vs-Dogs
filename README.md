

# 🐶🐱 Dog vs Cat Classification using CNN (TensorFlow/Keras)

This project builds a Convolutional Neural Network (CNN) to classify **Dog** and **Cat** images using the Kaggle *Dog and Cat Classification Dataset*.
The model uses **Batch Normalization**, **Dropout**, **Data Augmentation**, and a **lower learning rate** to reduce overfitting and improve generalization.

---

## 📂 Dataset

The dataset structure:

```
PetImages/
    Dog/
        *.jpg
    Cat/
        *.jpg
```

Loaded using:

```python
image_dataset_from_directory(
    directory=Data_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256,256)
)
```

---

## 🧠 Model Architecture

The CNN includes:

* Convolution layers (32, 64, 128 filters)
* MaxPooling layers
* Batch Normalization
* Dropout (to reduce overfitting)
* Dense layers (128 → 64 → 1)
* Sigmoid activation for binary classification
* Adam optimizer with low learning rate (`0.0001`)

---

## ⚙️ Training

```python
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train,
    epochs=10,
    validation_data=test
)
```

---

## 📈 Accuracy Visualization

Training and validation accuracy were plotted using Matplotlib:

```python
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.show()
```

After adding **BatchNormalization** and **Dropout**, the training and validation accuracy curves came closer, showing reduced overfitting.

---

## 🚀 Results

* The model achieved good accuracy on both training and validation.
* Overfitting was reduced by:

  * BatchNormalization
  * Dropout
  * Lower learning rate (Adam 0.0001)

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## 📦 How to Run

1. Download the dataset (Kaggle).
2. Place the `PetImages/` folder in your environment.
3. Run the training script.
4. View plots to analyze accuracy and loss.

---

## ✔️ Future Improvements

* Add transfer learning (e.g., MobileNetV2, ResNet50)
* Add confusion matrix & precision/recall
* Deploy model using Flask/Streamlit

---
