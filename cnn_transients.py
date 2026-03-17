import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


def read_data(filename: str):
    """Lee un CSV donde la primera columna es tiempo y las demás son señales."""
    dataset = np.genfromtxt(filename, delimiter=',', skip_header=1)
    dataframe = pd.read_csv(filename)

    labels = dataframe.columns.tolist()[1:]
    new_labels = []
    for label in labels:
        if 'linea' in label.lower():
            new_labels.append('Linea')
        else:
            new_labels.append('Banco')

    new_labels = np.array(new_labels)
    time = dataset[:, 0]
    data = dataset[:, 1:].T
    return data, new_labels, time


def scaler(x: np.ndarray):
    scaler_obj = MinMaxScaler(feature_range=(0, 1))
    scaler_obj.fit(x.T)
    return scaler_obj.transform(x.T).T


def plot_waveform(x1_train, x2_train, tiempo_b, tiempo_l):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        if i >= len(x1_train) or i >= len(x2_train):
            break
        ax.plot(range(tiempo_b), x1_train[i, :], color='b', linewidth=0.9)
        ax.plot(range(tiempo_l), x2_train[i, :], color='r', linewidth=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def make_model(input_shape, num_classes, num_filters, kernel_size, padding,
               conv_activation, pooling_size, dense_activation):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=conv_activation,
        ),
        keras.layers.MaxPooling1D(pool_size=pooling_size),
        keras.layers.Flatten(),
        keras.layers.Dense(num_classes, activation=dense_activation),
    ])
    return model


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment='center',
            color='white' if cm[i, j] > thresh else 'black',
        )

    plt.tight_layout()
    plt.ylabel('Etiqueta')
    plt.xlabel('Predicción')
    plt.show()


def main():
    banco_entreno = 'prueba_banco.csv'
    banco_test = 'test_banco.csv'
    linea_entreno = 'prueba_linea.csv'
    linea_test = 'test_linea.csv'

    required_files = [banco_entreno, banco_test, linea_entreno, linea_test]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            'Faltan archivos CSV requeridos: ' + ', '.join(missing)
        )

    x1_train, y1_train, _ = read_data(banco_entreno)
    x1_test, y1_test, _ = read_data(banco_test)
    x2_train, y2_train, _ = read_data(linea_entreno)
    x2_test, y2_test, _ = read_data(linea_test)

    x_train = np.concatenate((x1_train, x2_train), axis=0)
    y_train = np.concatenate((y1_train, y2_train), axis=0)
    x_test = np.concatenate((x1_test, x2_test), axis=0)
    y_test = np.concatenate((y1_test, y2_test), axis=0)

    x_train = scaler(x_train)
    x_test = scaler(x_test)

    tiempo_b = x1_train.shape[1]
    tiempo_l = x2_train.shape[1]
    plot_waveform(x1_train, x2_train, tiempo_b, tiempo_l)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = (x_train.shape[1], 1)
    clase = 2
    filtros = 16
    kernel = 123
    pooling_size = 4

    model = make_model(
        input_shape=input_shape,
        num_classes=clase,
        num_filters=filtros,
        kernel_size=kernel,
        padding='same',
        conv_activation='sigmoid',
        pooling_size=pooling_size,
        dense_activation='softmax',
    )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'best_model.keras', save_best_only=True, monitor='val_loss'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50, verbose=1
        ),
    ]

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=16,
        epochs=100,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    model = keras.models.load_model('best_model.keras')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    metric = 'sparse_categorical_accuracy'
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('Model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.show()
    plt.close()

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    sns.set_style('white')

    class_names = list(encoder.classes_)
    plot_confusion_matrix(
        confusion_matrix(y_test, y_pred_classes),
        classes=class_names,
        title='Matriz de Confusión: Clasificador CNN',
    )
    print(classification_report(y_test, y_pred_classes, target_names=class_names))


if __name__ == '__main__':
    main()
