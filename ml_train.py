# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--dataset",
                        required=True,
                        help="path to input dataset")

    parser.add_argument("-lr",
                        "--learning_rate",
                        type=float,
                        default=0.001,
                        help="define learning rate of the model")

    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=35,
                        help="define number of epochs")

    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=8,
                        help="define number of epochs")

    parser.add_argument("-p",
                        "--plot",
                        type=str,
                        default="plot.png",
                        help="path to output loss/accuracy plot")

    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default="covid19.model",
                        help="path to output loss/accuracy plot")

    args = vars(parser.parse_args())

    image_path = list(paths.list_images(args["dataset"]))
    # create the dataset
    data, labels = create_dataset(image_path)
    # apply one-hot encoding
    labels, label_binerizer = one_hot_encoding(labels)
    # split data into train and test
    (train_features, test_features, train_labels, test_labels) = split_data(
        data, labels)
    # data augmentation
    aug = data_augmentation()
    # create model
    base_model, final_model = create_model()
    # compile the model 
    train_network = compile_model(args['learning_rate'], args['epochs'], aug, train_features,
                  train_labels, test_features, test_labels, args['batch_size'], base_model, final_model)

    # predict on unseen data 
    predict(test_features, test_labels, args['batch_size'], label_binerizer, final_model)
    # plot the training loss and accuracy
    plot_loss_accuracy(args['epochs'], train_network, args['plot'])
    # save the model
    save_model(final_model, args['model'])


# read images and transform data into model competible 
def create_dataset(image_path):

    data = []
    labels = []
    for imagePath in image_path:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        data.append(image)
        labels.append(label)

    data = np.array(data) / 255.0
    labels = np.array(labels)
    return data, labels


# parserply one hot encoding
def one_hot_encoding(labels):
    label_binerizer = LabelBinarizer()
    labels = label_binerizer.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, label_binerizer


# split the dataset into train and test
def split_data(data, labels):
    (train_features, test_features, train_labels,
     test_labels) = train_test_split(data,
                                     labels,
                                     test_size=0.20,
                                     stratify=labels,
                                     random_state=42)
    return (train_features, test_features, train_labels, test_labels)


# parserply some data augmentation
def data_augmentation():
    augmentation = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
    return augmentation


def create_model():
    # load VGG16 network
    base_model = VGG16(weights="imagenet",
                       include_top=False,
                       input_tensor=Input(shape=(224, 224, 3)))
    # change the head of the model (on top of the base model)
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(64, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)
    # final model
    final_model = Model(inputs=base_model.input, outputs=head_model)
    return base_model, final_model


def compile_model(learning_rate, epochs, augmentation, train_features,
                  train_labels, test_features, test_labels, batch_size, base_model, final_model):
    # freeze the base model
    for layer in base_model.layers:
        layer.trainable = False

    print("[INFO] compiling model...")
    optimizer = Adam(lr=learning_rate, decay=learning_rate / epochs)
    final_model.compile(loss="binary_crossentropy",
                        optimizer=optimizer,
                        metrics=["accuracy"])
    # train the head of the network
    print("[INFO] training head...")
    # train the network
    train_network = final_model.fit_generator(
        augmentation.flow(train_features, train_labels, batch_size=batch_size),
        steps_per_epoch=len(train_features) // batch_size,
        validation_data=(test_features, test_labels),
        validation_steps=len(test_features) // batch_size,
        epochs=epochs)
    return train_network


def predict(test_features, test_labels, batch_size, label_binerizer, final_model):
    print("[INFO] evaluating network...")
    prediction_index = final_model.predict(test_features, batch_size=batch_size)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    prediction_index = np.argmax(prediction_index, axis=1)
    # show a nicely formatted classification report
    print(
        classification_report(test_labels.argmax(axis=1),
                              prediction_index,
                              target_names=label_binerizer.classes_))

    conf_matrix = confusion_matrix(test_labels.argmax(axis=1),
                                   prediction_index)
    total = sum(sum(conf_matrix))
    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / total
    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    specificity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(conf_matrix)
    print("accuracy: {:.4f}".format(accuracy))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))


def plot_loss_accuracy(epochs, train_network, plot):
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N),
             train_network.history["loss"],
             label="train_loss")
    plt.plot(np.arange(0, N),
             train_network.history["val_loss"],
             label="val_loss")
    plt.plot(np.arange(0, N),
             train_network.history["accuracy"],
             label="train_acc")
    plt.plot(np.arange(0, N),
             train_network.history["val_accuracy"],
             label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("static/image/"+ plot)


def save_model(final_model, model_name):
    print("[INFO] saving COVID-19 detector model...")
    final_model.save(model_name, save_format="h5")


if __name__ == "__main__":
    main()
