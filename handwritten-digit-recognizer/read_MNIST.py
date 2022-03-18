import numpy as np
import matplotlib.pyplot as plt

# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


def load_tests(path):
    # Reading The Train Set
    train_images_file = open(path + 'train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open(path + 'train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []
    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
        
        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        
        train_set.append((image, label))
       
    # Reading The Test Set
    test_images_file = open(path + 't10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open(path + 't10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256
        
        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        
        test_set.append((image, label))

    return train_set, test_set

if __name__ == '__main__':
    train_set, test_set = load_tests("tests/")
    
    # Plotting an image
    plt.gcf().canvas.set_window_title("first of train set")
    show_image(train_set[0][0])
    plt.show()

    plt.gcf().canvas.set_window_title("first of test set")
    show_image(test_set[0][0])
    plt.show()
    