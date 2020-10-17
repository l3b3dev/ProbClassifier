import numpy as np
import cv2 as cv
from skimage.feature import hog
from skimage import data, exposure
from sklearn import model_selection
import matplotlib.pyplot as plt


class FeaturesController:
    def __init__(self, images_path, labels_path):
        self.labels = np.load(labels_path)
        self.images = np.load(images_path)
        # set defaults for feature params
        self.diffusion_vector = 10
        self.hog_params = [18, 6, 1]
        self.X_train = self.images
        self.y_train = self.labels

    def train_test_split(self, split_percent):
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.images,
                                                                                                self.labels,
                                                                                                train_size=split_percent)

    def extract_diffusion_features(self, image, vector_size=10):
        alg = cv.KAZE_create()
        # image keypoints
        kps = alg.detect(image)

        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        dsc = dsc.flatten()
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

        return dsc

    def tune_diffusion(self, model):
        # tune KAZE vector sizes
        kaze_tuning_list = range(10, 100, 5)
        kaze_score = []
        for v in kaze_tuning_list:
            features = self.extract_diffusion_all_images(vector_size=v)

            run_score = model_selection.cross_val_score(model, features, self.y_train, cv=5, scoring='accuracy')
            kaze_score.append(np.mean(run_score))

        self.diffusion_vector = kaze_tuning_list[kaze_score.index(max(kaze_score))]

    def extract_diffusion_all_images(self, vector_size=10, training=True):
        ims = self.X_train if training else self.X_test
        features = self.extract_diffusion_features(ims[0], vector_size)
        for i in range(1, ims.shape[0]):
            features = np.vstack((features, self.extract_diffusion_features(ims[0], vector_size)))

        return features

    def tune_gradients(self, model):
        # must tune hog
        hog_tuning_list = [[y, x, j] for y in [4, 6, 9, 12, 18] for x in [6, 8, 10, 12] for j in [1, 2]]

        hog_score = []
        for hvals in hog_tuning_list:
            features = hog(self.X_train[0], orientations=hvals[0], pixels_per_cell=(hvals[1], hvals[1]),
                           cells_per_block=(hvals[2], hvals[2]), feature_vector=True)
            for i in range(1, self.X_train.shape[0]):
                features = np.vstack(
                    (features, hog(self.X_train[i], orientations=hvals[0], pixels_per_cell=(hvals[1], hvals[1]),
                                   cells_per_block=(hvals[2], hvals[2]), feature_vector=True)))

            run_score = model_selection.cross_val_score(model, features, self.y_train, cv=5, scoring='accuracy')
            hog_score.append(np.mean(run_score))

        self.hog_params = hog_tuning_list[hog_score.index(max(hog_score))]

    def extract_gradients_all_images(self, training=True):
        ims = self.X_train if training else self.X_test
        features = hog(ims[0], orientations=self.hog_params[0],
                       pixels_per_cell=(self.hog_params[1], self.hog_params[1]),
                       cells_per_block=(self.hog_params[2], self.hog_params[2]), feature_vector=True)
        for i in range(1, ims.shape[0]):
            features = np.vstack((features, hog(ims[i], orientations=self.hog_params[0],
                                                pixels_per_cell=(self.hog_params[1], self.hog_params[1]),
                                                cells_per_block=(self.hog_params[2], self.hog_params[2]))))

        return features

    def extract_features(self, training=True):
        diff_features = self.extract_diffusion_all_images(vector_size=self.diffusion_vector, training=training)
        hog_features = self.extract_gradients_all_images(training)

        return np.hstack((diff_features, hog_features))

    def plot_kaze_descriptors(self, vec_size=10):
        # get an image from each class
        images = np.array([self.X_train[self.y_train == i][0] for i in range(5)])
        for img in images:
            alg = cv.KAZE_create()
            kp = alg.detect(img)

            kp = sorted(kp, key=lambda x: -x.response)[:vec_size]
            # computing descriptors vector
            kp, dsc = alg.compute(img, kp)

            img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
            plt.imshow(img2), plt.show()

    def plot_gradients(self):
        # get an image from each class
        images = np.array([self.X_train[self.y_train == i][0] for i in range(5)])
        for img in images:
            fd, hog_image = hog(img, orientations=self.hog_params[0],
                                pixels_per_cell=(self.hog_params[1], self.hog_params[1]),
                                cells_per_block=(self.hog_params[2], self.hog_params[2]), visualize=True,
                                multichannel=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

            ax1.axis('off')
            ax1.imshow(img, cmap=plt.cm.gray)
            ax1.set_title('Input image')

            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
