from ClassificationPipeline import ClassificationPipeline
from FeaturesController import FeaturesController
import pandas as pd

if __name__ == '__main__':
    # question one, features for one image
    data = FeaturesController('one-image.npy', 'one-label.npy')
    one_image_features = data.extract_features()
    print('Features:')
    print(one_image_features)

    # question two, run on trained model
    pipeline = ClassificationPipeline()
    # assuming model.pkl exists in current folder
    pipeline.load_model()
    # assuming features.npy and labels.npy exist in current folder
    features, labels = pipeline.load_features()
    test_classification, conf_matrix = pipeline.test(features, labels)
    #pipeline.plot_roc_curve(features, labels)  in case no scikitplot lib

    print('Classification accuracy:')
    print(test_classification)
    print('Confusion matrix:')
    cm = pd.DataFrame(conf_matrix)
    print(cm)