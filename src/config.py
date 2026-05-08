# I define the size that  I would like to open  the format image
# It is essential to have a fixed size; otherwise, the features extracted from different
# images may have non-uniform dimensions, and as a result, the classifier may not train the model correctly.

IMAGE_SIZE = (256,128)
RANDOM_STATE = 42
TRAIN_DIR = "data/train"
TEST_DIR= "data/test"

MODEL_PATH = "models/svm_banknote_model.pkl"
SCALER_PATH = "models/scaler.pkl"

LABELS = {
    "genuine":0,
    "fake":1

}

LABELS_NAMES = {
    0:"genuine",
    1:"fake"
}
