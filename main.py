import glob, os, shutil
import Opensmile_Feature #Feature Extraction
from ML_Model import SVM_Model, MLP_Model # SVM, MLP - Train & Evaluate
from Config import Config #File Path

from Utils import load_model, Radar, playAudio, Waveform, Spectrogram #Audio Data Utils


print ("TEST-SVM using opensmile for SAVEE")
print ("FEATURE: IS10_paraling")


def Train(save_model_name: str):
    Config.save_model_name = save_model_name
    x_train, x_test, y_train, y_test = Opensmile_Feature.get_data(Config.DATA_PATH, Config.TRAIN_FEATURE_PATH_OPENSMILE, train=True)
    model = SVM_Model()
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    model.save_model(save_model_name)

    return model

def Predict(model, file_path: str):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path
    Opensmile_Feature.get_data(file_path, Config.PREDICT_FEATURE_PATH_OPENSMILE, train=False)
    test_feature = Opensmile_Feature.load_feature(Config.PREDICT_FEATURE_PATH_OPENSMILE, train=False)

    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)[0]
    print('Recogntion: ', Config.CLASS_LABELS[int(result)])
    print('Probability: ', result_prob)
    # Radar(result_prob)
    # Waveform(file_path)
    # Spectrogram(file_path)

## Data Labeling
# emotions = {
#     '01': '1Neutral',
#     '02': '2Calm',
#     '03': '3Happy',
#     '04': '4Sad',
#     '05': '5Angry',
#     '06': '6Fearful',
#     '07': '7Disgust',
#     '08': '8Surprised'
# }
# for file in glob.glob("E:\\EMOTION\\RAVDESS_speech\\Actor_*\\*.wav"):
#     file_name = os.path.basename(file)
#     emotion = emotions[file_name.split("-")[2]]
#     print("file:", file)
#     print("file_name:", file_name)
#     newpath = "E:\EMOTION\RAVDESS_labeled\\" + emotion + "\\" + file_name
#     print("newpath", newpath)
#     try:
#         shutil.move(file, newpath)
#         print("Move ", file, " to ", newpath)
#     except:
#         continue


## Trainig & Validating
Train("SVM_LIB")


## Prediction (each file)
# file_path = "file_name"
# model = load_model("SVM_LIB")
# Predict(model, file_path)