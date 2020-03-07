import glob, os, shutil
import Opensmile_Feature #Feature Extraction
from ML_Model import SVM_Model, MLP_Model # SVM, MLP - Train & Evaluate
from Config import Config #File Path


print ("TEST-SVM using opensmile for RAVEE")
print ("FEATURE: IS10_paraling")


def Train(save_model_name: str):
    Config.save_model_name = save_model_name
    x_train, x_test, y_train, y_test = Opensmile_Feature.get_data(Config.DATA_PATH, Config.TRAIN_FEATURE_PATH_OPENSMILE, train=True)
    model = SVM_Model()
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    model.save_model(save_model_name)

    return model

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

