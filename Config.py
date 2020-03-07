class Config:

    save_model_name = ''

    DATA_PATH = 'E:\EMOTION\RAVDESS_labeled'
    CLASS_LABELS = ("1Neutral", "2Calm","3Happy", "4Sad", "5Angry", "6Fearful", "7Disgust","8Surprised")
    #     '01': '1Neutral',
    #     '02': '2Calm',
    #     '03': '3Happy',
    #     '04': '4Sad',
    #     '05': '5Angry',
    #     '06': '6Fearful',
    #     '07': '7Disgust',
    #     '08': '8Surprised'


    # Opensmile
    CONFIG = 'IS10_paraling'
    OPENSMILE_PATH = '../../opensmile'
    FEATURE_NUM = {
        'IS09_emotion': 384,
        'IS10_paraling': 1582,
        'IS11_speaker_state': 4368,
        'IS12_speaker_trait': 6125,
        'IS13_ComParE': 6373,
        'ComParE_2016': 6373
    }

    #Feature
    FEATURE_PATH = 'Features/'

    # Opensmile
    TRAIN_FEATURE_PATH_OPENSMILE = FEATURE_PATH + 'SVM_LIB.csv'
    PREDICT_FEATURE_PATH_OPENSMILE = FEATURE_PATH + 'SVM_LIB.csv'

    # Model
    MODEL_PATH = 'Models/'