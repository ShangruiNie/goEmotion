class configs:
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    batch_size = 16
    learning_rate = 1e-5
    num_classes = 6
    max_len_of_sequence = 30
    max_epoch_num = 50
    random_seed = 111
    num_workers = 4
    emotion_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise']
    six_emotion_label = {"anger": 0,"disgust": 1,"fear": 2,"joy": 3,"sadness": 4,"surprise": 5}
    emotion_mapping = {'anger': 0, 'annoyance': 0, 'disapproval': 0, 'disgust': 1, 'fear': 2, 'nervousness': 2, 'joy': 3, 'amusement': 3, 'approval': 3, 'excitement': 3, 'gratitude': 3, 'love': 3, 'optimism': 3, 'relief': 3, 'pride': 3, 'admiration': 3, 'desire': 3, 'caring': 3, 'sadness': 4, 'disappointment': 4, 'embarrassment': 4, 'grief': 4, 'remorse': 4, 'surprise': 5, 'realization': 5, 'confusion': 5, 'curiosity': 5}
    
    emotion_labels_27 = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise"
    }