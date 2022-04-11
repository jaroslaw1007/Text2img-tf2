from easydict import EasyDict as edict

cfg = edict()

# Dataset
cfg.DATASET = edict()
cfg.DATASET.IMAGE_PATH = './dataset/text2ImgData.pkl'
cfg.DATASET.CAPTION_PATH = './dataset/skip_train_caption.pkl'
cfg.DATASET.LOOK_UP = './dataset/lookup_table.npy'
cfg.DATASET.DICTIONARY_PATH = './dictionary'
cfg.DATASET.MAX_CAPTION_PER_IMAGE = 10

# Training options
cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 24
cfg.TRAIN.MAX_EPOCH = 600
cfg.TRAIN.DISCRIMINATOR_LR = 2e-4
cfg.TRAIN.GENERATOR_LR = 2e-4
cfg.TRAIN.LR_DECAY = 0.5
cfg.TRAIN.LR_DECAY_STEP = 100
cfg.TRAIN.CKPT_DIR = './checkpoints2'

cfg.TRAIN.COEFF = edict()
cfg.TRAIN.COEFF.KL = 2.0
cfg.TRAIN.COEFF.UNCOND_LOSS = 1.0
cfg.TRAIN.COEFF.COLOR_LOSS = 50.0

cfg.TRAIN.SNAPSHOT = edict()
cfg.TRAIN.SNAPSHOT.INTERVAL = 2000
cfg.TRAIN.SNAPSHOT.SAMPLE_COL = 6
cfg.TRAIN.SNAPSHOT.SAMPLE_ROW = 4
cfg.TRAIN.SNAPSHOT.IMAGE_DIR = './imgs2'

cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 21
cfg.TEST.CAPTION_PATH = './dataset/skip_test_caption.pkl'
cfg.TEST.ID_PATH = './dataset/testData.pkl'
cfg.TEST.INFERENCE_PATH = './inference'

# Model options
cfg.GAN = edict()
cfg.GAN.EMBEDDING_DIM = 128
cfg.GAN.DF_DIM = 64
cfg.GAN.GF_DIM = 64
cfg.GAN.Z_DIM = 128

cfg.TEXT = edict()
cfg.TEXT.DIMENSION = 4800
