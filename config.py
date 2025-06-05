# Hyperparameters
NUM_POINTS = 4002   # Number of points to sample from each point cloud
BATCH_SIZE = 32 # Batch size for DataLoaders
EPOCHS = 100    # Total number of training epochs
LEARNING_RATE = 0.00025 # Learing rate for training
NUM_CLASSES = 3 # Number of segmentation classes
WEIGHT_DECAY = 0.005    # Regularization strength
CLASS_WEIGHTS = [0.0577, 0.3883, 0.7539]

ALPHA = 1.0    
BETA = 1.0
GAMMA = 10.0

DILATION_RADIUS = 15.0     # Dilation of only ligament labels 

# Model and checkpoint paths
SAVE_DIR = "checkpoints"
NPZ_PATH = "" # path to npz files

CLASS_NAMES = ["Liver", "Ridge", "Ligament"]
CLASS_COLORS = ['gray', 'red', 'blue']
