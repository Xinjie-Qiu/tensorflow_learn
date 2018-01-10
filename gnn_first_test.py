import tensorflow as tf
import numpy as np

BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEA = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.stack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
adfaefa
gag
