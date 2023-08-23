from averageAttack import AverageAttack
from bandwagonAttack import BandWagonAttack
from randomAttack import RandomAttack
from RR_Attack import RR_Attack
from hybridAttack import HybridAttack
import numpy as np

# np.random.seed(11)

attack = RandomAttack('./config/config.conf')
attack.insertSpam()
#attack.farmLink()
attack.generateLabels('RT_0.04_random_0.07_5_label.txt')
attack.generateProfiles('RT_0.04_random_0.07_5_all.txt')
#attack.generateSocialConnections('relations.txt')