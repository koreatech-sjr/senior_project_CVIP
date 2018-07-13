import os
from datetime import datetime

today = datetime.today().strftime("%m_%d_%H_%M")
os.makedirs(os.path.join("./model", today))
