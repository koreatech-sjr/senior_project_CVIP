# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import os
from datetime import datetime
import pprint
import json

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# 랜덤시드 고정시키기
np.random.seed(3)

# 1. 데이터 생성하기
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './data/train',
    target_size=(128, 128),
    batch_size=1,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    './data/test',
    target_size=(128, 128),
    batch_size=1,
    class_mode='categorical')


### 하이퍼 파리미터 #######################################################################################################################
class_num = 40
STEP_PER_EPOCH = 10 # class_num * 7
EPOCHS = 1
#######################################################################################################################################
# channel_size = 2 
# channel_multiple = 5
### max channel size == 2^channel_multiple+3
convolutionSize = [5,10,50,100]
FC = [1000]
#######################################################################################################################################
#######################################################################################################################################
print()
for idx, n in enumerate(convolutionSize):
    print("***** Convolution layer "+str(idx+1)+" :",n)
print()
for idx, n in enumerate(FC):
    print("***** Fully Connected Layer "+str(idx+1)+" :", n)
print()
# 2. 모델 구성하기
model = Sequential()
# Conv 1 (3,3)
model.add(Conv2D(convolutionSize[0], kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3))) # 126 * 126 * 3
# MAXPOOL1
model.add(MaxPooling2D(pool_size=(2, 2)))

# CONV 2 (4,4)
model.add(Conv2D(convolutionSize[1], kernel_size=(4, 4), activation='relu', input_shape=(63, 63, 3))) # => 60 * 60 * 3
# MAXPOOL 2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Conv 3 (3,3)
model.add(Conv2D(convolutionSize[2], kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 3))) # => 28 * 28 * 3
# MAXPOOL3
model.add(MaxPooling2D(pool_size=(2, 2)))

# CONV 4 (3,3)
model.add(Conv2D(convolutionSize[3], kernel_size=(3, 3), activation='relu', input_shape=(14, 14, 3))) # => 12 * 12 *3
# MAXPOOL 4
model.add(MaxPooling2D(pool_size=(2, 2)))

# => 6 * 6 * 3
model.add(Flatten())
# model.add(Dense(pow(channel_size, channel_multiple+3), activation='relu'))
model.add(Dense(FC[0], activation='relu'))
model.add(Dense(class_num, activation='softmax'))

# 모델 출력
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit_generator(
    train_generator,
    steps_per_epoch=STEP_PER_EPOCH,
    epochs=EPOCHS,
)

# 5. 모델 평가하기
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps= 1)#class_num*3)
print(scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 6. 모델 사용하기
print("-- Predict --")
output = model.predict_generator(test_generator, steps=1) #class_num*6)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)


# 7. 모델 저장 경로 생성
today = str(datetime.today().strftime("%m_%d_%H_%M_%s"))
os.makedirs(os.path.join("./model", today))

# 8. 모델 불러오기
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# 9. 웨이트 저장하기
path = "./model/"+today+"/SV13_model_weight.h5"
model.save_weights(path)

# 10. 모델 불러오기
json_file = open("model.json", "r") 
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json)

# 11. 모델에다가 웨이트 장착
loaded_model.load_weights(path) 
print("Loaded model from disk")

print(loaded_model)
