import os
import random
from PIL import Image
import sys


#파일 만들기
def makeDirs(YOUR_DIRECTORY_NAME):
    try:
        if not(os.path.isdir(YOUR_DIRECTORY_NAME)):
            os.makedirs(os.path.join(YOUR_DIRECTORY_NAME))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("aleady exist")

# 회전시키기
def rotateImage(img, angle):
    return img.rotate(angle)

# 사이즈줄이기
def resizeImage(img, size):
    return img.resize(size)

# 이미지 복사
def copyImage(img, trainNumberArray, testNumberArray, className, filename):
    for trainNumber in trainNumberArray:
        if trainNumber in fileName:
            # make folder by class 
            makeDirs( DEST_DIR + TRAIN_DIR + className )
            # image save in class folder
            img.save( DEST_DIR + TRAIN_DIR + className + '/' + filename )
    for testNumber in testNumberArray:    
        if testNumber in fileName:
            makeDirs(DEST_DIR + TEST_DIR + className)
            img.save(DEST_DIR + TEST_DIR + className + '/' + filename)
# 셔플
def shuffleArray(count):
    global totalNumbers
    global trainNumbers
    global testNumbers

    random.shuffle(totalNumbers)
    trainNumbers = []
    testNumbers = []
    for n in range(0,count):
        trainNumbers.append(totalNumbers[n])
        print(totalNumbers)
    for n in range(count, len(totalNumbers)):
        testNumbers.append(totalNumbers[n])
    
    totalNumbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
  
################################################################################
#                                                                              #
#                   ****folder type****                                        #
#                                                                              #
#    NormalizedImages  : 512*68 only iris rectangle                            #
#    NormalizedMasks   : 512*68 only iris rectangle + only black and white     #
#    SegmentatedImages : 640*480 original image                                #
#    Masks             : 640*480 original image + only black and white         #
#    IrisCode          : 512*384 adjust filter                                 #
#                                                                              #
################################################################################
# RESIZE : HEIGHT * WIDTH
RESIZE_IMAGE = (128,128)

# FILE ORIGIN
ORIGIN_DIR = 'NormalizedImages'
# Todo: input OSIV4.1 ROOT PATH
ROOT_DIR = '/Users/rock/seniorProject/OSIV4.1/'
DEST_DIR = '/Users/rock/seniorProject/data/'
# SAVED PATH
TRAIN_DIR = "train/"
TEST_DIR = "test/"
NOW_DIR = '../data'


# DEFUALT SHUFFLE NUMBER ARRAY
totalNumbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
trainNumbers = ["01", "02", "03", "04", "05", "06", "07"]
testNumbers = ["08", "09", "10"]
TRAIN_COUNT = 7


# ANGLE FALG
ANGLE_FLAG = False
# ROTATE ANGLE ARRAY
rotateAngleArray = []
# MAXIMUM ROTATE ANGLE
MAX_ROTATE_ANGLE = 5




###########################################################################################
# START
fileNames = os.listdir(ROOT_DIR + ORIGIN_DIR)
fileNames.sort()

# MAKE FOLDER
makeDirs(DEST_DIR)
makeDirs(DEST_DIR+TRAIN_DIR)
makeDirs(DEST_DIR+TEST_DIR)

# INIT ROTATE ANGLE ARRAY
for n in range(-MAX_ROTATE_ANGLE, MAX_ROTATE_ANGLE+1):
    rotateAngleArray.append(n)
print(rotateAngleArray)

if sys.argv[1]==0 :
    CLASS_NUMBER = 1000

CLASS_NUMBER = int(sys.argv[1])*10
print("CLASS Number: ", CLASS_NUMBER/10)
 
IDX = 0
for fileName in fileNames:
    # ORIGINAL IMAGE FULL NAME
    fullFilename = os.path.join(ROOT_DIR + ORIGIN_DIR, fileName)
    className = fileName.split('R')[0].strip()
    im = Image.open(fullFilename)
    # 1. IMAGE RESIZE
    im = resizeImage(im, RESIZE_IMAGE)
    if ANGLE_FLAG :
        # 2. ROTATE IMAGE
        for angle in rotateAngleArray:
            rotatedImage = rotateImage(im, angle) 
            # 3. SHUFFLE DATA
            # shuffleArray(TRAIN_COUNT)
            # 4. MAKE CLASS FOLDER + COPY ROTATED IMAGE
            copyImage(rotatedImage, trainNumbers, testNumbers, className, className + fileName.split(".")[0]+"_rotate_"+str(angle)+".bmp")
    # 2. MAKE CLASS FOLDER + COPY ROTATED IMAGE
    else :
        copyImage(im, trainNumbers, testNumbers, className, fileName)
    IDX = IDX+1
    if IDX == CLASS_NUMBER-1:
        break;




