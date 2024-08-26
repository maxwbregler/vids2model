import tensorflow, keras, sys, os, time, numpy, random, matplotlib.pyplot, math, seaborn, pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GaussianNoise
from keras.optimizers import SGD
from keras import callbacks
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from random import randrange
def shuffleAndReformatData(images, values, dataCutoff, imgTrain, imgValidation, valTrain, valValidation, width, height):
	imagesAndValues = list(zip(images, values))
	random.shuffle(imagesAndValues)
	images, values = zip(*imagesAndValues)
	imgTrain = images[:dataCutoff]
	imgValidation = images[dataCutoff:]
	valTrain = values[:dataCutoff]
	valValidation = values[dataCutoff:]
	newImgTrain = numpy.array([x for x in imgTrain])
	newImgValidation = numpy.array([x for x in imgValidation])
	newValTrain = numpy.array([x for x in valTrain])
	newValValidation = numpy.array([x for x in valValidation])
	newValTrain = keras.utils.to_categorical(newValTrain, 2)
	newValValidation = keras.utils.to_categorical(newValValidation, 2)
	newImgTrain = newImgTrain.reshape(newImgTrain.shape[0], width, height, 3)
	newImgValidation = newImgValidation.reshape(newImgValidation.shape[0], width, height, 3)
	newImgTrain = newImgTrain.astype("float32")
	newImgValidation = newImgValidation.astype("float32")
	newImgTrain = newImgTrain / 255
	newImgValidation = newImgValidation / 255
	masterData = [newImgTrain, newImgValidation, newValTrain, newValValidation]
	return masterData
class shuffleDataCallback(keras.callbacks.Callback):
	def __init__(self, images, values, dataCutoff, imgTrain, imgValidation, valTrain, valValidation, width, height):
		super(shuffleDataCallback, self).__init__()
		self.images = images
		self.values = values
		self.dataCutoff = dataCutoff
		self.imgTrain = imgTrain
		self.imgValidation = imgValidation
		self.valTrain = valTrain
		self.valValidation = valValidation
		self.width = width
		self.height = height
	def on_epoch_end(self, epoch, logs = None):
		masterData = shuffleAndReformatData(self.images, self.values, self.dataCutoff, self.imgTrain, self.imgValidation, self.valTrain, self.valValidation, self.width, self.height)
		newImgTrain, newImgValidation, newValTrain, newValValidation = masterData[0], masterData[1], masterData[2], masterData[3]
def createModel(modelName, data1Path, data2Path, width, height, learningRate, epochs, isShufflingBetweenEpochs, isChangingBrightness, isChangingSharpness, isChangingContrast, isChangingRotation, isChangingPosition):
	random.seed(time.time())
	print("Creating " + modelName + " Model")
	width = int(width)
	height = int(height)
	learningRate = float(learningRate)
	epochs = int(epochs)
	isShufflingBetweenEpochs = int(isShufflingBetweenEpochs)
	isChangingBrightness = int(isChangingBrightness)
	isChangingSharpness = int(isChangingSharpness)
	isChangingContrast = int(isChangingContrast)
	isChangingRotation = int(isChangingRotation)
	isChangingPosition = int(isChangingPosition)
	originalData1Path = data1Path
	originalData2Path = data2Path
	os.system("rm " + modelName + ".zip " + modelName + "Accuracy.png " + modelName + "Matrix.png " + "model.keras")
	os.system("rm -rf data1")
	os.system("mkdir data1")
	os.system("ffmpeg -i " + data1Path + " data1/%04d.png" + " >/dev/null 2>&1")
	data1Path = "data1"
	os.system("rm -rf data2")
	os.system("mkdir data2")
	os.system("ffmpeg -i " + data2Path + " data2/%04d.png" + " >/dev/null 2>&1")
	data2Path = "data2"
	dataLen = 0
	images, values, imgTrain, imgValidation, valTrain, valValidation = [], [], [], [], [], []
	for x in os.listdir(data1Path):
		if(os.path.isfile(data1Path + "/" + x)):
			imageData = Image.open(data1Path + "/" + x)
			imageData = imageData.resize((width, height))
			imageData = numpy.array(imageData)
			imageData = imageData.reshape(width, height, 3)
			dataLen += 1
			images.append(imageData)
			values.append(0)
	for x in os.listdir(data2Path):
		if(os.path.isfile(data2Path + "/" + x)):
			imageData = Image.open(data2Path + "/" + x)
			imageData = imageData.resize((width, height))
			imageData = numpy.array(imageData)
			imageData = imageData.reshape(width, height, 3)
			dataLen += 1
			images.append(imageData)
			values.append(1)
	originalDataLen = dataLen
	if(isChangingBrightness == 1 or isChangingSharpness == 1 or isChangingContrast == 1 or isChangingRotation == 1 or isChangingPosition == 1):
		for x in range(0, originalDataLen):
			newImg = images[x]
			newVal = values[x]
			newImg = Image.fromarray(newImg)
			if(isChangingBrightness == 1):
				filter = ImageEnhance.Brightness(newImg)
				brightness = random.uniform(0.25, 1.75)
				newImg = filter.enhance(brightness)
				filter = ImageEnhance.Sharpness(newImg)
			if(isChangingSharpness == 1):
				filter = ImageEnhance.Sharpness(newImg)
				sharpness = random.uniform(-2, 2)
				newImg = filter.enhance(sharpness)
				filter = ImageEnhance.Contrast(newImg)
			if(isChangingContrast == 1):
				filter = ImageEnhance.Contrast(newImg)
				contrast = random.uniform(0.25, 1.75)
				newImg = filter.enhance(contrast)
				newImg = numpy.array(newImg)
			if(isChangingRotation == 1):
				maxRot = 30
				newImg = numpy.expand_dims(newImg, axis = 0)
				dataGen = ImageDataGenerator(rotation_range = maxRot, fill_mode = "nearest")
				augIter = dataGen.flow(newImg, batch_size = 1)
				newImg = next(augIter)[0].astype('uint8')
			if(isChangingPosition == 1):
				maxMove = 0.1
				newImg = numpy.expand_dims(newImg, axis = 0)
				dataGen = ImageDataGenerator(width_shift_range = maxMove, height_shift_range = maxMove)
				augIter = dataGen.flow(newImg, batch_size = 1)
				newImg = next(augIter)[0].astype('uint8')
			newImg = newImg.reshape(width, height, 3)
			images.append(newImg)
			values.append(newVal)
			dataLen += 1
	dataAmount = dataLen * width * height * epochs
	dataLog = math.log10(dataAmount)
	print("Model has data logarithm of " + str(round(dataLog, 3)))
	dataCutoff = round(dataLen * 0.8)
	batchSize = round(dataCutoff / 100)
	masterData = shuffleAndReformatData(images, values, dataCutoff, imgTrain, imgValidation, valTrain, valValidation, width, height)
	newImgTrain, newImgValidation, newValTrain, newValValidation = masterData[0], masterData[1], masterData[2], masterData[3]
	imageData = Image.open("data1/0001.png")
	inputShape = (width, height, 3)
	model = Sequential()
	model.add(GaussianNoise(0.1))
	model.add(Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = inputShape))
	model.add(Conv2D(64, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(256, activation =  "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation="softmax"))
	optimizer = SGD(learning_rate = learningRate, momentum = 0.9)
	model.compile(loss = "mse", optimizer = optimizer, metrics = ["accuracy"])
	if(isShufflingBetweenEpochs == 1):
		callbacks = [shuffleDataCallback(images, values, dataCutoff, imgTrain, imgValidation, valTrain, valValidation, width, height)]
	else:
		callbacks = []
	fittedModel = model.fit(newImgTrain, newValTrain, batch_size = batchSize, epochs = epochs, verbose = 1, validation_data = (newImgValidation, newValValidation), callbacks = callbacks)
	images = numpy.asarray(images)
	images = images.reshape(-1, width, height, 3)
	prediction = model.predict(images)
	predictions = []
	for i in prediction:
		predictions.append(numpy.argmax(i))
	os.system("mkdir model")
	confusionMatrix = confusion_matrix(values, predictions, normalize="pred")
	confusionMatrix = pandas.DataFrame(confusionMatrix, range(2), range(2))
	seaborn.set(font_scale=1.4)
	seaborn.heatmap(confusionMatrix, annot = True, annot_kws={"size": 16})
	matplotlib.pyplot.ylabel("predicted value")
	matplotlib.pyplot.xlabel("actual value")
	matplotlib.pyplot.savefig("model/" + modelName + "Matrix.png")
	matplotlib.pyplot.clf()
	matplotlib.pyplot.plot(fittedModel.history['accuracy'])
	matplotlib.pyplot.plot(fittedModel.history['val_accuracy'])
	matplotlib.pyplot.title("Model Accuracy vs Epoch")
	matplotlib.pyplot.ylabel("accuracy")
	matplotlib.pyplot.xlabel("epoch")
	matplotlib.pyplot.legend(["train", "validation"], loc = "upper left")
	matplotlib.pyplot.savefig("model/" + modelName + "Accuracy.png", bbox_inches = "tight")
	model.save("model/model.keras")
	os.system("touch model/config.txt")
	configFile = open("model/config.txt", "w")
	configFile.write(str(width) + " " + str(height))
	configFile.close()
	os.system("zip -r " + modelName + ".zip model")
	os.system("rm -rf data1 data2 model")
	os.system("rm " + originalData1Path + " " + originalData2Path)
	print("Model complete")
	return
if(len(sys.argv) == 14):
	createModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13])
else:
	print("Error: Not enough arguments\n")
