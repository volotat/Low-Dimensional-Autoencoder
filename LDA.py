from keras.layers   import Input, Dense, Reshape, Flatten
from keras.models   import Model, load_model
from keras.datasets import mnist
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

INPUT_DATA_SHAPE = (28,28)
INPUT_DATA_SIZE  = np.prod(INPUT_DATA_SHAPE)
LATENT_SPACE_DIM = 2

MOVE_REFERENCE_POINTS = 10
REPULSION_INTENSITY = 0.001 #This parameter should be smaller if dataset is denser and/or latent space dimension is smaller

DPI = 80

COMMON_FIT_METHOD = False

def rui(x): # round up to int
	return int(round(x + .5))

def set_models():
	#Setup the model
	input   =  Input(shape = INPUT_DATA_SHAPE)
	encoded = Flatten()(input)
	encoded = Dense(rui(INPUT_DATA_SIZE / 2 ), activation='tanh')(encoded) 
	encoded = Dense(rui(INPUT_DATA_SIZE / 4 ), activation='tanh')(encoded) 
	encoded = Dense(rui(INPUT_DATA_SIZE / 8 ), activation='tanh')(encoded) 
	encoded = Dense(rui(INPUT_DATA_SIZE / 16), activation='tanh')(encoded) 
	encoded = Dense(LATENT_SPACE_DIM, activation='tanh')(encoded) 

	decoded = Dense(rui(INPUT_DATA_SIZE / 16), activation='tanh')(encoded) 
	decoded = Dense(rui(INPUT_DATA_SIZE / 8 ), activation='tanh')(decoded)
	decoded = Dense(rui(INPUT_DATA_SIZE / 4 ), activation='tanh')(decoded)
	decoded = Dense(rui(INPUT_DATA_SIZE / 2 ), activation='tanh')(decoded)
	decoded = Dense(INPUT_DATA_SIZE   , activation='sigmoid')(decoded)
	decoded = Reshape(INPUT_DATA_SHAPE)(decoded)

	#Create models
	autoencoder = Model(input, decoded)
	encoder = 	  Model(input, encoded)

	encoded_input = Input(shape=(LATENT_SPACE_DIM,))
	set = autoencoder.layers[-6](encoded_input)
	for i in range(-5,0):
		set = autoencoder.layers[i](set)

	decoder = Model(encoded_input, set)


	autoencoder.compile(optimizer='adam', loss='mse')
	decoder.compile(optimizer='adam', loss='mse')
	
	return autoencoder, decoder, encoder
	
def get_data():
	(x_train, _), (x_test, _) = mnist.load_data()
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	
	return x_train, x_test
		
def save_results(step):
	b = np.random.randint(0,train_data.shape[0],10)
	test_arr = train_data[b]

	encoded_imgs = encoder.predict(test_arr)
	decoded_imgs = decoder.predict(encoded_imgs)

	n = 10  # how many digits we will display
	plt.figure(figsize=(10, 4), dpi=DPI)
	for k in range(n):
		# display original
		ax = plt.subplot(2, n, k + 1)
		plt.imshow(test_arr[k].reshape(INPUT_DATA_SHAPE))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, k + 1 + n)
		plt.imshow(decoded_imgs[k].reshape(INPUT_DATA_SHAPE))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		
	plt.savefig(str(step)+'_.png', dpi=DPI)
	plt.clf()
	plt.close()
		
def set_latent_target(shape):
	latent_target = np.random.uniform(-1,1,shape)
	
	return latent_target

def stretch_data(data, step):
	start = time.time()	
		
	amin = np.amin(data, axis=0)
	amax = np.amax(data, axis=0)
	
	ones = np.ones(LATENT_SPACE_DIM)
	
	indx = 0
	for d in data:
		#Here we stretch our given latent space in the way so it touches the boundaries
		data[indx] += -amin
		data[indx] *= 2. / (amax - amin)
		data[indx] += -1.
		
		#And here we move current point out of some other datapoints so they will not stuck together, with respect to distances between them
		b = np.random.choice(data.shape[0], MOVE_REFERENCE_POINTS)
		b = np.delete(b, np.argwhere(b == indx))
		
		repuls = data[b]
		
		moves = (repuls - data[indx])
		norms = np.linalg.norm(moves, axis = 1)
		norms = np.power(norms, 2) + norms
		moves = moves / norms[:, np.newaxis]
			
		#If you expect to have a lot of training steps and your dataset is not as good as MNIST, you should use "math.sqrt(step)" instead of "step" in this formula
		lss = (1./(step + 1.)) * REPULSION_INTENSITY 
			
		data[indx] += -np.sum(moves, axis=0) * lss
		
		data[indx] = np.minimum(data[indx],  ones)
		data[indx] = np.maximum(data[indx], -ones)
		
		indx += 1
		
	print ('Move data time:', time.time()-start)
	

train_data, test_data = get_data()

latent_target = set_latent_target((train_data.shape[0], LATENT_SPACE_DIM))

autoencoder, decoder, encoder = set_models()

leaning_time = time.time()
step = 0
max_step = 10
while step < max_step:
	if COMMON_FIT_METHOD:
		autoencoder.fit(train_data, train_data, epochs=1, batch_size=1024, shuffle=True)	
		
	else:
		#It's absolutely not useful to fit encoder by the way. It will converge very fast to totally wrong results
		decoder.fit (latent_target, train_data, epochs=1, batch_size=1024, shuffle=True)
		autoencoder.fit(train_data, train_data, epochs=1, batch_size=1024, shuffle=True)
			
		if step < max_step-1:
			latent_target = encoder.predict(train_data) 
			stretch_data(latent_target, step)
	
	step += 1
	
	save_results(step)	
	print('Step: ', step)
	print()
	
print ('For ',step,' steps learning time:', time.time()-leaning_time)	


#make image from latent space
X,Y = (np.mgrid[-30:30,-30:30] + 0.5) / 30.
grid = np.vstack((X.flatten(), Y.flatten())).T

X,Y = np.mgrid[0:60,0:60] * 28
box = np.vstack((X.flatten(), Y.flatten())).T
	
im_grid = (decoder.predict(grid) * 255.).astype(int)

im_out = Image.new("L", (1680, 1680))

indx = 0
for b in box:
	im_temp = Image.fromarray(im_grid[indx])
	im_out.paste(im_temp, tuple(box[indx]))
	indx += 1
	
im_out.save('latent_space.png','PNG')
