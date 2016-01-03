import sys
import numpy as np
import math as mt
from sklearn import preprocessing as pr
from optparse import OptionParser
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#Citing http://www.cs.hmc.edu/~kpang/nn/som.html
#http://www.ai-junkie.com/ann/som/som1.html for pseudocode.
#If learning rate or neighborhood radius decreases too much, add a cap.

parser = OptionParser()
parser.add_option("-r","--radius",type="int",help="enter a number for the starting radius for the neighboring neurons. By default it will be equal to the minimum of the two dimensions of your neuron lattice.",dest="radius")
parser.add_option("-n","--number",type="int",help="enter the number of neurons desired for self organizing map.",dest="number")
parser.add_option("-e","--epoch",type="int",help="enter the number of iterations desired for self organizing map.",dest="epoch")
parser.add_option("-l","--learn-rate",help="enter a number for the learning rate of the self organizing map.",dest="learn")
parser.add_option("-i","--input-vector",help="enter an input vector of data. required.",dest="vectors")
parser.add_option("-o","--output-vector",help="enter an output file name. will output to a file called 'som_bins.txt' by default in the current path.",dest="out")
parser.add_option("-t","--test",action="store_true",help="if given, program will only print the first 4 vectors in the input so you can see your data as the program is about to read it.",dest="test")
parser.add_option("-c","--cluster",type="int",help="enter a radius for the number of neurons to include that are closest to your bestmatch for each vector",dest="cluster")

options,arguments = parser.parse_args()
if not options.vectors:
	parser.error("Need an input vector!")
else:
	file = open(options.vectors)
	windows = []
	vectors = []
	for i,vector in enumerate(file):
		if not vector.startswith('%'):
			vector = vector.split()
			if 'A' not in vector[0]:
				windows.append(vector[0])
				vector = [float(i) for i in vector[1::]]
				vectors.append(vector)
	
	file.close()
	vectors_prior = vectors
	scaler = pr.MinMaxScaler()
	vectors = scaler.fit_transform(vectors)
	if options.test:
		print vectors_prior[0][0:10]
		print '\n'
		print vectors[0][0:10]
		sys.exit("Done testing.")
if options.cluster:
	if options.cluster < 1:
		parser.error("radius should be at least 1!")
if options.number:
	num = options.number
else:
	num = len(vectors)*5.5
if options.radius:
	rad = options.radius
else:
	rows = int(np.sqrt(num/float(2))+0.5)
	cols = 2*rows
	rad = np.max([rows,cols])
if options.epoch:
	epo = options.epoch
else:
	epo = 100
if options.learn:
	rate = float(options.learn)
else:
	rate = 0.05
if options.out:
	outfile = open(options.out,"w")
else:
	outfile = open("som_bins.txt","w")
	
class neuron:
	def __init__(self,dim,x,y):
		'''
		weights: the weights of the vector for this neuron.
		x: the position along the x axis on the lattice.
		y: the position along the y axis on the lattice.
		'''
		self.weights = np.random.random(dim)
		self.bestmatch = None
		self.x = x
		self.y = y

def plotter(winner,nbr_rad,neurons,i,rows,cols):
	neuron_coord = [[],[]]
	winner_radius = [[],[]]
	for neuron in neurons:
		neuron_coord[0].append(neuron.y+1)
		neuron_coord[1].append(neuron.x+1)
		if np.sqrt((neuron.x-winner.x)**2+(neuron.y-winner.y)**2) < nbr_rad:
			winner_radius[0].append(neuron.y+1)
			winner_radius[1].append(neuron.x+1)
	plt.plot(neuron_coord[0],neuron_coord[1],'b.',winner_radius[0],winner_radius[1],'g.',markersize=8)
	plt.title("The Neuron Lattice")
	plt.axis([0,cols*1.1,0,rows*1.1])
	plt.savefig('frame_'+str(i)+'.png',dpi=1000)
	plt.clf()
	
def bestmatch(neurons,vector,ind):
	dist = [0]*len(neurons)
	for i,neuron in enumerate(neurons):
		for j in xrange(len(vector)):
			dist[i] += (float(vector[j])-neuron.weights[j])**2
		dist[i] = np.sqrt(dist[i])
	max_idx = np.argmin(dist)
	neurons[max_idx].bestmatch = ind # Save the best match for this neuron so that it can be clustered later.
	return max_idx

def adjust(winner,neurons,nbr_rad,lrn_rate,vector,rows,cols):
	count = 0
	timer = 5
	for i,neuron in enumerate(neurons):
		central = (neuron.x-winner.x)**2+(neuron.y-winner.y)**2 # Calculates all distances in the torus.
		toroidx = (neuron.x+rows-winner.x)**2+(neuron.y-winner.y)**2
		toroidxy = (neuron.x+rows-winner.x)**2+(neuron.y+cols-winner.y)**2
		toroid_x = (neuron.x-rows-winner.x)**2+(neuron.y-winner.y)**2
		toroid_x_y = (neuron.x-rows-winner.x)**2+(neuron.y-cols-winner.y)**2
		toroidy = (neuron.x-winner.x)**2+(neuron.y+cols-winner.y)**2
		toroid_y = (neuron.x-winner.x)**2+(neuron.y-cols-winner.y)**2
		toroidx_y = (neuron.x+rows-winner.x)**2+(neuron.y-cols-winner.y)**2
		toroid_xy = (neuron.x-rows-winner.x)**2+(neuron.y+cols-winner.y)**2
		dist_sq = np.min([central,toroidx,toroidxy,toroid_x,toroid_x_y,toroidy,toroid_y,toroidx_y,toroid_xy]) # For calculating influence of the neighborhood radius.
		if dist_sq < nbr_rad**2:
			count += 1
			infl = mt.exp(-dist_sq/float(2*(nbr_rad**2)))
			#if timer >= 0:
				#print "neuron before: "+str(neurons[i].weights[1:10])
			neurons[i].weights = neurons[i].weights+(infl*lrn_rate*(np.array(vector).astype(np.float)-np.array(neurons[i].weights))).tolist()
			#if timer >= 0:
				#print "neuron after: "+str(neurons[i].weights[1:10])
			timer -= 1
	print "neurons adjusted: "+str(count)

def outlier(dists): 
	d = np.abs(dists - np.median(dists))
	q_1 = np.percentile(dists,25)
	q_3 = np.percentile(dists,75)
	IQR = q_3 - q_1
	if dists[-1] > q_3+1.5*IQR:
		return True
	else:
		return False

def cluster(neurons,vectors,windows):
	'''
	Repeats best match search except for all input vectors.
	'''
	print "clustering and writing to file now..."
	radius = 1
	prev_cluster = []
	prev_bestmatch = 0
	if options.cluster:
		radius = options.cluster
	for i in prange(len(vectors)):
		cluster_row=[windows[i],[]]
		dist = [0]*len(neurons)
		for j in prange(len(neurons)):
			for x in prange(len(vectors[i])):
				dist[j] += (float(vectors[i][x])-neurons[j].weights[x])**2
			dist[j] = np.sqrt(dist[j])
		dist = np.array(dist)
		cluster = np.argsort(dist)[0:radius]
		bestmatch = cluster[0]
		cluster_row[1] = str(bestmatch)
		#if not outlier(dist[list(cluster)]):
		#	if bestmatch in prev_cluster:
		#		cluster_row[1] = str(prev_bestmatch)
		#prev_cluster = cluster
		#prev_bestmatch = bestmatch
		print cluster_row
		outfile.write(str(cluster_row)+'\n')

def som(num,rad,vectors,epo,rate,windows):
	neurons = []
	###initialize weights
	rows = int(np.sqrt(num/float(2))+0.5)
	cols = 2*rows
	for i in xrange(rows): 
		for j in xrange(cols):
			neurons.append(neuron(len(vectors[0]),i,j))
	###training epochs
	print "starting training..."
	for i in xrange(epo):
		ind = np.random.randint(0,len(vectors))
		vector = vectors[ind]
		winner = neurons[bestmatch(neurons,vector,ind)]
		time_const = epo/mt.log(rad)
		nbr_rad = rad*mt.exp(-i/float(time_const))
		print "current neighborhood radius: "+str(nbr_rad)+"..."
		lrn_rate = rate*mt.exp(-i/float(epo))
		print "current learning rate: "+str(lrn_rate)+"..."
		adjust(winner,neurons,nbr_rad,lrn_rate,vector,rows,cols)
		print "epochs remaining: "+str(epo-i)+"..."
		plotter(winner,nbr_rad,neurons,i,rows,cols)
	cluster(neurons,vectors,windows)

som(num,rad,vectors,epo,rate,windows)	
outfile.close()
