import os
import caffe
import numpy as np

deploy='deploy.prototxt'
caffe_model='train_iter_70000_0.6%_64_clusters.caffemodel'
file_name='val_random.txt'
mean_file='ilsvrc_2012_mean.npy'


file_list=[]
label=[]

file_val=open(file_name)

for line in file_val.readlines():
	path_label=line.strip().split(' ')
	file_path=path_label[0]
	file_label=path_label[1]
	file_list.append(file_path)
	label.append(int(file_label))
file_val.close()

caffe.set_device(0)
caffe.set_mode_gpu()


net = caffe.Net(deploy, caffe.TEST, weights=caffe_model) #load model and network

#image preprocess
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1)) # change the dimenstion order, from (64,64,3) to (3,28,28)
transformer.set_mean('data',np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0)) #change the channel, from RGB to BGR	
	
def test(img):

	
	im = caffe.io.load_image(img)
	net.blobs['data'].data[...]=transformer.preprocess('data',im)

	#inference
	out = net.forward()
	
	prob = net.blobs['prob'].data[0].flatten()
	order = prob.argsort()	
	
	return order[-1]	

results=[]

for i in range(0,len(file_list)):
	img=file_list[i]
	result = test(img)
	print i, ': ', result
	results.append(result)

np.save('results_compress.npy',results)

#results = np.load('results_compress.npy')
a=np.equal(label,results)

print np.where(a==0)


accuracy = np.mean(np.equal(label,results))

print 'accuracy is : ', accuracy

