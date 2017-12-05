import numpy as np
import math
import caffe
import os
from sklearn.cluster import KMeans
import struct

def initialize_network(deploy, phase, caffemodel=None):
    if caffemodel is None:
        if phase == 'test':
                return caffe.Net(deploy,caffe.TEST)
        else:
            return caffe.Net(deploy, caffe.TRAIN)
    if phase == 'test':
        return caffe.Net(deploy,caffe.TEST,weights=caffemodel)
    else:
        return caffe.Net(deploy,caffe.TRAIN,weights=caffemodel)


def extract(deploy, caffemodel, phase):
     
    prefix = caffemodel.split('.')[0]
    if os.path.exists(prefix+'_layers.npy') and os.path.exists(prefix+'_weights.npz') and os.path.exists(prefix+'_biases.npz'):
        return prefix+'_layers.npy', prefix+'_weights.npz', prefix+'_biases.npz'
    
    dnn = initialize_network(deploy, phase, caffemodel)
    layers = dnn.params.keys()
    for l in layers:
	    print l, dnn.params[l][0].data.shape
    #'conv1', 'conv2', 'ip1', 'ip2'
    np.save(prefix+'_layers.npy',layers)
    
    #print dnn.params['conv1'][0].data.shape
    np.savez(prefix+'_weights.npz', **{l: dnn.params[l][0].data for l in layers})
    np.savez(prefix+'_biases.npz', **{l: dnn.params[l][1].data for l in layers})    
    
    return prefix+'_layers.npy', prefix+'_weights.npz', prefix+'_biases.npz'


def prune_edges_with_small_weight(ndarray,percent):
    weights = ndarray.flatten()
    
    abso = np.absolute(weights)
    threshold = np.sort(abso)[int(math.ceil(weights.size * percent))]
    weights[abso<threshold] = 0
    return weights.reshape(ndarray.shape)
#the purpose of this function is to make every item in 'relative' less than 'max_index', to satisfy the store requirement of 4 bit. Meanwhile, the center labels of the increased place above is set to 0.  
def relative_index(absolute_index, ndarray, max_index):
    first = absolute_index[0]
    relative = np.insert(np.diff(absolute_index), 0, first)
	#np.diff(absolute_index)
	#349 [5 1 1 1 1 2 1 1 1]
    dense = ndarray.tolist()
    max_index_or_less = relative.tolist()
    shift = 0
    
    for i in np.where(relative > max_index)[0].tolist():
        # minus max_index every iteration until max_index_or_less[i] less than max_index 
        while max_index_or_less[i+shift] > max_index:
            max_index_or_less.insert(i+shift,max_index)
            dense.insert(i+shift,0)
            shift+=1
            max_index_or_less[i+shift] -= max_index
    
    return (np.array(max_index_or_less), np.array(dense))


def store_compressed_network(path, layers):
    
    if os.path.exists(path):
        return path
    with open(path, 'wb') as f:
        for layer in layers:
            f.write(struct.pack('Q',layer[1].size))
            f.write(layer[0].tobytes())
            f.write(layer[1].tobytes())
            f.write(struct.pack('Q',layer[2].size))
            f.write(layer[2].tobytes())
            f.write(layer[3].tobytes())
    return path


def compress(target_layers, weights, biases, pruning_percent, cluster_num, store_path_prefix):
    
    #if os.path.exists(store_path_prefix+'_'+str(pruning_percent)+'%_'+str(cluster_num)+'_clusters.npy'):
     #   return store_path_prefix+'_'+str(pruning_percent)+'%_'+str(cluster_num)+'_clusters.npy'
    
    components=[]
    
	
    for l in target_layers:
        
	# pruning
	if 'expand/3x3' in l:
		sparse_1d = prune_edges_with_small_weight(weights[l],pruning_percent).flatten()
        else:
		#sparse_1d = weights[l].flatten() sparse_1d.shape: (500,)
		sparse_1d = prune_edges_with_small_weight(weights[l],pruning_percent*0.5).flatten()
	#K-means
        nonzero = sparse_1d[sparse_1d != 0]
        # nonzero = [1,2,3] nonzero: (350,) -> reshape(-1,1) (350,1)
        print cluster_num, nonzero.size
	#clusters = KMeans(n_clusters=cluster_num).fit(nonzero.reshape(-1,1))
        
	# if raise erro, use this
        clusters = KMeans(n_clusters=cluster_num).fit(nonzero.reshape(-1,1).astype(np.float64))
        
        
        relative_index_in_4bits, cluster_labels = relative_index(np.where(sparse_1d !=0)[0],clusters.labels_+1,max_index=16-1)
        #clusters.labels_:这350个数对应的64个聚类中心的索引值
		#350 [53 25  2 22 51 62 49 12 32 26 31 10  6 51...]
		#np.where(sparse_1d != 0)[0]:
		#350 [  1   6   7   8   9  10  11  ...490 491 493 494 496 497 498 499]
        if relative_index_in_4bits.size % 2 == 1:
            relative_index_in_4bits = np.append(relative_index_in_4bits, 0)
        
        # move the second value of the 'relative_index_in_4bits' 4 bits toward left, and plus the first value of the 'relative_index_in_4bits' to form 1 byte 
        pair_of_4bits_in_1byte = relative_index_in_4bits[np.arange(0, relative_index_in_4bits.size, 2)] * 16 + relative_index_in_4bits[np.arange(1, relative_index_in_4bits.size, 2)]
        

        components.append([pair_of_4bits_in_1byte.astype(np.dtype('u1')), cluster_labels.astype(np.dtype('u1')), clusters.cluster_centers_.astype(np.float32).flatten(), biases[l]])

    return store_compressed_network(store_path_prefix+'_'+str(pruning_percent)+'%_'+str(cluster_num)+'_clusters.npy', components)



def decode(deploy, compressed_network_path,phase,target_layers):
    caffemodel = initialize_network(deploy,phase)
    f = open(compressed_network_path,'rb')

    for l in target_layers:
        edge_num=np.fromfile(f,dtype=np.int64,count=1)
        indices_pair_of_4bits = np.fromfile(f,dtype=np.dtype('u1'),count=int(math.ceil(edge_num[0]/2.0)))
        cluster_labels=np.fromfile(f,dtype=np.dtype('u1'),count=edge_num)
        clusters_num=np.fromfile(f,dtype=np.int64,count=1)
        sharing_weights=np.fromfile(f,dtype=np.float32,count=clusters_num)
        
        biases=np.fromfile(f,dtype=np.float32,count=caffemodel.params[l][1].data.size)
        np.copyto(caffemodel.params[l][1].data,biases)

        relative=np.zeros(indices_pair_of_4bits.size*2,dtype=np.dtype('u1'))
        relative[np.arange(0,relative.size,2)] = indices_pair_of_4bits / 16
        relative[np.arange(1,relative.size,2)] = indices_pair_of_4bits % 16

        if relative[-1] == 0:
            relative = relative[:-1]

        weights = np.zeros(caffemodel.params[l][0].data.size,np.float32)
        index = np.cumsum(relative)
        weights[index] = np.insert(sharing_weights,0,0)[cluster_labels]
        np.copyto(caffemodel.params[l][0].data, weights.reshape(caffemodel.params[l][0].data.shape))
    remain = f.read()
    f.close()
    if len(remain) != 0:
        sys.exit("Decode error!")
    return caffemodel
        



def main(args):
    target_layers_path, weights_path, biases_path = extract(args.deploy, args.caffemodel, args.phase)
    
    target_layers = np.load(target_layers_path)
    
    weights = np.load(weights_path)

    biases = np.load(biases_path)
   

    compressed_network_path = compress(target_layers, weights, biases, float(args.pruning_percent), int(args.cluster_num), args.caffemodel.split('.')[0])
    
    
    compressed_caffemodel = decode(args.deploy, compressed_network_path, args.phase, target_layers)
    
    compressed_caffemodel_path = '.'.join(compressed_network_path.split('.')[:2])+'.caffemodel'

    compressed_caffemodel.save(compressed_caffemodel_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #python easy_deep_compression.py lenet_deploy.prototxt 0.3 256
    parser.add_argument('deploy') #the name of first parameter
    parser.add_argument('caffemodel')
    parser.add_argument('-p', '--phase', default='test')
    parser.add_argument('pruning_percent')
    parser.add_argument('cluster_num')
    args = parser.parse_args()
    

    main(args)


