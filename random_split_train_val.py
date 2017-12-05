import numpy as np
import sys


for i in range(1,len(sys.argv)):

    filename = sys.argv[i]

    train_file = filename.split('.')[0]+'_train.txt'
    val_file = filename.split('.')[0]+'_val.txt'
    
    idx = []
    with open(filename) as f:
        for line in f:
            idx.append(line.strip())
    f.close()
    
    idx = np.random.permutation(idx)

    train_idx = sorted(idx[:int(len(idx)*0.8)])
    val_idx = sorted(idx[int(len(idx)*0.8):])

    with open(train_file,'w') as f:
        for i in train_idx:
            f.write('{}\n'.format(i))
    f.close()

    with open(val_file, 'w') as f:
        for i in val_idx:
            f.write('{}\n'.format(i))
    f.close()

    print 'Training set is save to ' + train_file
    print 'Validation set is saved to ' + val_file
    
