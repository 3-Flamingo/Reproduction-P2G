import pickle
from new_utils import *

# a=pickle.load(open("/home/chenzhen/work/P2G/data/original_basic_train_bertdata.pickle",'rb'))
# print(a[:10])

train_data = Data_data(
            pickle.load(open('/home/chenzhen/work/P2G/data/original_basic_train_bertdata.pickle', 'rb')),
            batch_size=32, debug=True, percent=1.0)
while train_data.epoch_flag:
    data = train_data.next_batch(10)
    inputs, targets, token_masks = data[0], data[1], data[2]
print(inputs)