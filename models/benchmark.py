import numpy as np

def naive_predict(train_y, test_y, fh):
    pred_y = np.asarray([train_y[-1]]*fh) 
    return pred_y