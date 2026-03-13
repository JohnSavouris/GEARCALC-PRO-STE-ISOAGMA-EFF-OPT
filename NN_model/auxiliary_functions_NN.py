import numpy as np
from keras.layers import Input, Lambda, concatenate, Dense, Dropout, LSTM, BatchNormalization
from keras.initializers import glorot_normal
from keras.models import Model, Sequential
from keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


def create_NN_features(xvals,features,periodic_features=False,flip_features=False,\
                       iflip1=[],iflip2=[],flip_flags=[]):
    # xvals -> [ncurve,npoint]
    # features -> [ncurve,nfeat]
    ncurve=xvals.shape[0]
    npoint=xvals.shape[1]
    nfeat=features.shape[1]
    # X related features are always added at the end of the array, so as not
    # to create confusion with indexing related to the actual features
    # (for example used for siamese networks)
    if flip_features:
        inverted_indexes=list(range(nfeat))
        #print(inverted_indexes)
        for i in range(len(iflip1)):
            inverted_indexes[iflip1[i]]=iflip2[i]
            inverted_indexes[iflip2[i]]=iflip1[i]
        # for i in iflip2:
        #     inverted_indexes[i]=iflip1[i]
        #print(inverted_indexes)
    if periodic_features:
        NN_features=np.zeros((ncurve*npoint,nfeat+2))
        for i in range(ncurve):
            if flip_features and flip_flags[i]:
                # flipping x coordinates to "scan STE backwards"
                NN_features[i*npoint:(i+1)*npoint,-1]=np.sin(2*np.pi*(1-xvals[i,:]))
                NN_features[i*npoint:(i+1)*npoint,-2]=np.cos(2*np.pi*(1-xvals[i,:]))
                # This assumes that x coordinates are expressed in a dimensionless
                # manner and range from 0 to 1
                
                # flipping features corresponding to 1 and 2
                NN_features[i*npoint:(i+1)*npoint,:-2]=np.repeat(features[[i],[inverted_indexes]], npoint, axis=0)
            else:
                NN_features[i*npoint:(i+1)*npoint,-1]=np.sin(2*np.pi*xvals[i,:])
                NN_features[i*npoint:(i+1)*npoint,-2]=np.cos(2*np.pi*xvals[i,:])
                # This assumes that x coordinates are expressed in a dimensionless
                # manner and range from 0 to 1
                NN_features[i*npoint:(i+1)*npoint,:-2]=np.repeat(features[[i],:], npoint, axis=0)
    else:
        NN_features=np.zeros((ncurve*npoint,nfeat+1))
        for i in range(ncurve):
            if flip_features and flip_flags[i]:
                # flipping x coordinates to "scan STE backwards"
                NN_features[i*npoint:(i+1)*npoint,-1]=1-xvals[i,:]#np.swapaxes(xvals[i,:],0,1)
                # flipping features corresponding to 1 and 2
                NN_features[i*npoint:(i+1)*npoint,:-1]=np.repeat(features[[i],[inverted_indexes]], npoint, axis=0)
            else:
                NN_features[i*npoint:(i+1)*npoint,-1]=xvals[i,:]#np.swapaxes(xvals[i,:],0,1)
                NN_features[i*npoint:(i+1)*npoint,:-1]=np.repeat(features[[i],:], npoint, axis=0)
    
    return NN_features

# This function was left here to be able to revert to versions before 09/06/2022
def create_NN_features_bkp(xvals,features,periodic_features=False):
    # xvals -> [ncurve,npoint]
    # features -> [ncurve,nfeat]
    ncurve=xvals.shape[0]
    npoint=xvals.shape[1]
    nfeat=features.shape[1]
    # X related features are always added at the end of the array, so as not
    # to create confusion with indexing related to the actual features
    # (for example used for siamese networks)
    if periodic_features:
        NN_features=np.zeros((ncurve*npoint,nfeat+2))
        for i in range(ncurve):
            NN_features[i*npoint:(i+1)*npoint,-1]=np.sin(2*np.pi*xvals[i,:])
            NN_features[i*npoint:(i+1)*npoint,-2]=np.cos(2*np.pi*xvals[i,:])
            # This assumes that x coordinates are expressed in a dimensionless
            # manner and range from 0 to 1
            NN_features[i*npoint:(i+1)*npoint,:-2]=np.repeat(features[[i],:], npoint, axis=0)
    else:
        NN_features=np.zeros((ncurve*npoint,nfeat+1))
        for i in range(ncurve):
            NN_features[i*npoint:(i+1)*npoint,-1]=xvals[i,:]#np.swapaxes(xvals[i,:],0,1)
            NN_features[i*npoint:(i+1)*npoint,:-1]=np.repeat(features[[i],:], npoint, axis=0)
    
    return NN_features

def flip_or_noflip(featvals,indexes1,indexes2):
    if len(indexes1): # the list still has elements
        # print(featvals[indexes1[0]])
        # print(featvals[indexes2[0]])
        # print(' ')
        if featvals[indexes1[0]] == featvals[indexes2[0]]:
            flip_or_noflip(featvals,indexes1[1:],indexes2[1:])
            # tie is to be resolved by the next feature in recursive fashion
        elif featvals[indexes1[0]] > featvals[indexes2[0]]:
            return True
        else:
            return False
    else: # no feature has broken the tie, so just use False
        return False

def convert_features_to_symmetric(featvals,indexes1,indexes2):
    # featvals -> [ncurve*npoint,nfeat_NN]
    featvals_new=featvals#np.zeros(featvals.shape)
    for i in range(len(indexes1)):
        featvals_new[:,indexes1[i]]=(featvals[:,indexes1[i]]+featvals[:,indexes2[i]])/2
        featvals_new[:,indexes2[i]]=np.abs(featvals[:,indexes1[i]]-featvals[:,indexes2[i]])
        # the above definition keeps outputs within the interval [-1, 1] when the inputs
        # originate from the interval [-1, 1]
    
    return featvals_new

def create_NN_output(yvals):
    # yvals -> [ncurve,npoint]
    ncurve=yvals.shape[0]
    npoint=yvals.shape[1]
    
    NN_output=np.zeros((ncurve*npoint,1))
    for i in range(ncurve):
        NN_output[i*npoint:(i+1)*npoint,0]=yvals[i,:]
    
    return NN_output

def subsample_curve(xvals,yvals,subsample_factor,random_subsampling=False):
    ncurve=xvals.shape[0]
    npoint=xvals.shape[1]
    if random_subsampling:
        xsubsampled=np.zeros((ncurve,npoint//subsample_factor))
        ysubsampled=np.zeros((ncurve,npoint//subsample_factor))
        for i in range(ncurve):
            rand_loc = np.random.RandomState(i)
            # in this way, rand_loc takes the place of np.random locally and results can be reproduced
            subsampling_indexes=rand_loc.choice(npoint, npoint//subsample_factor, replace=False)
            xsubsampled[i,:]=xvals[i,subsampling_indexes]
            ysubsampled[i,:]=yvals[i,subsampling_indexes]
    else:
        subsampling_indexes=np.arange(0, npoint, subsample_factor, dtype=int)
        xsubsampled=xvals[:,subsampling_indexes]
        ysubsampled=yvals[:,subsampling_indexes]
    
    return xsubsampled, ysubsampled

def create_fully_connected_network(input_size=3, hidden_size_start=60, hidden_size_end=60, nlayer_hidden=10, output_size=3, output_activation="linear"):
    
    init_fc = glorot_normal()
    
    layer_sizes=np.linspace(hidden_size_start, hidden_size_end, nlayer_hidden, dtype='int64')
    input_layer = Input(shape=(input_size,) )
    for i_layer in range(nlayer_hidden):
        # x = BatchNormalization()(x)
        if i_layer==0:
            x = Dense(layer_sizes[i_layer], activation="tanh", kernel_initializer=init_fc)(input_layer)
        else:
            x = Dense(layer_sizes[i_layer], activation="tanh", kernel_initializer=init_fc)(x)

    x = Dense(output_size, activation=output_activation, kernel_initializer=init_fc)(x)
    
    model = Model(inputs=input_layer, outputs=x)
    return model
    
def build_model(nfeat_fc=3, size_fc_start=60, size_fc_end=60, nlayer_fc=10):
    
    total_input = Input(shape=(nfeat_fc,))
    total_network = create_fully_connected_network\
    (nfeat_fc, size_fc_start, size_fc_end, nlayer_fc, 1, "linear")
    total_output = total_network(total_input)
    
    model = Model(inputs=total_input, outputs=total_output)
    
    model.compile(loss='mean_absolute_percentage_error', optimizer=Adam()) #'mean_absolute_error'

    #model.compile(optimizer=SGD(learning_rate=0.01), loss='mean_squared_error', metrics=['accuracy'])

    
    return model
