import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout,BatchNormalization,AveragePooling2D,concatenate,Input, concatenate,add,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.optimizers import Adam

#Define convolution with batchnromalization
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x

  
#Define Inception structure
def Inception(x,nb_filter_para):
    (branch1,branch2,branch3,branch4)= nb_filter_para
    branch1x1 = Conv2D(branch1[0],(1,1), padding='same',strides=(1,1),name=None)(x)

    branch3x3 = Conv2D(branch2[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch3x3 = Conv2D(branch2[1],(3,3), padding='same',strides=(1,1),name=None)(branch3x3)

    branch5x5 = Conv2D(branch3[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch5x5 = Conv2D(branch3[1],(1,1), padding='same',strides=(1,1),name=None)(branch5x5)
    
    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = Conv2D(branch4[0],(1,1),padding='same',strides=(1,1),name=None)(branchpool)
    
    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x
  
#Build InceptionV1 model
def D_InceptionV1(width, height, depth):
    
    inpt = Input(shape=(width,height,depth))

    x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Conv2d_BN(x,64,(3,3),strides=(1,1),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

    x = Inception(x,[(64,),(96,128),(16,32),(32,)]) #Inception 3a 28x28x256
    x = Inception(x,[(128,),(128,192),(32,96),(64,)]) #Inception 3b 28x28x480
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x) #14x14x480

    x = Inception(x,[(192,),(96,208),(16,48),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(160,),(112,224),(24,64),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(128,),(128,256),(24,64),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(112,),(144,288),(32,64),(64,)]) #Inception 4a 14x14x528
    x = Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 4a 14x14x832
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x) #7x7x832

    x = Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 5a 7x7x832
    x = Inception(x,[(384,),(192,384),(48,128),(128,)]) #Inception 5b 7x7x1024

    #Using AveragePooling replace flatten
    x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation='relu')(x)
       
    return x

#Define Residual Block for ResNet34(2 convolution layers)
def Residual_Block(input_model,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut =False):
    x = Conv2d_BN(input_model,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    
    #need convolution on shortcut for add different channel
    if with_conv_shortcut:
        shortcut = Conv2d_BN(input_model,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,input_model])
        return x

def D_Resnet(width, height, depth):
    Img = Input(shape=(width,height,depth))
    x = Conv2d_BN(Img,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  

    #Residual conv2_x ouput 56x56x64 
    x = Residual_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=64,kernel_size=(3,3))
    
    #Residual conv3_x ouput 28x28x128 
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)# need do convolution to add different channel
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3))
    
    #Residual conv4_x ouput 14x14x256
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)# need do convolution to add different channel
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    
    #Residual conv5_x ouput 7x7x512
    x = Residual_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Residual_Block(x,nb_filter=512,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=512,kernel_size=(3,3))

    #Using AveragePooling replace flatten
    x = GlobalAveragePooling2D()(x)

    return x  

def paper_D(inpt_layer):
    x = Conv2D(32, kernel_size=5)(inpt_layer)
#     x = Conv2D(32, kernel_size=5,padding='same')(inpt_layer) # 
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(3,3))(x)
    x = Conv2D(64, kernel_size=5)(x)
#     x = Conv2D(64, kernel_size=5,padding='same')(inpt_layer) # 
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Conv2D(128, kernel_size=5)(x)
#     x = Conv2D(128, kernel_size=5,padding='same')(inpt_layer) # 
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(400, activation='tanh')(x)

    return x

def paper_D_mnist(inpt_layer):
    x = Conv2D(32, kernel_size=5)(inpt_layer)
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(3,3))(x)
    x = Conv2D(64, kernel_size=5)(x)
    x = Conv2D(64, kernel_size=5,padding='same')(x) #
    x = Conv2D(64, kernel_size=5,padding='same')(x) #
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(200, activation='tanh')(x)

    return x

def D(width, height, channel, classes):
    # change different Discriminator here
    import Discriminator
    from keras.layers import Dense,Input
    from keras.models import Model
    width = 64
    height = 64
    channel = 1
    classes = 41
    
    Img = Input(shape=(width,height,channel))
    x = Discriminator.paper_D(Img)

    x1 = Dense(classes,activation='softmax',name='label')(x)
    x2 = Dense(1, activation='sigmoid', name='auth')(x)
    
    D = Model(name='Discriminator' ,inputs= Img, outputs=[x1, x2])
    
def compile(D):
    from keras.optimizers import SGD
    D.trainable = True
    D.compile(    loss= ['sparse_categorical_crossentropy','binary_crossentropy'],
                                    optimizer= SGD(lr=0.02, decay=1e-7) ,
                                    metrics=['accuracy'])
    
    
def paper_D_cut1(inpt_layer):
    x = Conv2D(32, kernel_size=5)(inpt_layer)
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(3,3))(x)
    x = Conv2D(64, kernel_size=5)(x)
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Conv2D(128, kernel_size=5)(x)
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    return x

def paper_D_cut2(x):
    x = Flatten()(x)
    x = Dense(400, activation='tanh')(x)
    return x

def paper_D_cut11(inpt_layer):
    x = Conv2D(32, kernel_size=5)(inpt_layer)
    return x

def paper_D_cut22(x):
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(3,3))(x)
    x = Conv2D(64, kernel_size=5)(x)
    return x

def paper_D_cut33(x):
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Conv2D(128, kernel_size=5)(x)
    x = Conv2D(128, kernel_size=5 ,padding = 'same')(x) #
    x = Conv2D(128, kernel_size=5 ,padding = 'same')(x) #
    x = Activation("tanh")(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(400, activation='tanh')(x)
    return x
