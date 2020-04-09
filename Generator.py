from keras.layers import Dense, Activation, Dropout, Conv2D, Input, Reshape, UpSampling2D ,concatenate ,BatchNormalization, Embedding, MaxPooling2D,Conv2DTranspose ,Flatten, multiply
from keras.models import Model

def deConv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = BatchNormalization(axis=3,name=bn_name)(x)
    x = Conv2DTranspose(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    return x

def unInception(x,nb_filter_para):
    (branch1,branch2,branch3,branch4)= nb_filter_para
    branch1x1 = Conv2DTranspose(branch1[0],(1,1),padding='same',strides=(1,1),name=None)(x)

    branch3x3 = Conv2DTranspose(branch2[1],(3,3),padding='same',strides=(1,1),name=None)(x)
    branch3x3 = Conv2DTranspose(branch2[0],(1,1),padding='same',strides=(1,1),name=None)(branch3x3)

    branch5x5 = Conv2DTranspose(branch3[1],(1,1),padding='same',strides=(1,1),name=None)(x)
    branch5x5 = Conv2DTranspose(branch3[0],(1,1),padding='same',strides=(1,1),name=None)(branch5x5)

    branchpool = Conv2DTranspose(branch4[0],(1,1),padding='same',strides=(1,1),name=None)(x)
    # branchpool = MaxPooling2D(pool_size=(1,1),strides=(1,1),padding='same')(branchpool)
    
    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)
    return x

def unInceptionV1(inpt_layer):

    x = Dense(units=1000, activation= 'relu')(inpt_layer)
    x = Dense(units=1024*1*1, activation= 'relu')(x)
    x = Dropout(0.2)(x)
    x = Reshape(target_shape=(1, 1 ,1024))(x)
    x = UpSampling2D(size=(7, 7))(x)
    
    x = unInception(x,[(384,),(192,384),(48,128),(128,)]) #Inception 5b 7x7x1024
    x = unInception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 5a 7x7x832
    x = UpSampling2D(size=(2, 2))(x)
    
    x = unInception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 4a 14x14x832
    x = unInception(x,[(112,),(144,288),(32,64),(64,)]) #Inception 4a 14x14x528
    x = unInception(x,[(128,),(128,256),(24,64),(64,)]) #Inception 4a 14x14x512
    x = unInception(x,[(160,),(112,224),(24,64),(64,)]) #Inception 4a 14x14x512
    x = unInception(x,[(192,),(96,208),(16,48),(64,)]) #Inception 4a 14x14x512
    x = UpSampling2D(size=(2, 2))(x)

    x = unInception(x,[(128,),(128,192),(32,96),(64,)]) #Inception 3b 28x28x480
    x = unInception(x,[(64,),(96,128),(16,32),(32,)]) #Inception 3a 28x28x256
    x = UpSampling2D(size=(2, 2))(x)

    x = deConv2d_BN(x,192,(3,3),strides=(1,1),padding='same')
    x = UpSampling2D(size=(2, 2))(x)
    x = deConv2d_BN(x,64,(7,7),strides=(2,2),padding='same')

    x = BatchNormalization(axis=3,name='lastnormalize')(x)
    x = Conv2DTranspose(3,kernel_size=(1,1),padding='same',strides=(1,1),activation='tanh',name='draw')(x)
     
    return x

def unVGG(inpt_layer):
    num = 256
    normalize_momentum = 0.8

    x = Dense(units=1000, activation= 'elu')(inpt_layer)
    x = Dropout(0.4)(x)
    x = Dense(units=4096, activation='elu')(x)
    x = Dropout(0.4)(x)
    x = Dense(units=8*8*num, activation='elu')(x)
    x = Reshape((8, 8, num))(x)

#     x = BatchNormalization(momentum=normalize_momentum)(x)
#     x = UpSampling2D(size=(2,2))(x)
#     x = Conv2DTranspose(filters=num, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
#     x = Conv2DTranspose(filters=num, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)
#     x = Conv2DTranspose(filters=num, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(x)

    x = BatchNormalization(momentum=normalize_momentum)(x)
#     x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(filters=num, kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)
    x = Conv2DTranspose(filters=num, kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)
    x = Conv2DTranspose(filters=num, kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)

    x = BatchNormalization(momentum=normalize_momentum)(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(filters=int(num/2), kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)
    x = Conv2DTranspose(filters=int(num/2), kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)
    x = Conv2DTranspose(filters=int(num/4), kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)

    x = BatchNormalization(momentum=normalize_momentum)(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(filters=int(num/4), kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)
    x = Conv2DTranspose(filters=int(num/8), kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)

    x = BatchNormalization(momentum=normalize_momentum)(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(filters=int(num/8), kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)

    x = Conv2DTranspose(filters=1, kernel_size=3, padding='same', activation='tanh')(x)
    
    return x

def paper_G(inpt_layer):
    x = Dense((4*4*512), activation="relu", input_dim=100)(inpt_layer)
    x = Reshape((4,4,512))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(256, kernel_size=4,strides=2, padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(128, kernel_size=4,strides=2, padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(64, kernel_size=4,strides=2, padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(1, kernel_size=4,strides=2, padding='same',use_bias=False)(x)

    return x

def paper_G_mnist(inpt_layer):
    x = Dense((7*7*256), activation="relu", input_dim=100)(inpt_layer)
    x = Reshape((7,7,256))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(128, kernel_size=4,strides=2, padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(64, kernel_size=4,strides=2, padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(1, kernel_size=4,strides=1, padding='same',use_bias=False)(x)
    x = Activation("tanh")(x)

    return x

def G1(noise):
    import Generator
    from keras.layers import Dense,Input
    from keras.models import Model
    
    inpt1 = Input(shape=(noise_size,))
    x = Generator.paper_G(inpt1)

    G = Model(name='Paper Generator', inputs=[inpt1], outputs=x)

def G2(noise,classes):
    inpt1 = Input(shape=(noise,))
    inpt2 = Input(shape=(1,), dtype='int32')

    # change different Generator here
    x = Embedding(classes,noise,trainable=False)(inpt2)
    x = Flatten()(x)
    x = multiply([x, inpt1])

    x = paper_G(x)

    model = Model(name='Generator', inputs=[inpt1, inpt2], outputs=x)
    return model

def paper_G_deep(inpt_layer):
    x = Dense((4*4*512), activation="relu", input_dim=100)(inpt_layer)
    x = Reshape((4,4,512))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(256, kernel_size=4,strides=2, padding='same',use_bias=False)(x)
    x = Conv2DTranspose(256, kernel_size=4,strides=1, padding='same',use_bias=False)(x)#
    x = Conv2DTranspose(256, kernel_size=4,strides=1, padding='same',use_bias=False)(x)#
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(128, kernel_size=4,strides=2, padding='same',use_bias=False)(x)
    x = Conv2DTranspose(128, kernel_size=4,strides=1, padding='same',use_bias=False)(x)#
    x = Conv2DTranspose(128, kernel_size=4,strides=1, padding='same',use_bias=False)(x)#
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(64, kernel_size=4,strides=2, padding='same',use_bias=False)(x)
    x = Conv2DTranspose(64, kernel_size=4,strides=1, padding='same',use_bias=False)(x)#
    x = Conv2DTranspose(64, kernel_size=4,strides=1, padding='same',use_bias=False)(x)#
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(1, kernel_size=4,strides=2, padding='same',use_bias=False)(x)

    return x