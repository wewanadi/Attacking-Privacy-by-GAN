def save_model(model):
    import os
    model.save('{}//discriminator_untrained'.format(os.getcwd()))


def load_model(model):
    from keras.models import load_model
    import os
    D = load_model('{}/discriminator_trained_SGD2[100]'.format(os.getcwd()))

def combine(D, G):
    from keras.layers import Dense, Activation, Dropout, Conv2D, Input, Reshape, UpSampling2D ,concatenate ,BatchNormalization, Embedding, MaxPooling2D,Conv2DTranspose ,Flatten, multiply
    from keras.models import Model
    from keras.models import Sequential, Model
    from keras.optimizers import Adam, Nadam ,Adamax , SGD
    noise_shape = 100

    inpt1 = Input(shape=(noise_shape,))
    inpt2 = Input(shape=(1,))

    img = G([inpt1])
    classes, auth = D(img)

    C =  Model(name='Combined', inputs=[inpt1,inpt2], outputs=[classes, auth])

    C.compile(  loss= ['sparse_categorical_crossentropy'],
                optimizer= Adam(lr=0.00002 , beta_1=0.5, epsilon=1e-08),
                metrics=['accuracy']
                )

def evaluate(model):
    score = model.evaluate (data,target)
    print ('--------------------\nTrain: loss:%f | label acc: %.2f%% \n-------------------'% (score[0],score[1]*100))

def np_show():
    import sys
    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)

def rand():
    import numpy as np
    batch_size = 16
    noise_shape = 100

    noise = np.random.normal(loc=0, scale=1, size=(batch_size, noise_shape))
    label = np.random.randint(low=0, high=10, size=(batch_size, 1))
    idx = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)

def summary(D,G,C):
    D.summary()
    G.summary()
    C.summary()

    from keras.utils.vis_utils import plot_model
    plot_model(D, to_file='Discriminator.png', show_shapes=True, show_layer_names=True)
    plot_model(G, to_file='Generator.png', show_shapes=True, show_layer_names=True)
    plot_model(C, to_file='Combination.png', show_shapes=True, show_layer_names=True)

    from IPython.core.display import Image, display
    display(Image('./Discriminator.png', width=900, unconfined=True))
    display(Image('./Generator.png', width=400, unconfined=True))
    display(Image('./Combination.png', width=400, unconfined=True))

def D_ev(D):
    score = D.evaluate (X_train,[y_label_train,y_auth_train])
    print ('--------------------\nTrain: loss:%f | label acc: %.2f%% | auth acc: %.2f%%\n-------------------'% (score[0],score[3]*100,score[4]*100))

    score = D.evaluate (X_test,[y_label_test,y_auth_test])
    print ('--------------------\nTrain: loss:%f | label acc: %.2f%% | auth acc: %.2f%%\n-------------------'% (score[0],score[3]*100,score[4]*100))

    a = D.predict (X_test[0:1])
    print(a)
    

def train_D(D):
    epoch = 0
    while (1):
        epoch += 1
        batch_size = 16
        idx = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)
        x = X_train[idx]
        y1 = y_label_train[idx]
        y2 = y_auth_train[idx]

        History = D.train_on_batch(x,[y1,y2])
        
        if epoch % 10 == 0:
            print (epoch,History)
        
        if epoch % 100 == 0:
            score = D.evaluate (X_train,[y_label_train,y_auth_train])
            print ('--------------------\nTrain: loss:%f | label acc: %.2f%% | auth acc: %.2f%%\n-------------------'% (score[0],score[3]*100,score[4]*100))

            score = D.evaluate (X_test,[y_label_test,y_auth_test])
            print ('--------------------\nTest: loss:%f | label acc: %.2f%% | auth acc: %.2f%%\n-------------------'% (score[0],score[3]*100,score[4]*100))

            if score[3]>0.99:
                break

def train_G():
    while (1):
        print (count)
        batch_size = 3000
        
        count = count + 1
        noise = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
        label = np.random.randint(low=0, high= 10, size=(batch_size, 1))
        true = np.ones((batch_size, 1))
        
        score = model.combined.fit(
            x=[noise,label], y=[label,true] ,validation_split=0.05, epochs=1, batch_size=8, verbose=1)
        
        model.sample_images('Generator{}'.format(str(count).zfill(6)))

def sample_images(name, G, D):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Dir set
    dirname = 'Generator_Image'
    path = '{}//{}//'.format(os.getcwd(), dirname)
    if not os.path.isdir(path):
        os.mkdir(path)

    r, c = 3, 10
    batch_size = r * c
    noise_shape = 100

    noise = np.random.normal(loc=0, scale=1, size=(batch_size, noise_shape))
    label = np.random.randint(low=0, high= 10, size=(batch_size, 1))
    gen_imgs = G.predict([noise,label])
    ans = D.predict(gen_imgs)

    gen_imgs = gen_imgs.reshape(gen_imgs.shape[0],gen_imgs.shape[1],gen_imgs.shape[2])
    
    # print the answer
    print ('Discrimiating GAN image label')
    for i in range(0, r):
        for j in range(0, c):
            print('%2d' %(np.argmax(ans, axis=1))[i*c+j],end=' ')
        print('')
        
    print ('--------------------\nTrue label') 
               
    gen_imgs = (gen_imgs) / 2
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Generator_Image/{}.png".format(name),dpi=1000)
    plt.close()
    
def mnist_shrink():
    get = 10
    idx = np.arange(0,get)
    for i in range(1,10):
        a = np.arange(i*6000 +2000,i*6000 +2000+get)
        idx = np.append(idx,a)
        
    X_train = X_train[idx]
    y_train = y_train[idx]