def load1():
    import numpy as np
    from sklearn.model_selection import train_test_split

    data = np.load('olivetti_faces.npy')
    target = np.load('olivetti_faces_target.npy')
    # data = data*2 - 1

    classes = len(np.unique(target))

    latent_dim = 100
    channel = 1

    data = data.reshape(data.shape[0],data.shape[1],data.shape[2],channel)
    target_onehot = np.zeros((np.shape(target)[0], classes))

    for i in range(0,np.shape(target)[0]):
        target_onehot[i][target[i]] = 1

    X_train, X_test, y_train, y_test = train_test_split(data, target,test_size=0.2, shuffle=True)

def load2():
    import numpy as np
    from sklearn.model_selection import train_test_split

    data = np.load('olivetti_faces.npy')
    target = np.load('olivetti_faces_target.npy')

    channel = 1
    real_face = 10

    for idx, num in enumerate(target):
        if num >= real_face:
            target[idx] = real_face

    data = data.reshape(data.shape[0],data.shape[1],data.shape[2],channel)

    auth = []
    for label in target:
        if label < real_face:
            auth.append(True)
        else:
            auth.append(False)
    auth = np.array(auth)
        
    X_train, X_test, y_label_train, y_label_test, y_auth_train, y_auth_test = train_test_split(data, target, auth,test_size=0.2, shuffle=True)
