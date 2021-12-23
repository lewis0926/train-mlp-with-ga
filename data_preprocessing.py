from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


def pca(x_train, x_test, variance_retained):

    pca = PCA(variance_retained)
    pca.fit(x_train)
    print("Number of PCA components:", pca.n_components_)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print("x_train.shape after PCA:", x_train.shape)
    print("x_test.shape after PCA:", x_test.shape)

    return x_train, x_test


def standardization(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    return scaler.transform(x_train), scaler.transform(x_test)


def normalization(x_train, x_test):
    return normalize(x_train), normalize(x_test)






