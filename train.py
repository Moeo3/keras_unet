from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import data, model

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def unet_train(unet, features, labels):
    unet.compile(
        optimizer = Adam(),
        loss = dice_coef_loss,
        metrics = ['accuracy']
    )
    unet.fit(
        features, labels,
        # batch_size = 1,
        verbose = 2,
    )
    return unet

if __name__ == "__main__":
    features, labels = data.read_train_data()
    unet_train(model.UNet(), features, labels)
    pass