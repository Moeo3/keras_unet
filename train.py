from keras.models import Model
import data, model

def unet_train(unet, features, labels):
    
    pass

if __name__ == "__main__":
    features, labels = data.read_train_data()
    unet_train(model.UNet(), features, labels)
    pass