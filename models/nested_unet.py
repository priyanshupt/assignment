from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model

def build_nested_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Additional layers for Nested U-Net would go here
    # More blocks with skip connections
    
    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(up1)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv2)
    return Model(inputs=[inputs], outputs=[outputs])

model = build_nested_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
