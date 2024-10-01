from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate, Activation, Multiply
from keras.models import Model

def attention_block(g, x):
    theta_x = Conv2D(1, 1)(x)
    phi_g = Conv2D(1, 1)(g)
    attn = Activation('sigmoid')(theta_x + phi_g)
    return Multiply()([x, attn])

def build_attention_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Attention block
    attn1 = attention_block(conv1, pool1)
    
    # Decoder
    up1 = UpSampling2D(size=(2, 2))(attn1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(up1)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv2)
    return Model(inputs=[inputs], outputs=[outputs])

model = build_attention_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
