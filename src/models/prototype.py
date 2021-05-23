from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D, concatenate
from keras.models import Model


def create_feature_model(image_shape):
    img_in = Input(shape=image_shape, name='FeatureNet_ImageInput')
    n_layer = img_in
    for i in range(2):
        n_layer = Conv2D(8 * 2 ** i, kernel_size=(3, 3), activation='linear')(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        n_layer = Conv2D(16 * 2 ** i, kernel_size=(3, 3), activation='linear')(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        n_layer = MaxPool2D((2, 2))(n_layer)
    n_layer = Flatten()(n_layer)
    n_layer = Dense(32, activation='linear')(n_layer)
    n_layer = Dropout(0.5)(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)
    feature_model = Model(inputs=[img_in], outputs=[n_layer], name='FeatureGenerationModel')
    return feature_model


def create_model(image_shape):
    img_a_in = Input(shape=image_shape, name='ImageA_Input')
    img_b_in = Input(shape=image_shape, name='ImageB_Input')

    feature_model = create_feature_model(image_shape)

    img_a_feat = feature_model(img_a_in)
    img_b_feat = feature_model(img_b_in)
    combined_features = concatenate([img_a_feat, img_b_feat], name='merge_features')
    combined_features = Dense(16, activation='linear')(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)
    combined_features = Dense(4, activation='linear')(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)
    combined_features = Dense(1, activation='sigmoid')(combined_features)
    similarity_model = Model(inputs=[img_a_in, img_b_in], outputs=[combined_features], name='Similarity_Model')
    similarity_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return similarity_model
