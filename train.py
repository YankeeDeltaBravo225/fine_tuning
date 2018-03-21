import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks
from conf import conf
import numpy as np
import time

def vgg_model_maker():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(conf.img_width, conf.img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(conf.nb_classes, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    return model


def image_generator():
    """ ディレクトリ内の画像を読み込んでトレーニングデータとバリデーションデータの作成 """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / conf.color_scale,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / conf.color_scale)

    train_generator = train_datagen.flow_from_directory(
        conf.train_data_dir,
        target_size=(conf.img_width, conf.img_height),
        color_mode='rgb',
        classes=conf.classes,
        class_mode='categorical',
        batch_size=conf.batch_size,
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        conf.validation_data_dir,
        target_size=(conf.img_width, conf.img_height),
        color_mode='rgb',
        classes=conf.classes,
        class_mode='categorical',
        batch_size=conf.batch_size,
        shuffle=True)

    return (train_generator, validation_generator)


def dir_maker(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

if __name__ == '__main__':
    start = time.time()

    # 出力用ディレクトリを作成
    dir_maker(conf.log_dir)
    dir_maker(conf.result_dir)

    # モデル作成
    vgg_model = vgg_model_maker()

    # 最後のconv層の直前までの層をfreeze
    for layer in vgg_model.layers[:15]:
        layer.trainable = False

    # 多クラス分類を指定
    vgg_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

    # 画像のジェネレータ生成
    train_generator, validation_generator = image_generator()

    # Tensorboard用Callback生成
    tb_callback = callbacks.TensorBoard(log_dir=conf.log_dir)

    # Fine-tuning
    history = vgg_model.fit_generator(
        train_generator,
        samples_per_epoch=conf.nb_train_samples,
        nb_epoch=conf.nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=conf.nb_validation_samples,
        callbacks=[tb_callback]
        )

    # 重みファイルを出力
    vgg_model.save_weights(os.path.join(conf.result_dir, conf.weight_file))

    process_time = (time.time() - start) / 60
    print(u'学習終了。かかった時間は', process_time, u'分です。')
