import os, sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras import optimizers
from conf import conf
from PIL import Image
import matplotlib.pyplot as plt

def model_load():
    # VGG16, FC層は不要なので include_top=False
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

    # 学習済みの重みをロード
    model.load_weights(os.path.join(conf.result_dir, conf.weight_file))

    # 多クラス分類を指定
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

    return model


if __name__ == '__main__':

    # テストしたい画像を格納したディレクトリを引数で指定
    test_data_dir = word = sys.argv[1]

    # テスト用画像取得
    test_imagelist = os.listdir(test_data_dir)

    # モデルのロード
    model = model_load()

    for test_image in test_imagelist:
        filename = os.path.join(test_data_dir, test_image)
        raw_img = image.load_img(filename, target_size=(conf.img_width, conf.img_height))
        img_array = image.img_to_array(raw_img)
        expanded_image = np.expand_dims(img_array, axis=0)
        # 学習時に正規化してるので、ここでも正規化
        normalized_image = expanded_image / conf.color_scale
        pred_result = model.predict(normalized_image)[0]

        # 予測確率を出力
        print('=======================================')
        print('file name:', test_image)
        result_msgs = ['%s:%.2f%s' % (conf.classes[i], pred_result[i] * 100, '%') for i in range(len(pred_result))]
        print(result_msgs)

        # 画像を表示
        plt.title(', '.join(result_msgs))
        plt.imshow(np.asarray(image.load_img(filename)))
        plt.show()
