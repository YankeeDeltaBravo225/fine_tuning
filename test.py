import os, sys
import numpy as np
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras import optimizers
from conf import conf
from PIL import Image
import matplotlib.pyplot as plt

def load_model():
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


def load_normalized_image(file_path):
    raw = image.load_img(file_path, target_size=(conf.img_width, conf.img_height))
    array_img = image.img_to_array(raw)
    expanded = np.expand_dims(array_img, axis=0)
    # 学習時に正規化してるので、ここでも正規化
    normalized = expanded / conf.color_scale

    return normalized


if __name__ == '__main__':

    # テストしたい画像を格納したディレクトリを引数で指定
    image_dir = word = sys.argv[1]

    # テスト用画像取得
    image_files = os.listdir(image_dir)

    # モデルのロード
    model = load_model()

    for image_file in image_files:

        # 画像を読み込んで正規化
        image_path = os.path.join(image_dir, image_file)
        normalized_image = load_normalized_image(image_path)

        # 予測を実施
        pred_result = model.predict(normalized_image)[0]

        # 予測確率を出力
        print('=======================================')
        print('file name:', image_file)
        result_msgs = ['%s:%.2f%s' % (conf.classes[i], pred_result[i] * 100, '%') for i in range(len(pred_result))]
        print(result_msgs)

        # 画像を表示
        plt.title(', '.join(result_msgs))
        plt.imshow(np.asarray(image.load_img(image_path)))
        plt.show()
