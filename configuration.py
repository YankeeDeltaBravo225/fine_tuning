class configuration:
    # 分類するクラス
    classes = ['kyoko', 'noriko']
    nb_classes = len(classes)

    # トレーニング用とバリデーション用の画像格納先
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'

    # 出力先ディレクトリ
    result_dir = 'results'

    # 重みファイル名
    weight_file = 'finetuning.h5'

    # トレーニング用サンプル数
    nb_train_samples = 25

    # バリデーション用サンプル数
    nb_validation_samples = 10

    # バッチサイズ
    batch_size = 16

    # エポック数
    nb_epoch = 40

    # 画像の処理サイズ
    img_width = 150
    img_height = 150


if __name__ == '__main__':
    print(configuration.__dict__)
