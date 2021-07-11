"""great_predictor
もちろん Great Pretender オマージュ。
"""

# Built-in modules
import sys

# 他、必要なライブラリを import します。
# NOTE: 今回は試しに keras. から参照するようにしてみます。
import tensorflow.keras
import numpy
from google_drive_downloader import GoogleDriveDownloader


# Prediction に使う hdf5 です。
HDF5 = dict(
    drive_id='10PjbaGKUoNs1a0vPkAZ5z4Inl-ioaW6c',
    destination_path='./inception_v3_n_flowers_fine_tuning_2nd_cell.hdf5',
)
# 画像を用意するのが面倒くさいひとのためのサンプル画像です。
SAMPLE_IMAGE = dict(
    drive_id='1MJyIANeYSRHw4_02TnyM2KVCtdvMeMMC',
    destination_path='./Hydrangea-1.png',
)

# その他定数です。
INCEPTION_V3_TARGET_SIZE = (299, 299)
CLASSES_FOR_N_FLOWERS = sorted([
    'Anthurium',
    'Bluebell',
    'Buttercup',
    'ColtsFoot',
    'Cowslip',
    'Crocus',
    'Daffodil',
    'Daisy',
    'Dandelion',
    'Fritillary',
    'Hydrangea',
    'Iris',
    'LilyValley',
    'Pansy',
    'Snowdrop',
    'Sunflower',
    'Tigerlily',
    'Tulip',
    'Windflower',
])


def main(prediction_target_path: str = None):
    """さっくり prediction を行う関数です。

    Args:
        prediction_target_path (str, optional): Image path you want to predict. Defaults to None.
    """

    # Prediction に使う hdf5 をダウンロードしてきます。
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id=HDF5['drive_id'],
        dest_path=HDF5['destination_path'],
        unzip=False,
    )
    # 画像を用意するのが面倒な困ったサンのために画像をダウンロードしてきます。
    if not prediction_target_path:
        print('Use sample image... since no image was detected.')
        GoogleDriveDownloader.download_file_from_google_drive(
            file_id=SAMPLE_IMAGE['drive_id'],
            dest_path=SAMPLE_IMAGE['destination_path'],
            unzip=False,
        )

    # モデルをロードします。
    model = tensorflow.keras.models.load_model(HDF5['destination_path'])

    # 画像を img_nad へ変換します。
    # NOTE: nad = Normalization and Division(正規化と割り算)
    img_nad = convert_image_for_prediction(
        SAMPLE_IMAGE['destination_path'],
        target_image_size=INCEPTION_V3_TARGET_SIZE,
        color_mode='rgb',
        verbose=False,
    )

    # 予測を行います。
    prediction = predict_top_classes(model, img_nad, CLASSES_FOR_N_FLOWERS, 3)

    # 予測の結果を出力します。
    print('\n'.join([
        '*****',
        f'Image: {prediction_target_path}',
        f'    Predicted: {prediction}',
        '*****',
    ]))


def predict_top_classes(model,
                        img_nad: numpy.ndarray,
                        classes: list,
                        top_n: int) -> list:
    """ひとつの画像を predict します。結果はクラス名と確信度のリスト。
    トップ n 件出します。

    Args:
        model (keras.engine.functional.Functional): モデル。
        img_nad (numpy.ndarray): 正規化を終えた画像。
        classes (list): クラスのリスト。
        top_n (int): トップ n 件結果を返します。

    Returns:
        list: Prediction 結果のリスト。
    """
    prediction = model.predict(img_nad)[0]
    top_indices = prediction.argsort()[-top_n:][::-1]
    return [(classes[i], prediction[i]) for i in top_indices]


def convert_image_for_prediction(image_path: str,
                                 target_image_size: tuple,
                                 color_mode: str,
                                 verbose: bool) -> numpy.ndarray:
    """メチャクチャ苦労して、画像を prediction へ変換する処理を整理しました。

    Args:
        image_path (str): 画像パス。
        target_image_size (tuple): 変換後の画像サイズ。
        color_mode (str): 'rgb', 'rgba', 'grayscale'
        verbose (bool): 途中のロギングを行う。

    Returns:
        numpy.ndarray: Prediction へ投げられる形式へ変換された画像。
    """

    # 引数チェックを実施します。
    assert len(target_image_size) == 2, (
        f'target_image_size には2要素が期待されています。与えられた要素数: {len(target_image_size)}'
    )
    assert color_mode in ['rgb', 'rgba', 'grayscale'], (
        f"color_mode には次の3種類が期待されています。与えられた引数: {['rgb', 'rgba', 'grayscale']}"
    )

    # load_img は画像をファイルから読み込み、PIL 形式で返します。
    # NOTE: だから内部で Pillow を必要としています。
    img = tensorflow.keras.preprocessing.image.load_img(
        image_path,
        # grayscale と color_mode の指定により、読み込んだあとに rgb に変換します。
        color_mode=color_mode,
        # 読み込んだあと(たとえば)32x32にリサイズします。
        target_size=target_image_size,
    )
    if verbose:
        print(f'load_img の返り値: {type(img)}')  # <class 'PIL.Image.Image'>
        print(f'PIL.Image.Image size: {img.size}')

    # PIL 形式を numpy.ndarray へ変換します。
    # NOTE: ここでエラーが出る可能性があります。
    #       TypeError: __array__() takes 1 positional argument but 2 were given
    #       このエラーは Pillow のバージョンを変更することで解消します。
    #       具体的には pip install "pillow!=8.3.0"
    img = tensorflow.keras.preprocessing.image.img_to_array(img)
    if verbose:
        print(f'img_to_array の返り値: {type(img)}')  # <class 'numpy.ndarray'>

    # 0-1 に変換します。
    # NOTE: 色は 0~255 で表されているので 255 で割ると 0~1 に変換できる。
    # NOTE: nad = Normalization and Division(正規化と割り算)
    img_nad = img / 255
    if verbose:
        print(f'正規化後の img: {type(img_nad)}')  # <class 'numpy.ndarray'>
        print(f'正規化後の img.shape: {img_nad.shape}')  # (32, 32, 3)

    # 4次元配列に
    # XXX: なんで4次元配列にするのかわかってない。
    img_nad = img_nad[None, ...]

    return img_nad


if __name__ == '__main__':

    # 今回は引数ナシも許します。
    prediction_target_path = sys.argv[1] if len(sys.argv) >= 2 else None
    main(prediction_target_path)
