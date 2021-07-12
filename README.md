[![CI](https://github.com/python-mate/flower-stuff-workshop/actions/workflows/ci.yml/badge.svg)](https://github.com/python-mate/flower-stuff-workshop/actions/workflows/ci.yml)

flower-stuff-workshop
===

🌺 FlowerStuff project; this repository is for experiencing prediction with machine learning. People can experience the result of flower-stuff-lab easily.

## LT document

このプロジェクトは Lightening Talk を見据えて進められました。 LT 資料はこちら(...)。(準備中)

## Manual

LT の内容を書き下したものになります。

1. このリポジトリをローカルへダウンロードします。(zip でも OK、 clone でも勿論 OK)
2. ダウンロードしたリポジトリ内でシェルを開きます。(VSCode でも Terminal でも OK)
3. (必要であれば)おのおの仮想環境を用意。
4. 下記コマンドを順番に打つ。

```bash
# 「リポジトリ内でシェルを開」いているかどうか確認します。
ls

# これ↑でこう↓出れば OK です。
#       Pipfile README.md great_predictor.py requirements.txt 

# pip install で、必要なモジュールをインストールします。
pip install -r requirements.txt

# サンプル画像(あじさい)をダウンロードして予測を行います。
# LT で示したよう、これは2分近くかかる可能性があります。
python great_predictor.py

# 自分で用意した画像で予測を行います。
# "Anthurium-1.png" の部分は、自分が予測したい画像の名前、パスを書きます。
python great_predictor.py Anthurium-1.png
```

注: 予測を行えるのは、 `great_predictor.py` ファイルに書いてある花の種類に限られます。
