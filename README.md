[![CI](https://github.com/python-mate/flower-stuff-workshop/actions/workflows/ci.yml/badge.svg)](https://github.com/python-mate/flower-stuff-workshop/actions/workflows/ci.yml)

flower-stuff-workshop
===

๐บ FlowerStuff project; this repository is for experiencing prediction with machine learning. People can experience the result of flower-stuff-lab easily.

![flower-stuff-project main image](https://user-images.githubusercontent.com/28250432/125736317-740cd173-d30c-4e55-ab4a-765182601558.jpg)

## LT document

ใใฎใใญใธใงใฏใใฏ Lightening Talk ใ่ฆๆฎใใฆ้ฒใใใใพใใใ LT ่ณๆใฏใใกใ([docs/PythonPartyLT-FlowerStuff.md](docs/PythonPartyLT-FlowerStuff.md))ใ

## Manual

LT ใฎๅๅฎนใๆธใไธใใใใฎใซใชใใพใใ

1. ใใฎใชใใธใใชใใญใผใซใซใธใใฆใณใญใผใใใพใใ(zip ใงใ OKใ clone ใงใๅฟ่ซ OK)
2. ใใฆใณใญใผใใใใชใใธใใชๅใงใทใงใซใ้ใใพใใ(VSCode ใงใ Terminal ใงใ OK)
3. (ๅฟ่ฆใงใใใฐ)ใใฎใใฎไปฎๆณ็ฐๅขใ็จๆใ
4. ไธ่จใณใใณใใ้ ็ชใซๆใคใ

```bash
# ใใชใใธใใชๅใงใทใงใซใ้ใใใฆใใใใฉใใ็ขบ่ชใใพใใ
ls

# ใใโใงใใโๅบใใฐ OK ใงใใ
#       Pipfile README.md great_predictor.py requirements.txt 

# pip install ใงใๅฟ่ฆใชใขใธใฅใผใซใใคใณในใใผใซใใพใใ
pip install -r requirements.txt

# ใตใณใใซ็ปๅ(ใใใใ)ใใใฆใณใญใผใใใฆไบๆธฌใ่กใใพใใ
# LT ใง็คบใใใใใใใใฏ2ๅ่ฟใใใใๅฏ่ฝๆงใใใใพใใ
python great_predictor.py

# ่ชๅใง็จๆใใ็ปๅใงไบๆธฌใ่กใใพใใ
# "Anthurium-1.png" ใฎ้จๅใฏใ่ชๅใไบๆธฌใใใ็ปๅใฎๅๅใใในใๆธใใพใใ
python great_predictor.py Anthurium-1.png
```

ๆณจ: ไบๆธฌใ่กใใใฎใฏใ `great_predictor.py` ใใกใคใซใซๆธใใฆใใ่ฑใฎ็จฎ้กใซ้ใใใพใใ
