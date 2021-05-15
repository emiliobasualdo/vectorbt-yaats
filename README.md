# Yet another algorithmic trading study

Arrancar leyendo el código por `strategies/SellOff/SellOff.py`
## Setup
Python >= 3.7.10
```
source venv/bin/activate
```
### Install requirements
```
pip install -r requirements.txt
```
### Create requirements.txt
```
pip freeze > requirements. txt
```
### Run
El header del archivo .csv debe ser el siguiente formato: 
`unix,date,symbol,open,high,low,close,Volume BNB,Volume USDT,tradecount`
```
cd strategies/SellOff
python SellOff.py -m 500 archivo.csv
```
