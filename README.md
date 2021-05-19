# Yet another algorithmic trading study

Arrancar leyendo el código por `strategies/SellOff/SellOff.py`
## Setup
Python >= 3.7.10
```
python -m venv venv
source venv/bin/activate
```
### Install requirements
Tal vez haya que instalar gcc y otros antes
```
sudo yum install python3-devel gcc
```
Primero instalar [TA-LIB](https://github.com/mrjbq7/ta-lib#dependencies)  
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
