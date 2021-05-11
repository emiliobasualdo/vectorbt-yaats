# Yet another algorithmic trading study

Arrancar leyendo el código por `strategies/SellOff/SellOff.py`
## Setup
Using Python 3.7.10, should work with >= 3.7.10
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
```
cd strategies/SellOff
python SellOff.py -m 500 /Users/pilo/development/itba/pf/Binance_Minute_OHLC_CSVs/3000/Binance_BNBUSDT_minute_3000.csv
```