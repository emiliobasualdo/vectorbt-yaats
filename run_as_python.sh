set -e
cd $1
jupyter nbconvert *.ipynb --to python
echo "Running"
time python *.py
