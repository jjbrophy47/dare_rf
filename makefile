clean:
	cd dare; rm -rf *.so *.c *.html build/ __pycache__; cd -

build:
	cd dare; python3 setup.py build_ext --inplace; cd ..

get_deps:
	pip3 install -r requirements.txt

all: clean get_deps build