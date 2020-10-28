.ONESHELL:

SRC = $(wildcard *.ipynb)

all: lib docs

lib: $(SRC)
	nbdev_build_lib
	touch mcbbox

docs_serve: docs
	cd docs && bundle exec jekyll serve --incremental

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs --n_workers=1

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist
	rm -rf docs/_site
