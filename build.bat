python -m build
python -m twine upload --repository testpypi dist/*
rem python -m twine upload --repository pypi dist/*