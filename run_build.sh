rm -r dist/ build/ kanonym4text.egg-info/
python3 setup.py sdist bdist_wheel
twine upload dist/*
