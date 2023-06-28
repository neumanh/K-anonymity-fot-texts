from setuptools import setup, find_packages

VERSION = '0.1.3'
DESCRIPTION = 'k-anonymity for texts'
LONG_DESCRIPTION = 'A package that applies k-anonymity on extual documents.'

# Setting up
setup(
    name="kanonym4text",
    version=VERSION,
    author="Lior Trieman, Hadas Neuman",
    author_email="liortr30@gmail.com, hadas.doron@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    # package_dir={'kanonym4text': 'kanonym4text'},
    install_requires=[
        'numpy==1.23.0',
        'pandas==1.5.3',
        'matplotlib==3.7.1',
        'gensim==4.3.1',
        'torch==2.0.1',
        'transformers==4.29.0',
        'sentence-transformers==2.2.2',
        'spacy==3.5.3',
        'nltk==3.8.1',
        'annoy==1.17.2',
        'k-means-constrained==0.7.2'
    ],
    keywords=['python', 'k-anonymity', 'privacy', 'NLP'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    package_data={
        'kanonym4text': ['data/5000_most_common_words_by_order.txt']}
)

