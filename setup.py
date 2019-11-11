import setuptools

setuptools.setup( 
    name='movie_classifier', 
    version='0.0.1', 
    author='hdtrinh', 
    author_email='trinhhoangduy@gmail.com', 
    description='Movie Genre Classification from Description', 
    packages=setuptools.find_packages(), 
    entry_points={ 
        'console_scripts': [ 
            'movie_classifier = movie_classifier.movie_classifier:main' 
        ] 
    },
    setup_requires = ['nltk'],
    install_requires = ['nltk']
)

