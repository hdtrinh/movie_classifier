import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup( 
    name='movie_classifier-hdtrinh', 
    version='0.0.9', 
    author='hdtrinh', 
    author_email='trinh.hoangduy@gmail.com', 
    description='Movie Genre Classification from Description', 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hdtrinh/movie_classifier",
    packages=setuptools.find_packages(), 
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.1,<3.8.0',
    entry_points={ 
        'console_scripts': [ 
            'movie_classifier = movie_classifier.movie_classifier:main' 
        ] 
    },
    install_requires = [
                        'keras<=2.1.5',
                        'tensorflow<=1.14.0',
                        'sklearn'
                           ]
)


