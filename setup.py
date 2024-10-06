from setuptools import setup, find_packages

setup(
    name='pydil',
    version='0.1.0',  # Update as needed
    author='Eduardo Fernandes Montesuma,Fred Ngolè Mboula,Antoine Souloumiac',
    author_email='edumontesuma@gmail.com',
    description='A Python package for dataset dictionary learning',
    long_description=open('README.md').read(),  # Ensure you have a README.md
    long_description_content_type='text/markdown',
    url='https://github.com/eddardd/PyDiL',  # Update with your repo URL
    packages=find_packages(),  # Automatically find your packages
    install_requires=[
        "matplotlib==3.6.2",
        "numpy==1.23.5",
        "POT==0.8.2",
        "scikit_learn==1.2.0",
        "torch==1.13.0",
        "tqdm==4.66.4"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: CeCILL-B License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
