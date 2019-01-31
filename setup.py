from pathlib import Path
from setuptools import setup, find_packages

readme = Path(__file__).resolve().parent / 'README.rst'

setup(
    name='text2array',
    version='0.0.1',
    description='Convert your NLP text data to arrays!',
    long_description=readme.read_text(),
    url='https://github.com/kmkurn/text2array',
    author='Kemal Kurniawan',
    author_email='kemal@kkurniawan.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=[
        'numpy ~=1.16.0',
    ],
    python_requires='>=3.6, <4',
)
