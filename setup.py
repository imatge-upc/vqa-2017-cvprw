import setuptools

import vqa

setuptools.setup(
    name='VQA',
    version=vqa.__version__,
    description='Visual Question-Answering',
    url='https://github.com/imatge-upc/vqa-2017-cvprw.git',
    author='Francisco Roldán Sánchez',
    author_email='fran.roldans@gmail.com',
    license='MIT',
    packages=['vqa'],
    zip_safe=False
)