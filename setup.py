from setuptools import setup, find_packages

setup(
    name='gar-feature-generator',
    version='1.0.0',
    description='GAR-based Financial Anti-Fraud Feature Engineering Toolkit',
    author='Matt',
    author_email='matt@example.com',
    url='https://github.com/mattzx-web/Gar-Feature-Generator',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'gar-gen=src.gar.gar_cpu:main',
            'gar-gen-dist=src.gar.gar_dist:main',
            'gar-gen-ascend=src.gar.gar_ascend:main',
            'kg-gen=src.kg.kg_brute_force:main',
            'kg-gen-cpu=src.kg.kg_cpu:main',
            'kg-gen-gpu=src.kg.kg_gpu:main',
        ],
    },
)
