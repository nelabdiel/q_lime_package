from setuptools import setup, find_packages

setup(
    name='qlime',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'pennylane', 'matplotlib', 
        'scikit-learn', 'IPython'
    ],
    description="A Quantum LIME Explanation Library",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-repo/qlime",
)
