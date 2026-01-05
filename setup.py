from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Read the requirements from a file and return them as a list."""

    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements =  [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='azure-ml-e2e',
    version='0.1.0',
    author='Shweta Bambal',
    author_email='shwetabambal18@gmai.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A package for Azure ML end-to-end workflows'

)