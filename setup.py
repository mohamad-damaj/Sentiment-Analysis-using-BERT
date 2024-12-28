from setuptools import find_packages,setup
from typing import List


endremoval = '-e .'
def get_requirements(filepath: str) -> List[str]:
    
    requirements = []
    with open(filepath) as filez:
        requirements = filez.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
        if endremoval in requirements:
            requirements.remove(endremoval)
        
    return requirements

setup(
name = 'Sentiment Analysis with BERT',
version = '0.0.1',
author = 'Mohamad',
author_email = 'mohamadamaj123@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt'),

)