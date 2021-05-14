import os
from setuptools import setup, find_packages

def readrequirements():
    with open(f"{os.path.abspath('.')}/rlcluster/requirements.txt", 'r') as f:
        content = f.readlines()
    content = [package.replace('\n','') for package in content]
    return content

setup(
    name='rlcluster',
    version='1.0',
    description=('Collection of Reinforcement learning algorithms'),
    author='Suraj',
    author_email='surajpedasingu@gmail.com',
    packages=find_packages(include=['rlcluster']),
    platforms=["all"],
    install_requires=readrequirements(),
    py_modules=['rlcluster.trainagent'],
    entry_points=
    '''
        [console_scripts]
        rlclustertrain=rlcluster.trainagent:rlclustercli
    ''',
)