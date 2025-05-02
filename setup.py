#!/usr/bin/env python

from setuptools import setup, Command, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='navigation-rl-agent',
      version='1.0.0',
      description='Deep Q-Network (DQN) for Unity Banana Collector Environment',
      license='MIT License',
      author='Kashad J. Turner‑Warren',
      author_email='krillavilla@example.com',
      url='https://github.com/krillavilla/navigation-rl-agent',
      packages=find_packages(),
      install_requires=required,
      python_requires='>=3.6',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      long_description=("A Deep Q-Network (DQN) implementation for solving the Unity Banana Collector "
                        "environment. This project demonstrates reinforcement learning techniques "
                        "including experience replay, target networks, and epsilon-greedy exploration "
                        "to train an agent to navigate and collect yellow bananas while avoiding blue ones.")
     )
