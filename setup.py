from setuptools import setup, find_packages

setup(
    name='scenerep',
    version='1.0.0',
    description='Scene representation and tracking for mobile manipulation.',
    packages=find_packages(include=[
        'scene', 'scene.*',
        'detection', 'detection.*',
        'pose_update', 'pose_update.*',
        'utils', 'utils.*',
    ]),
    include_package_data=True,
    license='MIT License',
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'opencv-python',
        'open3d>=0.18',
        'scipy',
    ],
)
