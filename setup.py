from setuptools import setup, find_packages

setup(
    name='dynamic_scene_graph',
    version='1.0.0',
    description='Dynamic scene graph: object-centric scene representation and tracking for mobile manipulation.',
    packages=find_packages(include=[
        'perception', 'perception.*',
        'utils', 'utils.*',
        'heuristic_tracker', 'heuristic_tracker.*',
        'ekf_tracker', 'ekf_tracker.*',
        'baselines', 'baselines.*',
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
