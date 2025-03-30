import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
	name='PyNN',
	version='1.0',
	author='Sari Sabban',
	author_email='',
	description='A Lightweight NumPy-based neural network library',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/sarisabban/PyNN',
	project_urls={'Bug Tracker':'https://github.com/sarisabban/PyNN/issues'},
	license='MIT',
	packages=['pynn'],
	install_requires=['numpy', 'cupy'])
