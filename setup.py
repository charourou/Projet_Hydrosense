from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='hydrosense',
      version="0.0.1",
      description="Hydrosense Model Local",
      license="MIT",
      author="Yann Romain Maxime",
      author_email="maxime.richard@gmail.com",
      #url="https://github.com/charourou/Projet_Hydrosense",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      # include_package_data=True,
      zip_safe=False
      )
