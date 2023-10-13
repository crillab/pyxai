import os

# How to build, test and publish on pypi a new version:
# - Put the good version in the pyxai/version.txt file 
# - Push on the public github of pyxai 
# - Two github actions are launch: Build and Tests
# - The github action Build create the wheels for pypi
# - The github action Build create the wheels for pypi
# - This script get the last wheels and publish them on pypi. 

print("Please use the 'gh auth login' command to connect the github API.")
print("Type 'enter' to execute: 'gh run download'")
input()

os.system("gh run download")

print("Type 'enter' to publish the wheels on pypi:")
input()
os.system("python3 -m twine upload --skip-existing *.whl")

print("Type 'enter' to delete the whell:")
os.system("rm -rf *.whl")
