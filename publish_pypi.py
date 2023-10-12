import os

print("Please use the 'gh auth login' command to connect the github API.")
print("Type 'enter' to execute: 'gh auth login'")
input()

os.system("gh run download")

print("Type 'enter' to publish the wheels on pypi:")
input()
os.system("python3 -m twine upload --skip-existing *.whl")
