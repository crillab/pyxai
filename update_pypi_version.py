import os

versionFile = os.getcwd() + os.sep + "pyxai" + os.sep + "version.txt"
__version__ = open(versionFile, encoding='utf-8').read()

print("Type on the keyboard the new version number (current " + str(__version__) + ") ?")

new_version = input()
if new_version != "":
    print("Write the new version ...")
    with open(versionFile, "r+") as f:
        data = f.read()
        f.seek(0)
        f.write(new_version)
        f.truncate()

# print("sudo rm -rf build/ dist/ wheelhouse/ rm -rf dist/ build/ pyxai-experimental.egg-info/ pyxai.egg-info/")

os.system("sudo rm -rf build/ dist/ wheelhouse/ rm -rf dist/ build/ pyxai-experimental.egg-info/ pyxai.egg-info/")

# print("Type Enter to execute python3 setup.py sdist bdist_wheel")
# input()
# os.system("python3 setup.py sdist bdist_wheel")

print("Type Enter to execute the docker that build whells:")
input()

os.system("sudo docker run --rm -e PLAT=manylinux2014_x86_64 -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 /io/travis/build-wheels.sh")

print("Type Enter to execute the publish on pypi the wheels:")
input()
os.system("python3 -m twine upload --skip-existing wheelhouse/*.whl")
