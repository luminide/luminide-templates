import os
import sys
import shutil
import subprocess as sp

print("Running post gen hook...")

# Rename source code directory to 'code/'
os.chdir(os.pardir)
os.rename('{{ cookiecutter.directory_name }}', 'code')
sys.exit(0)
