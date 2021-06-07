import os
import sys
import shutil
import subprocess as sp

print("Running post gen hook...")

# move generated source files to 'code/'
os.chdir(os.pardir)
os.mkdir('tmp')
os.rename('{{ cookiecutter.project_name }}', 'code')
shutil.move('code', 'tmp')
os.rename('tmp', '{{ cookiecutter.project_name }}')
# create output dir
os.chdir('{{ cookiecutter.project_name }}')
os.mkdir('output')
sys.exit(0)
