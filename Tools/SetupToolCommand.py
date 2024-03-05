import os
import subprocess
from setuptools import Command

class CompileFortranCommand(Command):
    description = "Compile Fortran sources using f2py"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.run(['f2py', 'sla.f', 'pysla.f90', '-m', 'pysla', '-h', 'pysla.pyf', '--overwrite-signature'], cwd='MapMaker/Tools')
        
        os.environ['CFLAGS'] = '-std=c99'
        subprocess.call(['f2py', '-c', 'pysla.pyf', 'sla.f', 'pysla.f90'], cwd='MapMaker/Tools')
