from distutils.core import setup

# I really prefer Markdown to reStructuredText.  PyPi does not.  This allows me
# to have things how I'd like, but not throw complaints when people are trying
# to install the package and they don't have pypandoc or the README in the
# right place.
try:
   import pypandoc
   longdesc = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   longdesc = ''

setup(name='schrod',
      version='0.1dev',
      description="A tool for accurately and rapidly solving batches of Schrodinger equations",
      long_description=longdesc,
      author="D. Hudson Smith",
      author_email="dane.hudson.smith@gmail.com",
      url="https://github.com/dhudsmith/schrod",
      py_modules=['schrod',],
      license="MIT License"
      )