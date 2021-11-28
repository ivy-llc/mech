# lint as: python3
# Copyright 2021 The Ivy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License..
# ==============================================================================
from distutils.core import setup
import setuptools

setup(name='ivy-mech',
      version='1.1.6',
      author='Ivy Team',
      author_email='ivydl.team@gmail.com',
      description='Mechanics functions with end-to-end support for deep learning developers, written in Ivy',
      long_description="""# What is Ivy Mechanics?\n\nIvy mechanics provides functions for conversions of orientation,
      pose, and positional representations, as well as frame-of-reference transformations, and other more applied functions.
      Ivy currently supports Jax, TensorFlow, PyTorch, MXNet and Numpy. Check out the [docs](https://ivy-dl.org/mech) for more info!""",
      long_description_content_type='text/markdown',
      url='https://ivy-dl.org/mech',
      project_urls={
            'Docs': 'https://ivy-dl.org/mech/',
            'Source': 'https://github.com/ivy-dl/mech',
      },
      packages=setuptools.find_packages(),
      install_requires=['ivy-core'],
      classifiers=['License :: OSI Approved :: Apache Software License'],
      license='Apache 2.0'
      )
