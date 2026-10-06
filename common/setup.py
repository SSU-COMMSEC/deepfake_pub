from setuptools import setup, find_packages
setup(name="dfbench", version="1.0",
      packages=find_packages(),
      description="Shared victim model, dataset, metrics and resumable record store "
                  "for the UCF-101 / C3D video adversarial-attack benchmark.")
