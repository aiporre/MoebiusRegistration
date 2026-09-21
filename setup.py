"""Build the Python package and include the compiled SphereMap executable."""

from pathlib import Path
import shutil
import subprocess
from setuptools import setup, Distribution
from setuptools.command.build_py import build_py as _build_py


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # The installed package contains the platform-specific SphereMap binary.
        return True


class build_py(_build_py):
    def run(self):
        root = Path(__file__).parent.resolve()
        binary = root / "Bin/Linux/SphereMap"
        if not binary.exists():
            subprocess.run(["make", "-j2", "Bin/Linux/SphereMap"], cwd=root, check=True)
        super().run()
        target = Path(self.build_lib) / "spheremap" / "_bin" / "SphereMap"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, target)
        target.chmod(target.stat().st_mode | 0o111)


setup(distclass=BinaryDistribution, cmdclass={"build_py": build_py})
