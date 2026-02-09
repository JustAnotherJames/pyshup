import shutil
import subprocess
import tempfile


def shellcheck(code: str) -> None:
    if shutil.which("shellcheck"):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".sh") as f:
            f.write(code)
            f.flush()
            result = subprocess.run(["shellcheck", "--exclude=SC2034,SC2154", f.name])
            assert result.returncode == 0
