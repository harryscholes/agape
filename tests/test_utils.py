from pytest import raises
from tempfile import TemporaryDirectory
from io import StringIO
from agape.utils import directory_exists, stdout


class TestDirectoryExists:
    def test_directory_exists(self):
        with TemporaryDirectory() as d:
            assert directory_exists(d) == d

    def test_raises_FileNotFoundError(self):
        with TemporaryDirectory() as d:
            with raises(FileNotFoundError):
                directory_exists(d + "NOTAPATH")


class TestStdout:
    def setup_method(self):
        self.s = "string"
        self.o = "object"
        self.f = StringIO()

    def test_string(self):
        stdout(self.s, file=self.f)
        assert self.f.getvalue() == 'string\n\n\n'

    def test_string_object(self):
        stdout(self.s, self.o, file=self.f)
        assert self.f.getvalue() == 'string:\n    object\n\n'
