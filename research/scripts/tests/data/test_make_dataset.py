import pytest
import shutil
from data.make_dataset import preprocess

@pytest.fixture
def raw_and_clean_data_file(tmpdir):
	raw_data_file_path = tmpdir.join("raw.txt")
	clean_data_file_path = tmpdir.join("clean.txt")
	with open(raw_data_file_path, "w") as f:
		f.write("1,801\t201,411\n"
		"1,767565,112\n"
		"2,002\t333,209\n"
		"1990\t782,911\n"
		"1,285\t389129\n")
	yield raw_data_file_path, clean_data_file_path




class TestPreprocess(object):
	def test_on_raw_to_clean_data(self,raw_and_clean_data_file):
		raw_path, clean_path = raw_and_clean_data_file
		preprocess(raw_path, clean_path)
		with open(clean_path) as f:
			lines = f.readlines()
			first_line = lines[0]
			assert first_line == "1,801\t201,411\n"
			second_line = lines[1]
			assert second_line == "1,767565,112\n"
