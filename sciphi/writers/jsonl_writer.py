"""A module which facilitates JSONL data writing."""
import json
import os
import time

from sciphi.writers.base import DataWriter


class JsonlDataWriter(DataWriter):
    """A class to write data to a JSONL file."""

    def __init__(self, output_path, overwrite=True):
        """Initialize the DataWriter."""
        self.output_path = output_path
        self.overwrite = overwrite

    def _get_modified_path(self):
        """Modify the output path if the file already exists and overwriting is not allowed."""
        if not self.overwrite and os.path.exists(self.output_path):
            base_name, ext = os.path.splitext(self.output_path)
            return f"{base_name}_{int(time.time())}{ext}"
        return self.output_path

    def write(self, data: list[dict]) -> None:
        """
        Write the provided data to the specified path.

        Args:
            data (list): List of data entries to be written.
        """
        path = self._get_modified_path()

        with open(path, "a") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    def append(self, data: list[dict]) -> None:
        path = self._get_modified_path()
        if os.path.exists(path):
            with open(path, 'r') as json_file:
                writing_data = json.load(json_file)
            writing_data.extend(data)
        else:
            writing_data = data
        with open(path, "w") as f:
            json.dump(writing_data, f)


if __name__ == '__main__':
    out_path = '../../outputs/llama_cpp/llama_v2_7b/test.jsonl'
    new_dict = {"formatted prompt": "New Prompt2", "completion": "New Completion"}

    writer = JsonlDataWriter(out_path)
    writer.append(new_dict)
