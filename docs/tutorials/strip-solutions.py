import argparse
import os
from traitlets.config import Config
from nbconvert.exporters import NotebookExporter

c = Config()
c.TagRemovePreprocessor.remove_cell_tags = ("solution", )
c.TagRemovePreprocessor.enabled = True
c.NotebookExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]

parser = argparse.ArgumentParser(
    description='Strip cells tagged with "solution" from notebooks.')
parser.add_argument('names',
                    metavar='input_path',
                    type=str,
                    nargs='+',
                    help='Paths of notebooks to convert.')
parser.add_argument('--output-dir', dest='path', help='Output folder')

args = parser.parse_args()

for name in args.names:
    print(f"Stripping solution cells from {name}")
    output = NotebookExporter(config=c).from_filename(name)
    with open(os.path.join(args.path, os.path.basename(name)), "w") as f:
        f.write(output[0])
