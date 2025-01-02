import torchdata.datapipes as dp
import os
import tarfile
from typing import cast, IO, Iterable, Iterator, Optional, Tuple, Dict
from torchdata.datapipes import functional_datapipe
from io import BufferedIOBase
from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
import warnings
from torchdata.datapipes.iter import IterDataPipe
import json


@functional_datapipe("parse_jsonl_files")
class JsonlParserIterDataPipe(IterDataPipe[Tuple[str, Dict]]):

    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO]], **kwargs) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IO]] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Tuple[str, Dict]]:
        for file_name, stream in self.source_datapipe:
            for idx, line in enumerate(stream):
                if line.strip() != '':
                    try:
                        yield f'{file_name}_line{idx}', json.loads(line)
                    except Exception as e:
                        warnings.warn(f"Error occured when parsing string to json due to: {e} abort!")
