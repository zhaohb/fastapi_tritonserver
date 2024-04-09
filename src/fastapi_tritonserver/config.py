import dataclasses
from dataclasses import dataclass
import argparse
import os

now_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(now_dir)


@dataclass
class ServerConf:
    '''arguments'''
    server_url: str = ''
    model_name: str = ''
    tokenizer_path: str = ''
    model_type: str = ''

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            '--server-url',
            type=str,
            default='127.0.0.1:8001',
            help='tritonserver url'
        )
        parser.add_argument(
            '--model-name',
            type=str,
            default="tensorrt_llm",
            help='model name'
        )
        tokenizer_dir = os.path.join(parent_dir, "qwen1.5_7b_chat")
        parser.add_argument(
            '--tokenizer-path',
            type=str,
            default="Qwen/Qwen1.5-4B-Chat",
            help='tokenzier ptah for load')
        parser.add_argument(
            '--model-type',
            type=str,
            default='qwen2-chat',
            help='the model type for load')
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'ServerConf':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        conf = cls(**{attr: getattr(args, attr) for attr in attrs})
        return conf
