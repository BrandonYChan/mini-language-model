from dataclasses import dataclass
from textwrap import dedent

@dataclass
class PrintFormatter:

    @staticmethod
    def print_header(text: str) -> None:
        print(dedent(f""" 
            -----------------------------------------------------
            ================={text}================
            -----------------------------------------------------
              """))