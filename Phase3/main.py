import os
import pickle
import click

from Phase3 import Parser
from Phase3.Parser import TextNormalizer as Normalizer, TextNormalizer
from Phase3 import Constants


def prompt_from_list(options: list, prompt_msg="Please Select One Option"):
    n = len(options)
    click.secho("\n:: ", nl=False, fg="green")
    click.echo(prompt_msg)
    for i in range(1, n + 1):
        click.secho(" " + str(i) + " ", nl=False, fg="blue")
        click.echo(options[i - 1])
    choice = click.prompt(
        click.style('> ', fg='green'),
        type=click.IntRange(1, n),
        prompt_suffix="",
    )
    print("")
    return choice - 1


class Main:

    def start(self):
        welcome_text = "Welcome to this application !"
        print(welcome_text)
        main_jobs = {
            "EXIT": -1
        }
        finish = False
        while not finish:
            job = prompt_from_list(list(main_jobs), "Please select a job you want to execute : ")
            command = list(main_jobs.values())[job]
            finish = command == -1
            if callable(command):
                command()


if __name__ == "__main__":
    test = "خروج خودروها از استان تهران ممنوع شد | کدام خودروها مستثنی شدند؟"
    print(Parser.TextNormalizer.prepare_text(test, "fa"))

    my_main = Main()
    my_main.start()


