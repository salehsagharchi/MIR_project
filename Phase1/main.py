import arabic_reshaper
import click
from bidi.algorithm import get_display

import Parser

reshape_persian_words = True


def prompt_from_list(options: list, prompt_msg="Please Select One Option"):
    n = len(options)
    click.secho(":: ", nl=False, fg="green")
    click.echo(prompt_msg)
    for i in range(1, n + 1):
        click.secho(" " + str(i) + " ", nl=False, fg="blue")
        click.echo(options[i - 1])
    choice = click.prompt(
        click.style('> ', fg='green'),
        type=click.IntRange(1, n),
        prompt_suffix="",
    )
    return choice - 1


def reshape_text(raw_str, lang, reshape=reshape_persian_words):
    if lang != "fa":
        return raw_str
    if not reshape:
        return raw_str
    reshaped_text = arabic_reshaper.reshape(raw_str)
    return get_display(reshaped_text)


def main():
    welcome_text = "با سلام به این برنامه خوش آمدید."
    print(reshape_text(welcome_text, "fa"))

    main_jobs = ["Parsing Raw Files and Generating Documents", "Remove Stopwords", "Make Positional Index"]
    #job = prompt_from_list(main_jobs, "Please select a job you want to execute : ")

    #Parser.remove_stopwords("fa")
    Parser.parse_tedtalks()


if __name__ == "__main__":
    main()
