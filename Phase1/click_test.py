import time

import click

lsp = range(100)


def modify_the_user(user):
    time.sleep(0.02)


def get_commit_message():
    MARKER = '# Everything below is ignored\n'
    message = click.edit('\n\n' + MARKER)
    if message is not None:
        return message


choice = click.prompt(
    "Please select:",
    type=click.IntRange(0, 10),
    show_default=True,
)
print(choice)


value = click.prompt('Please enter a valid integer', type=float)

print("سلام")
click.echo("سلام")
click.echo('ادامه? [yn] ', nl=False)
c = click.getchar()
click.echo()
if c == 'y':
    click.echo('اتاتاتات')
elif c == 'n':
    click.echo('Abort!')
else:
    click.echo('Invalid input :(')

with click.progressbar(
        label='Modifying user accounts',
        length=100, fill_char="█") as bar:
    for user in lsp:
        modify_the_user(user)
        bar.update(1)

click.echo_via_pager('\n'.join('Line %d' % idx
                               for idx in range(50)))
time.sleep(0.5)

click.echo("hello2", err=True)
time.sleep(0.5)
