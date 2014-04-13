from __future__ import print_function
from cStringIO import StringIO
import sys

from rdkit.Chem.rdmolfiles import (
    MolFromMol2Block,
    MolToPDBBlock,
)


def main(args, stdin=sys.stdin, stdout=sys.stdout):
    if len(args) > 0:
        path = args[0]
        with open(path) as f:
            structure = MolFromMol2Block(f.read())
    else:
        structure = MolFromMol2Block(stdin.read())
    pdb = MolToPDBBlock(structure)
    stdout.write(pdb)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:], sys.stdin, sys.stdout))

