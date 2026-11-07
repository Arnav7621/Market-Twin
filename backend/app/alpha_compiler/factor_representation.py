
from dataclasses import dataclass
from typing import Union


@dataclass
class FactorNode:

    operator: str

    left: Union["FactorNode", str, None] = None

    right: Union["FactorNode", str, None] = None


def create_example_factor():

    # Factor = (Momentum + Imbalance) / Volatility

    addition = FactorNode(
        operator="add",
        left="Momentum",
        right="Imbalance"
    )

    division = FactorNode(
        operator="div",
        left=addition,
        right="Volatility"
    )

    return division


def print_tree(node, level=0):

    if isinstance(node, str):

        print("  " * level + node)

        return

    print("  " * level + node.operator)

    if node.left is not None:

        print_tree(
            node.left,
            level + 1
        )

    if node.right is not None:

        print_tree(
            node.right,
            level + 1
        )


if __name__ == "__main__":

    factor = create_example_factor()

    print("Example Factor Expression Tree:\n")

    print_tree(factor)
