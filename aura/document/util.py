from .Cell import Cell


def get_aligned_cell(cells: list[Cell], col_offset: int):
    n_cols = 0

    cell = None

    for cell in cells:
        if col_offset <= n_cols:
            return cell

        n_cols += cell.n_cols

    return cell
