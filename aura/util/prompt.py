from os import path as os_path

from .string import read, dict_to_string


PROMPTS_PATH = 'assets/prompt'


def make_system_prompt():
    return read(
        os_path.join(
            PROMPTS_PATH,
            'system.md'
        )
    )


def make_annotation_prompt(table: dict, squeeze_rows: bool = False, squeeze_cols: bool = False):
    if squeeze_rows:
        n_chars_per_row = []

        for row in table['rows']:
            n_chars_per_cell = [len(cell.get('text', '')) for cell in row]
            n_chars_per_row.append(sum(n_chars_per_cell) / len(n_chars_per_cell))

        mean_n_chars_per_row = sum(n_chars_per_row) / len(n_chars_per_row)

        filtered_rows = []

        for i, row in enumerate(table['rows']):
            if n_chars_per_row[i] >= mean_n_chars_per_row:
                filtered_rows.append(row)

        table['rows'] = filtered_rows

    if squeeze_cols:
        for i, row in enumerate(table['rows']):
            n_chars_per_cell = [len(cell.get('text', '')) for cell in row]
            mean_n_chars_per_cell = sum(n_chars_per_cell) / len(n_chars_per_cell)

            squeezed_row = [
                cell
                for j, cell in enumerate(row)
                if n_chars_per_cell[j] >= mean_n_chars_per_cell
            ]

            table['rows'][i] = squeezed_row

    # for row in table['rows']:
    #     print([cell.get('text') for cell in row])

    prompt_template = read(
        os_path.join(
            PROMPTS_PATH,
            'annotation.md'
        )
    )

    return prompt_template.format(table = dict_to_string(table))
