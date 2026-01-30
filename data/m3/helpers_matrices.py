import matplotlib
import numpy as np
import pandas as pd
from IPython.display import display_html
from itertools import chain, cycle
from matplotlib.colors import LinearSegmentedColormap

cm = LinearSegmentedColormap.from_list(
    'rg', ["mistyrose", "w", "lightgreen"], N=512)


def display_gradient(df: pd.DataFrame) -> None:
    display(df.style.background_gradient(cmap=cm))


def color_nan(x: float) -> str:
    if np.isnan(x):
        return 'background-color: whitesmoke; text-align:center'
    else:
        return f'background-color: {matplotlib.colors.to_hex(cm(int(min(511, x*511))), keep_alpha=False)}'


def nice(df: pd.DataFrame, nan: str = 'nan'):
    return df.fillna(-9.99).style.format(precision=2).background_gradient(vmin=0,vmax=1).map(lambda v: "background-color: white; color: white" if v==-9.99 else "")


def random_df(index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.random.uniform(size=(len(index), len(columns))), index=index, columns=columns)


def display_side_by_side(*args, titles=cycle([''])) -> None:
    html_str = ''
    for style, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += style.set_table_attributes(
            "style='display:inline; padding:5px'").set_caption(title)._repr_html_()
        html_str += ""
    display_html(html_str, raw=True)
