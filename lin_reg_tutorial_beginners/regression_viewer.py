#!/usr/bin/env python
""" Regression analysis visualizaer.
Quickly visualize the regression analysis for a given region. To run this app,
execute
    $ bokeh serve regression_viewer.py --args <input.nc>
at the command prompt. Then navigate to the url
    http://[machine_ip]:5006/regression_viewer
On legion, the following additional command line arguments may be useful
    1. allow-websocket-origin {HOST}:{PORT<default=5006>}
    2. host {HOST}:{?}
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import xarray as xr

from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox, layout
from bokeh.models import ColumnDataSource, Spacer
from bokeh.models import HoverTool, ResetTool, PanTool, BoxZoomTool, SaveTool
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter
from bokeh.models.widgets import Div, Select
from bokeh.plotting import figure
from bokeh.charts import Scatter

import sys
try:
    MASTER_DATA_FN = sys.argv[1]
except IndexError:
    print("Need to pass an input dataset filename '--args <input.nc>'")
    sys.exit(1)

print("Loading dataset -", MASTER_DATA_FN)
MASTER_DATA = xr.open_dataset(MASTER_DATA_FN)

ACTS = sorted(list(np.unique(MASTER_DATA.act)))
FIELDS = sorted(list(MASTER_DATA.data_vars))
REGIONS = sorted(list(np.unique(MASTER_DATA.region)))
SEASONS = sorted(list(np.unique(MASTER_DATA.season)))
n = max(map(len, [ACTS, REGIONS, SEASONS]))
color_list = sns.color_palette("Dark2", n).as_hex()

HUES = ['act', 'season', 'region']
HUE_MAP = {'act': ACTS, 'region': REGIONS, 'season': SEASONS}

DATA = None


def nix(val, lst):
    """ Helper function to choose all *but* a given value from a list. """
    return [x for x in lst if x != val]


def _colors(lst):
    uniques = list(np.unique(lst))
    return [color_list[uniques.index(x)] for x in lst]

def _pct_bnds(data, q=0.02):
    return np.percentile(data, [q, 100-q])


def _quick_ols(data):
    data = data.copy().dropna()
    res = smf.ols("data_y ~ data_x", data=data).fit()
    CI = res.conf_int(alpha=0.05)
    CI_5, CI_95 = CI.ix['data_x'][:2]
    return pd.Series({'m': res.params['data_x'],
                      'b': res.params['Intercept'],
                      'CI_5': CI_5, 'CI_95': CI_95,
                      'p': res.pvalues['data_x'],
                      'r2': res.rsquared,
                      'bse': res.bse['data_x'],
                      'n': len(data)})


def get_field_data(field_x, field_y, **selection_kws):
    global DATA

    subset = (
        MASTER_DATA
        [[field_x, field_y]]
        .rename({field_x: 'data_x', field_y: 'data_y'})
        .assign(data_x=lambda ds: np.log10(ds.data_x),
                data_y=lambda ds: np.log10(ds.data_y))
    )

    DATA = subset

def select_data(**selection_kws):
    subset = DATA.copy()
    if "season" in selection_kws:
        season = selection_kws.pop('season')
        subset = subset.where(subset['season'] == season)
    subset = subset.sel(**selection_kws)
    subset = (
        subset
        .to_dataframe()
        .reset_index()
    )
    subset[~np.isfinite(subset[['data_x', 'data_y']])] = np.nan
    subset.dropna(inplace=True)
    subset['color'] = _colors(subset[hue.value])

    return subset

def get_fit_data(data):
    labels = HUE_MAP[hue.value]

    lo, hi = _pct_bnds(data['data_x'])
    xs = np.tile(np.linspace(lo, hi, 10), (len(labels), 1))

    fits = data.groupby(hue.value).apply(_quick_ols)
    ys = [
        fits.loc[val]['b'] + fits.loc[val]['m']*xs[i]
        for i, val in enumerate(labels)
    ]
    ys = np.array(ys)
    colors = color_list[:len(fits)]

    fits['label'] = labels
    source_stats.data = source_stats.from_df(fits)

    return xs.tolist(), ys.tolist(), colors, labels

def get_ctrs(data):
    means = data.groupby(hue.value).mean()
    labels = HUE_MAP[hue.value]
    ctrs = [
        (means.loc[val]['data_x'], means.loc[val]['data_y'])
        for val in labels
    ]
    colors = color_list[:len(means)]
    ctr_x, ctr_y = zip(*ctrs)
    return ctr_x, ctr_y, colors, labels

# Set up widgets
field_x = Select(title="Field (x)", value=FIELDS[0], options=nix(FIELDS[0], FIELDS))
field_y = Select(title="Field (y)", value=FIELDS[1], options=nix(FIELDS[1], FIELDS))

# Coloring widgets
hue = Select(title="Hue", value='season', options=HUES)
sel1 = Select(title="act", value=ACTS[0], options=ACTS)
sel2 = Select(title="region", value=REGIONS[0], options=REGIONS)

# Set up plots and data
source = ColumnDataSource(data=dict(date=[], data_x=[], data_y=[], color=[],
                                    season=[], region=[], act=[]))
source_fits = ColumnDataSource(data=dict(fit_x=[], fit_y=[], color=[], label=[]))
source_ctrs = ColumnDataSource(data=dict(ctr_x=[], ctr_y=[], color=[], label=[]))
source_stats = ColumnDataSource(data=dict())

# http://stackoverflow.com/questions/29435200/bokeh-plotting-enable-tooltips-for-only-some-glyphs
# hvr = HoverTool(tooltips=[(['Label', "@label"]), ])
tooltips = """
<div>
    <span style="font-size: 16px; color: @color">@label</span>
</div>
"""
hvr = HoverTool(tooltips=tooltips)

tools = [hvr, BoxZoomTool(), PanTool(), ResetTool(), SaveTool()]

corr = figure(webgl=True, tools=tools)
corr.grid.grid_line_alpha = 0.

line_kws = dict(line_width=6)
cs = corr.scatter(x="data_x", y="data_y", color='color',
                  size=2, alpha=0.5, source=source)
fits = corr.multi_line(xs='fit_x', ys='fit_y', color='color',
                       source=source_fits, **line_kws)
ctrs = corr.diamond(x='ctr_x', y='ctr_y', color='color',
                    line_color='black', line_width=2, size=20, source=source_ctrs)
corr.tools[0].renderers.append(ctrs)

narr, wide = 100, 200
nd = lambda n: NumberFormatter(format="0." + "0"*n)
columns = [
    TableColumn(field="label", title="Hue Value", width=200),
    TableColumn(field="m", title="Slope", width=150,
                formatter=nd(2)),
    TableColumn(field="CI_5", title="CI = [5,", width=150,
                formatter=nd(2)),
    TableColumn(field="CI_95", title="95]", width=150,
                formatter=nd(2)),
    TableColumn(field="bse", title="Std Error on Slope", width=250,
                formatter=nd(4)),
    TableColumn(field="p", title="P-value", width=150,
                formatter=nd(4)),
    TableColumn(field="r2", title="R-squared", width=150,
                formatter=nd(3)),
]
stats_table = DataTable(columns=columns, source=source_stats,
                        row_headers=False, fit_columns=True)

header_html = """
<h1>Regression Analysis Viewer</h1>
<p>
This application lets you dig into simple regression analyses
compiled from regionally-averaged timeseries from an ensemble
of simulations. You can facet via hue over any of the three
dimensions to compare them simultaneously, and still choose
freely between the two remaining ones.
</p>
<p>
Analyzing data from: <strong>{}</strong>
</p>
<br />
""".format(MASTER_DATA_FN)
header = Div(text=header_html, width=800)

def _make_legend():
    try:
        corr.legend[0].legends.clear()
    except:
        pass
    for col, hv in zip(color_list, HUE_MAP[hue.value]):
        print(hv, col)
        l = corr.line(x=[], y=[], color=col, legend=hv,
                      source=ColumnDataSource(), **line_kws)
corr.legend.location = 'top_left'
# _make_legend()

# Set up callbacks
def field_x_change(attrname, old, new):
    field_y.options = nix(new, FIELDS)
    get_field_data(new, field_y.value)
    update()
def field_y_change(attrname, old, new):
    field_x.options = nix(new, FIELDS)
    get_field_data(field_x.value, new)
    update()
def update_choices(attrname, old, new):
    sel1.title = nix(new, HUES)[0]
    x = HUE_MAP[sel1.title]
    sel1.value = x[0]
    sel1.options = x

    sel2.title = nix(new, HUES)[1]
    y = HUE_MAP[sel2.title]
    sel2.value = y[0]
    sel2.options = y

    # _make_legend()

    update()

def update_on_change(attrname, old, new):
    update()

def update():
    fx, fy = field_x.value, field_y.value

    if DATA is None:
        get_field_data(fx, fy)
    assert DATA is not None

    sel_kws = {sel1.title: sel1.value, sel2.title: sel2.value}
    data = select_data(**sel_kws)
    source.data = source.from_df(data[['data_x', 'data_y', 'color', 'season', 'region', 'act']])

    xs, ys, colors, labels = get_fit_data(data)
    source_fits.data = {'fit_x': xs, 'fit_y': ys,
                        'color': colors, 'label': labels}

    ctr_x, ctr_y, colors, labels = get_ctrs(data)
    # print(ctr_x)
    # print(ctr_y)
    # print(colors)
    source_ctrs.data = {'ctr_x': list(ctr_x), 'ctr_y': list(ctr_y),
                        'color': colors, 'label': labels}

    corr.title.text = ", ".join(["{}: {}".format(key, val)
                                 for(key, val) in sel_kws.items()]) + \
                      " | hue over {}".format(hue.value)
    corr.xaxis.axis_label = "log10(%s)" % fx
    corr.yaxis.axis_label = "log10(%s)" % fy


field_x.on_change('value', field_x_change)
field_y.on_change('value', field_y_change)
sel1.on_change('value', update_on_change)
sel2.on_change('value', update_on_change)
hue.on_change('value', update_choices)

# Set up layout

widgets = widgetbox(field_x, field_y, hue, sel1, sel2,
                    sizing_mode="fixed", width=150)
# layout = column(
#     row(
#         widgets, Spacer(width=25), corr
#     ),
#     row(
#         stats_table
#     )
# )
layout = layout([
    [header],
    [widgets, Spacer(width=25), corr],
    [stats_table]
])

# Initialize
# update()

curdoc().add_root(layout)
curdoc().title = "Regression Viewer"