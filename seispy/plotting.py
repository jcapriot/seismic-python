import warnings
import matplotlib.pyplot as plt
import numpy as np

def wiggle(data, ax=None, color='k'):

    if ax is None:
        plt.figure()
        ax = plt.gca()

    trace_plots = []
    line_plots = []
    y_max = 0
    for i, trace in enumerate(data):
        n1 = trace.n_sample
        dt = trace.d_sample
        y_max = max(y_max, dt * (n1 - 1))
        if dt == 0:
            warnings.warn("trace has a 0 sample spacing, setting to 4ms")
            dt = 0.004
        s_locs = np.linspace(0, n1*dt, n1, endpoint=False)
        dat = np.asarray(trace)

        fill = ax.fill_betweenx(s_locs, dat+i, i, where=dat>0, interpolate=True, color=color)
        line = ax.plot(dat+i, s_locs, color=color)
        trace_plots.append(fill)
        line_plots.append(line)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_ylim([y_max, 0])
    return trace_plots, line_plots

def image(data, ax=None, cmap='seismic', **kwargs):

    if ax is None:
        plt.figure()
        ax = plt.gca()
        ax.invert_yaxis()

    im_dat = np.array([trace for trace in data]).T
    if 'clim' not in kwargs or 'vmin' not in kwargs or 'vmax' not in kwargs:
        max_v = np.abs(im_dat).max()
        if 'vmin' not in kwargs:
            kwargs['vmin'] = -max_v
        if 'vmax' not in kwargs:
            kwargs['vmax'] = max_v

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    return ax.pcolormesh(im_dat, cmap=cmap, **kwargs)