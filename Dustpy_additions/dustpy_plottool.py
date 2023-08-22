from dustpy import plot
import dustpy.constants as c
import matplotlib.pyplot as plt
from dustpy import Simulation
import os
from simframe.io.writers import hdf5writer
from types import SimpleNamespace
import numpy as np
import warnings
from scipy.interpolate import interp1d

def my_panel(data, filename="data", extension="hdf5", width = 3.5, im=0, ir=0, it=0, show_limits=True, show_St1=True):
    """Simple plotting script for data files or simulation objects.

    Parameters
    ----------
    data : ``dustpy.Simulation`` or string
        Either instance of ``dustpy.Simulation`` or path to data directory to be plotted
    filename : string, optional, default : "data"
    extension : string, optional, default : "hdf5"
        Plotting script is looking for files with pattern ``<data>/<filename>*.<extension>``
    im : int, optional, default : 0
        Number of mass bin along which density distribution is plotted
    ir : int, optional, default : 0
        Number of radial grid index along density distribution is plotted
    it : int, optional, default : 0
        Index of snapshot to be plotted
    show_limits : boolean, optional, default : True
        If True growth limits are plotted
    show_St1 : boolean, optional, default : True
        If True St=1 line is plotted"""

    from dustpy.plot import __version__

    data = _readdata(data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data.Nt-1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nm[it, ...]-1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...]-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDust.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max()/c.M_sun)) + 1
    levels = np.linspace(sd_max-6, sd_max, 7)

    
    fig = plt.figure(figsize=(3.*width, 2.*width/1.618), dpi=150)
    ax00 = fig.add_subplot(231)
    ax01 = fig.add_subplot(232)
    ax02 = fig.add_subplot(233)
    ax10 = fig.add_subplot(234)
    ax11 = fig.add_subplot(235)
    ax11r = ax11.twinx()

    # Density distribution
    plt00 = ax00.contourf(data.r[it, ...]/c.au,
                          data.m[it, ...],
                          np.log10(data.sigmaDust[it, ...].T),
                          levels=levels,
                          cmap="magma",
                          extend="both"
                          )
    if show_St1:
        ax00.contour(data.r[it, ...]/c.au,
                     data.m[it, ...],
                     data.St[it, ...].T,
                     levels=[1.],
                     colors="white",
                     linewidths=2
                     )
    if show_limits:
        ax00.contour(data.r[it, ...]/c.au,
                     data.m[it, ...],
                     (data.St - data.StDr[..., None])[it, ...].T,
                     levels=[0.],
                     colors="C2",
                     linewidths=1
                     )
        ax00.contour(data.r[it, ...]/c.au,
                     data.m[it, ...],
                     (data.St - data.StFr[..., None])[it, ...].T,
                     levels=[0.],
                     colors="C0",
                     linewidths=1
                     )

    ax00.axhline(data.m[it, im], color="#AAAAAA", lw=1, ls="--")
    ax00.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")

    cbar00 = plt.colorbar(plt00, ax=ax00)
    cbar00.ax.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
    cbar00ticklabels = []
    for i in levels:
        cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
    cbar00.ax.set_yticklabels(cbar00ticklabels)
    ax00.set_xscale("log")
    ax00.set_yscale("log")
    ax00.set_xlabel("Distance from star [AU]")
    ax00.set_ylabel("Particle mass [g]")

    ax01.loglog(data.m[it, ...], data.sigmaDust[it, ir, :], c="C3")
    ax01.set_xlim(data.m[it, 0], data.m[it, -1])
    ax01.set_ylim(10.**(sd_max-6.), 10.**sd_max)
    ax01.set_xlabel("Particle mass [g]")
    ax01.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    if data.Nt < 3:
        ax02.set_xticks([0., 1.])
        ax02.set_yticks([0., 1.])
        ax02.text(0.5,
                  0.5,
                  "Not enough data points.",
                  verticalalignment="center",
                  horizontalalignment="center",
                  size="large")
    else:
        ax02.loglog(data.t/c.year, data.Mgas/c.M_sun, c="C0", label="Gas")
        ax02.loglog(data.t/c.year, data.Mdust /
                    c.M_sun, c="C1", label="Dust")
        ax02.axvline(data.t[it]/c.year, c="#AAAAAA", lw=1, ls="--")
        ax02.set_xlim(data.t[1]/c.year, data.t[-1]/c.year)
        ax02.set_ylim(10.**(Mmax-6.), 10.**Mmax)
        ax02.legend()
    ax02.set_xlabel("Time [yrs]")
    ax02.set_ylabel("Mass [$M_\odot$]")

    ax10.loglog(data.r[it, ...]/c.au, data.sigmaDust[it, :, im], c="C3")
    ax10.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
    ax10.set_ylim(10.**(sd_max-6.), 10.**sd_max)
    ax10.set_xlabel("Distance from star [au]")
    ax10.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")

    ax11.loglog(data.r[it, ...]/c.au, data.SigmaGas[it, ...], label="Gas")
    ax11.loglog(data.r[it, ...]/c.au,
                data.SigmaDust[it, ...].sum(-1), label="Dust")
    ax11.set_xlim(data.r[it, 0]/c.au, data.r[it, -1]/c.au)
    ax11.set_ylim(10.**(sg_max-6), 10.**sg_max)
    ax11.set_xlabel("Distance from star [AU]")
    ax11.set_ylabel("$\Sigma$ [g/cm²]")
    ax11.legend()
    ax11r.loglog(data.r[it, ...]/c.au, data.eps[it, ...], color="C7", lw=1)
    ax11r.set_ylim(1.e-5, 1.e1)
    ax11r.set_ylabel("Dust-to-gas ratio")

    fig.tight_layout()

    fig.text(0.99, 0.01, "DustPy v"+__version__, horizontalalignment="right",
             verticalalignment="bottom")
    
    plt.suptitle(f'Time {data.t[it]/c.year:9.3e} yr, Mass = {data.m[it, im]:9.3e} g, Distance, {data.r[it, ir]/c.au:9.3e} AU')
    plt.tight_layout()

    plt.show()

plot.my_panel = my_panel


def _readdata(data, filename="data", extension="hdf5"):

    ret = {}

    if isinstance(data, Simulation):

        m = data.grid.m[None, ...]
        Nm = data.grid.Nm[None, ...]
        r = data.grid.r[None, ...]
        ri = data.grid.ri[None, ...]
        Nr = data.grid.Nr[None, ...]
        t = data.t[None, ...]
        Nt = np.array([1])[None, ...]

        SigmaDust = data.dust.Sigma[None, ...]
        SigmaGas = data.gas.Sigma[None, ...]
        eps = data.dust.eps[None, ...]

        cs = data.gas.cs[None, ...]
        delta = data.dust.delta.turb[None, ...]
        OmegaK = data.grid.OmegaK[None, ...]
        St = data.dust.St[None, ...]
        vK = OmegaK[None, ...] * r
        vFrag = data.dust.v.frag[None, ...]

    elif os.path.isdir(data):

        writer = hdf5writer()

        # Setting up writer
        writer.datadir = data
        writer.extension = extension
        writer.filename = filename

        m = writer.read.sequence("grid.m")
        Nm = writer.read.sequence("grid.Nm")
        r = writer.read.sequence("grid.r")
        ri = writer.read.sequence("grid.ri")
        Nr = writer.read.sequence("grid.Nr")
        t = writer.read.sequence("t")
        Nt = np.array([len(t)])

        SigmaDust = writer.read.sequence("dust.Sigma")
        SigmaGas = writer.read.sequence("gas.Sigma")
        RhoSolid = writer.read.sequence('dust.rhos')
        Grainsize = writer.read.sequence("dust.a")
        eps = writer.read.sequence("dust.eps")

        cs = writer.read.sequence("gas.cs")
        delta = writer.read.sequence("dust.delta.turb")
        OmegaK = writer.read.sequence("grid.OmegaK")
        St = writer.read.sequence("dust.St")
        vK = OmegaK * r
        vFrag = writer.read.sequence("dust.v.frag")

    else:

        raise RuntimeError("Unknown data type.")

    # Masses
    Mgas = (np.pi * (ri[..., 1:]**2 - ri[..., :-1]**2) * SigmaGas[...]).sum(-1)
    Mdust = (np.pi * (ri[..., 1:]**2 - ri[..., :-1]**2)
             * SigmaDust[...].sum(-1)).sum(-1)

    # Transformation of the density distribution
    a = np.array(np.mean(m[..., 1:] / m[..., :-1], axis=-1))
    dm = np.array(2. * (a - 1.) / (a + 1.))
    sigmaDust = SigmaDust[...] / dm[..., None, None]

    # Fragmentation limit
    b = vFrag**2 / (delta * cs**2)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            r'invalid value encountered in sqrt')
        StFr = 1 / (2 * b) * (3 - np.sqrt(9 - 4 * b**2))

    # Drift limit
    p = SigmaGas * OmegaK * cs / np.sqrt(2.*np.pi)
    StDr = np.zeros_like(StFr)
    for i in range(int(Nt)):
        _f = interp1d(np.log10(r[i, ...]), np.log10(
            p[i, ...]), fill_value='extrapolate')
        pi = 10.**_f(np.log10(ri[i, ...]))
        gamma = np.abs(r[i, ...] / p[i, ...] *
                       np.diff(pi) / np.diff(ri[i, ...]))
        StDr[i, ...] = eps[i, ...] / gamma * (vK[i, ...] / cs[i, ...])**2

    ret["m"] = m
    ret["Nm"] = Nm
    ret["r"] = r
    ret["ri"] = ri
    ret["Nr"] = Nr
    ret["t"] = t
    ret["Nt"] = Nt

    ret["SigmaDust"] = SigmaDust
    ret["sigmaDust"] = sigmaDust
    ret["SigmaGas"] = SigmaGas
    ret["Grainsize"] = Grainsize
    ret["RhoSolid"] = RhoSolid
    ret["eps"] = eps

    ret["Mdust"] = Mdust
    ret["Mgas"] = Mgas

    ret["cs"] = cs
    ret["delta"] = delta
    ret["OmegaK"] = OmegaK
    ret["St"] = St
    ret["StDr"] = StDr
    ret["StFr"] = StFr
    ret["vK"] = vK
    ret["vFrag"] = vFrag

    return SimpleNamespace(**ret)

plot._readdata = _readdata
    

def my_density(data, filename="data", extension="hdf5", width = 3.5, height = None, im=0, ir=0, it=0, plot_grainsize = True, show_limits=False, show_St1=False, return_data = False, return_axis = False):
    data = _readdata(data, filename=filename, extension=extension)

    # Fix indices if necessary
    it = np.maximum(0, it)
    it = np.minimum(it, data.Nt-1)
    it = int(it)
    im = np.maximum(0, im)
    im = np.minimum(im, data.Nm[it, ...]-1)
    im = int(im)
    ir = np.maximum(0, ir)
    ir = np.minimum(ir, data.Nr[it, ...]-1)
    ir = int(ir)

    # Get limits/levels
    sd_max = np.ceil(np.log10(data.sigmaDust.max()))
    sg_max = np.ceil(np.log10(data.SigmaGas.max()))
    Mmax = np.ceil(np.log10(data.Mgas.max()/c.M_sun)) + 1
    levels = np.linspace(sd_max-6, sd_max, 7)

    if height == None:
        fig, ax00 = plt.subplots(figsize=(3.*width, 2.*width/1.618))
    else:
        fig, ax00 = plt.subplots(figsize=(width, height))
    plt.suptitle(f'Time {data.t[it]/c.year:9.3e} yr, Mass = {data.m[it, im]:9.3e} g, Distance, {data.r[it, ir]/c.au:9.3e} AU')

    # Density distribution
    if plot_grainsize:
        if len(np.unique(data.RhoSolid)) > 1:
            print('OBS - not constant solid dust density')
        grain_yaxis = 1e1 * (3 * data.m / (4 * np.pi * np.unique(data.RhoSolid)[0]))**(1/3)
        plt00 = ax00.contourf(data.r[it, ...]/c.au,
                    grain_yaxis[it, ...],
                    np.log10(data.sigmaDust[it, ...].T),
                    levels=levels,
                    cmap="magma",
                    extend="both")
        ax00.set_ylabel("Particle size [mm]")


    else:
        plt00 = ax00.contourf(data.r[it, ...]/c.au,
                        data.m[it, ...],
                        np.log10(data.sigmaDust[it, ...].T),
                        levels=levels,
                        cmap="magma",
                        extend="both")
        ax00.set_ylabel("Particle mass [g]")

    if show_St1:
        ax00.contour(data.r[it, ...]/c.au,
                    data.m[it, ...],
                    data.St[it, ...].T,
                    levels=[1.],
                    colors="white",
                    linewidths=2
                    )
    if show_limits:
        ax00.contour(data.r[it, ...]/c.au,
                    data.m[it, ...],
                    (data.St - data.StDr[..., None])[it, ...].T,
                    levels=[0.],
                    colors="C2",
                    linewidths=1
                    )
        ax00.contour(data.r[it, ...]/c.au,
                    data.m[it, ...],
                    (data.St - data.StFr[..., None])[it, ...].T,
                    levels=[0.],
                    colors="C0",
                    linewidths=1
                    )

    ax00.axhline(data.m[it, im], color="#AAAAAA", lw=1, ls="--")
    ax00.axvline(data.r[it, ir]/c.au, color="#AAAAAA", lw=1, ls="--")

    cbar00 = plt.colorbar(plt00, ax=ax00)
    cbar00.ax.set_ylabel("$\sigma_\mathrm{d}$ [g/cm²]")
    cbar00ticklabels = []
    for i in levels:
        cbar00ticklabels.append("$10^{{{:d}}}$".format(int(i)))
    cbar00.ax.set_yticklabels(cbar00ticklabels)
    ax00.set_xscale("log")
    ax00.set_yscale("log")
    ax00.set_xlabel("Distance from star [AU]")

    if plot_grainsize:
        ax00.set_yticks(np.logspace(-3, 2, base=10, num=6));

    if return_data:
        return data

plot.my_density = my_density
