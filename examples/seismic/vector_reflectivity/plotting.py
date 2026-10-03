import numpy as np
import matplotlib.pyplot as plt

def plot_model(model, title=""):
    vpmem = model.vp_m_edit_mode
    model.make_vp_editable()
    plt_options_model = {'cmap': 'jet', 'extent': [model.origin[0],
                         model.origin[0] + model.domain_size[0],
                         model.origin[1] + model.domain_size[1],
                         model.origin[1]]}

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))
    
    fig.suptitle(title, fontsize=28, y=1.05)

    slc = (slice(model.nbl, -model.nbl),) * 2

    im_vp = ax[0].imshow(np.transpose(model.vp.data[slc]), **plt_options_model)
    fig.colorbar(im_vp, ax=ax[0])
    ax[0].set_title("$V_P$ (km/s)", fontsize=20)
    ax[0].set_xlabel('x (m)', fontsize=20)
    ax[0].set_ylabel('z (m)', fontsize=20)
    ax[0].set_aspect('auto')

    plt_options_model.update({'cmap': 'grey'})
    maxr0val = np.max(np.abs(model.r[0].data))
    maxr1val = np.max(np.abs(model.r[1].data))
    maxrval = max(maxr0val, maxr1val)
    if maxrval > 0:
        e = np.floor(np.log10(maxrval))
        rscale = 10**e
    else:
        rscale = 1

    im_rx = ax[1].imshow(np.transpose(model.r[0].data[slc]/rscale), **plt_options_model)
    fig.colorbar(im_rx, ax=ax[1])
    if rscale != 1:
        ax[1].set_title("$R_x$ ($\\times 10^{" + str(int(e)) + "}$)", fontsize=20)
    else:
        ax[1].set_title("$R_x$", fontsize=20)
    ax[1].set_xlabel('x (m)', fontsize=20)
    ax[1].set_ylabel('z (m)', fontsize=20)
    ax[1].set_aspect('auto')

    im_rz = ax[2].imshow(np.transpose(model.r[1].data[slc]/rscale), **plt_options_model)
    fig.colorbar(im_rz, ax=ax[2])
    if rscale != 1:
        ax[2].set_title("$R_z$ ($\\times 10^{" + str(int(e)) + "}$)", fontsize=20)
    else:
        ax[1].set_title("$R_z$", fontsize=20)
    ax[2].set_xlabel('x (m)', fontsize=20)
    ax[2].set_ylabel('z (m)', fontsize=20)
    ax[2].set_aspect('auto')
    if vpmem == 'm':
        model.make_m_editable()