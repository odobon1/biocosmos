import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.gridspec as gridspec  # type: ignore[import]
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator  # type: ignore[import]
import numpy as np  # type: ignore[import]


def plot_metrics(
        data_tracker, 
        dpath_trial,
        fontsize_axes       =12, 
        fontsize_ticks      =8, 
        fontsize_legend     =8,
        subplot_border_width=1,
        figsize             =(10, 12),
        height_ratios       =[2, 2, 2, 2, 1],
    ):
    data = data_tracker.data

    x_len = len(data["id_img2txt_prec1"])
    x     = list(range(0, x_len))

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0)

    ##########################################################################################################
    ##########################################################################################################
    # Plot 1: mAP Scores
    ax0 = fig.add_subplot(gs[0, 0])

    ax0.plot(x, data["id_img2txt_map"],  label="ID img2txt mAP",  color="blue")
    ax0.plot(x, data["id_img2img_map"],  label="ID img2img mAP",  color="red")
    ax0.plot(x, data["id_txt2img_map"],  label="ID txt2img mAP",  color="green")
    ax0.plot(x, data["ood_img2txt_map"], label="OOD img2txt mAP", color="blue",  linestyle="--")
    ax0.plot(x, data["ood_img2img_map"], label="OOD img2img mAP", color="red",   linestyle="--")
    ax0.plot(x, data["ood_txt2img_map"], label="OOD txt2img mAP", color="green", linestyle="--")

    ax0.set_ylabel("mAP Scores", fontsize=fontsize_axes, fontweight="bold")
    ax0.set_ylim(0, 1)

    ax0.legend(loc="lower right", fontsize=fontsize_legend)
    ax0.grid(True)
    ax0.tick_params(labelbottom=False, labelsize=fontsize_ticks)
    ##########################################################################################################
    # Plot 2: mAP Composites
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    ax1.plot(x, data["comp_map"], label="Composite", color="black")
    ax1.plot(x, data["img2img_map"], label="img2img Composite", color="#B22222")

    ax1.set_ylabel("mAP Composites", fontsize=fontsize_axes, fontweight="bold")
    ax1.set_ylim(0, 1)

    ax1.legend(loc="lower right", fontsize=fontsize_legend)
    ax1.grid(True)
    ax1.tick_params(labelbottom=False, labelsize=fontsize_ticks)
    ##########################################################################################################
    # Plot 3: Precision@1
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    ax2.plot(x, data["id_img2txt_prec1"], label="ID img2txt Prec@1", color="blue")
    ax2.plot(x, data["ood_img2txt_prec1"], label="OOD img2txt Prec@1", color="blue", linestyle="--")

    ax2.set_ylabel("Precision@1", fontsize=fontsize_axes, fontweight="bold")
    ax2.set_ylim(0, 1)

    ax2.legend(loc="lower right", fontsize=fontsize_legend)
    ax2.grid(True)
    ax2.tick_params(labelbottom=False, labelsize=fontsize_ticks)
    ##########################################################################################################
    # Plot 4: Loss
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)

    ax3.plot(x, [np.nan] + data["loss_train"], label="Train Loss")
    ax3.plot(x, data["id_loss"], label="ID Val Loss")
    ax3.plot(x, data["ood_loss"], label="OOD Val Loss")
    ax3.plot(x, data["comp_loss"], label="Comp Val Loss")

    ax3.set_ylabel("Loss", fontsize=fontsize_axes, fontweight="bold")
    ax3.set_yscale("log")
    # for plotting minor gridlines on the y axis
    ax3.minorticks_on()
    ax3.grid(which="minor", axis="y")

    ax3.legend(loc="upper right", fontsize=fontsize_legend)
    ax3.grid(True)
    ax3.tick_params(labelbottom=False, labelsize=fontsize_ticks)
    ##########################################################################################################
    # Plot 5: Learning Rate
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)

    ax4.plot(x, [np.nan] + data["lr"])

    ax4.set_ylabel("Learning Rate", fontsize=fontsize_axes, fontweight="bold")
    ax4.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax4.yaxis.set_offset_position("right")
    ax4.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax4.yaxis.get_offset_text().set_visible(False)

    ax4.set_xlabel("Epochs", fontsize=fontsize_axes, fontweight="bold")  # last subgraph gets the x label
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.xaxis.set_minor_locator(NullLocator())

    ax4.grid(True)
    ax4.tick_params(labelsize=fontsize_ticks)
    ##########################################################################################################

    for ax in (ax0, ax1, ax2, ax3):
        ax.label_outer()

    # thick black borders on all subplots
    for idx_ax, ax in enumerate([ax0, ax1, ax2, ax3, ax4]):
        for spine in ax.spines.values():
            spine.set_linewidth(subplot_border_width)
            spine.set_edgecolor("black")
        if idx_ax % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    fig.suptitle("Train Metrics", fontweight="bold", y=0.98, fontsize=20)

    plt.subplots_adjust(hspace=0)  # remove extra whitespace to ensure plots are vertically flush
    plt.tight_layout()

    savepath = dpath_trial / "plots" / f"train_metrics.png"
    fig.savefig(savepath)
    plt.close(fig)
