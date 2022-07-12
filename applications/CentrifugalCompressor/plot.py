import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, folder, settings='normal', cmap='viridis'):
        script_dir = os.path.dirname(__file__)
        self.jpeg_dir = os.path.join(script_dir, folder, 'jpeg')
        self.tiff_dir = os.path.join(script_dir, folder, 'tiff')
        self.cmap = cmap
        if not os.path.isdir(self.jpeg_dir):
            os.makedirs(self.jpeg_dir)
        if not os.path.isdir(self.tiff_dir):
            os.makedirs(self.tiff_dir)

        if settings == 'large':
            config = {  # setup matplotlib to use latex for output
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "font.family": "DejaVu Sans",
                "axes.titlesize": 36,
                "axes.labelsize": 36,
                "font.size": 36,
                "legend.fontsize": 24,
                "legend.frameon": True,
                "xtick.labelsize": 24,
                "ytick.labelsize": 24,
                "xtick.major.pad": 12,
                "ytick.major.pad": 12,
                "figure.autolayout": True,
                "figure.figsize": (9.6, 7.2)}
        else:
            config = {  # setup matplotlib to use latex for output
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "font.family": "DejaVu Sans",
                "axes.titlesize": 24,
                "axes.labelsize": 24,
                "font.size": 24,
                "legend.fontsize": 18,
                "legend.frameon": True,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
                "xtick.major.pad": 8,
                "ytick.major.pad": 8,
                "figure.autolayout": True,
                "figure.figsize": (9.6, 7.2)}

        mpl.rcParams.update(config)

    def plot_feature_distribution(self, X):

        n_plots = 15
        new_colors = [plt.get_cmap(self.cmap)(1. * i / n_plots) for i in range(n_plots)]
        sns.set_style("darkgrid")
        sns.set_context("paper")

        fig, axs = plt.subplots(5, 3, figsize=(8, 10))

        ax00 = sns.histplot(X[:, 0], ax=axs[0, 0], color=new_colors[0])
        ax00.set_xlabel(r'$\phi_\mathrm{t,1}$')
        ax00.set_yticklabels([])
        ax00.set_ylabel('')

        ax01 = sns.histplot(X[:, 1], ax=axs[0, 1], color=new_colors[1])
        ax01.set_xlabel(r'$\psi_\mathrm{is}$')
        ax01.set_yticklabels([])
        ax01.set_ylabel('')

        ax02 = sns.histplot(X[:, 2], ax=axs[0, 2], color=new_colors[2])
        ax02.set_xlabel(r'$\alpha_2$')
        ax02.set_yticklabels([])
        ax02.set_ylabel('')

        ax10 = sns.histplot(X[:, 3], ax=axs[1, 0], color=new_colors[3])
        ax10.set_xlabel('$R_3 / R_2$')
        ax10.set_yticklabels([])
        ax10.set_ylabel('')

        ax11 = sns.histplot(X[:, 4], ax=axs[1, 1], color=new_colors[4])
        ax11.set_xlabel('$k$')
        ax11.set_yticklabels([])
        ax11.set_ylabel('')

        ax12 = sns.histplot(X[:, 5], ax=axs[1, 2], color=new_colors[5])
        ax12.set_xlabel(r'$N_\mathrm{bl}$')
        ax12.set_yticklabels([])
        ax12.set_ylabel('')

        ax20 = sns.histplot(X[:, 6], ax=axs[2, 0], color=new_colors[6])
        ax20.set_xlabel(r'$H_\mathrm{r,pinch}$')
        ax20.set_yticklabels([])
        ax20.set_ylabel('')

        ax21 = sns.histplot(X[:, 7], ax=axs[2, 1], color=new_colors[7])
        ax21.set_xlabel(r'$R_\mathrm{r,pinch}$')
        ax21.set_yticklabels([])
        ax21.set_ylabel('')

        ax22 = sns.histplot(X[:, 8], ax=axs[2, 2], color=new_colors[8])
        ax22.set_xlabel(r'$\beta_\mathrm{tt,target}$')
        ax22.set_yticklabels([])
        ax22.set_ylabel('')

        ax30 = sns.histplot(X[:, 9], ax=axs[3, 0], color=new_colors[9])
        ax30.set_xlabel(r'$\dot{m}$')
        ax30.set_yticklabels([])
        ax30.set_ylabel('')

        ax31 = sns.histplot(X[:, 10], ax=axs[3, 1], color=new_colors[10])
        ax31.set_xlabel('$N$')
        ax31.set_yticklabels([])
        ax31.set_ylabel('')

        ax32 = sns.histplot(X[:, 11], ax=axs[3, 2], color=new_colors[11])
        ax32.set_xlabel(r'$R_\mathrm{shaft}$')
        ax32.set_yticklabels([])
        ax32.set_ylabel('')

        ax40 = sns.histplot(X[:, 12], ax=axs[4, 0], color=new_colors[12])
        ax40.set_xlabel(r'$\epsilon_\mathrm{t}/H_2$')
        ax40.set_yticklabels([])
        ax40.set_ylabel('')

        ax41 = sns.histplot(X[:, 13], ax=axs[4, 1], color=new_colors[13])
        ax41.set_xlabel(r'$\epsilon_\mathrm{b}/H_2$')
        ax41.set_yticklabels([])
        ax41.set_ylabel('')

        ax42 = sns.histplot(X[:, 14], ax=axs[4, 2], color=new_colors[14])
        ax42.set_xlabel(r'$\gamma_{Pv}$')
        ax42.set_yticklabels([])
        ax42.set_ylabel('')

        fig.tight_layout()
        fig.savefig(self.jpeg_dir + '/norm_features_distribution.jpeg', dpi=400)
        fig.savefig(self.tiff_dir + '/norm_features_distribution.tiff')
        plt.close(fig)

    def plot_constraints_distribution(self, Y_con):

        n_plots = 6
        new_colors = [plt.get_cmap(self.cmap)(1. * i / n_plots) for i in range(n_plots)]
        sns.set_style("darkgrid")
        sns.set_context("paper")

        fig, axs = plt.subplots(2, 3, figsize=(8, 10))

        ax00 = sns.histplot(Y_con[:, 0], ax=axs[0, 0], color=new_colors[0])
        ax00.set_xlabel(r'$\Omega$')
        ax00.set_yticklabels([])
        ax00.set_ylabel('')

        ax01 = sns.histplot(Y_con[:, 1], ax=axs[0, 1], color=new_colors[1])
        ax01.set_xlabel(r'$F_\mathrm{ax}$')
        ax01.set_yticklabels([])
        ax01.set_ylabel('')

        ax02 = sns.histplot(Y_con[:, 2], ax=axs[0, 2], color=new_colors[2])
        ax02.set_xlabel(r'$R_\mathrm{1,h}$')
        ax02.set_yticklabels([])
        ax02.set_ylabel('')

        ax10 = sns.histplot(Y_con[:, 3], ax=axs[1, 0], color=new_colors[3])
        ax10.set_xlabel('$H_2$')
        ax10.set_yticklabels([])
        ax10.set_ylabel('')

        ax11 = sns.histplot(Y_con[:, 4], ax=axs[1, 1], color=new_colors[4])
        ax11.set_xlabel(r'$\beta_\mathrm{2,bl}$')
        ax11.set_yticklabels([])
        ax11.set_ylabel('')

        ax12 = sns.histplot(Y_con[:, 5], ax=axs[1, 2], color=new_colors[5])
        ax12.set_xlabel('$R_4$')
        ax12.set_yticklabels([])
        ax12.set_ylabel('')

        fig.tight_layout()
        fig.savefig(self.jpeg_dir + '/norm_con_distribution.jpeg', dpi=400)
        fig.savefig(self.tiff_dir + '/norm_con_distribution.tiff')
        plt.close(fig)

    def plot_objectives_distribution(self, Y_obj):
        n_plots = 4
        new_colors = [plt.get_cmap(self.cmap)(1. * i / n_plots) for i in range(n_plots)]
        sns.set_style("darkgrid")
        sns.set_context("paper")

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        ax00 = sns.histplot(Y_obj[:, 0], ax=axs[0, 0], color=new_colors[0])
        ax00.set_xlabel(r'$\beta_\mathrm{tt}$')
        ax00.set_yticklabels([])
        ax00.set_ylabel('')

        ax01 = sns.histplot(Y_obj[:, 1], ax=axs[0, 1], color=new_colors[1])
        ax01.set_xlabel(r'$\eta_\mathrm{tt}$')
        ax01.set_yticklabels([])
        ax01.set_ylabel('')

        ax10 = sns.histplot(Y_obj[:, 2], ax=axs[1, 0], color=new_colors[2])
        ax10.set_xlabel('$OR$')
        ax10.set_yticklabels([])
        ax10.set_ylabel('')

        ax11 = sns.histplot(Y_obj[:, 3], ax=axs[1, 1], color=new_colors[3])
        ax11.set_xlabel(r'$\dot{m}_\mathrm{choke}$')
        ax11.set_yticklabels([])
        ax11.set_ylabel('')

        fig.tight_layout()
        fig.savefig(self.jpeg_dir + '/norm_obj_distribution.jpeg', dpi=400)
        fig.savefig(self.tiff_dir + '/norm_obj_distribution.tiff')
        plt.close(fig)

    def plot_error_distribution_obj(self, error, pct_error, lb, ub, lb_pct, ub_pct):
        """
        :param error: array of absolute errors evaluated over the test set
        :param pct_error: array of relative percentage errors evaluated over the test set
        """
        n_plots = 4
        fig1, axs1 = plt.subplots(2, 2, figsize=(8, 10))
        fig2, axs2 = plt.subplots(2, 2, figsize=(8, 10))

        new_colors = [plt.get_cmap(self.cmap)(1. * i / n_plots) for i in range(n_plots)]

        sns.set_style("darkgrid")
        sns.set_context("paper")

        # plot absolute error distribution
        ax00 = sns.histplot(error[:, 0], ax=axs1[0, 0], color=new_colors[0])
        ax00.set_xlabel('beta tt [-]')
        ax00.set_yticklabels([])
        ax00.set_ylabel('')
        ax01 = sns.histplot(error[:, 1], ax=axs1[0, 1], color=new_colors[1])
        ax01.set_xlabel('eta tt [-]')
        ax01.set_yticklabels([])
        ax01.set_ylabel('')
        ax10 = sns.histplot(error[:, 2], ax=axs1[1, 0], color=new_colors[2])
        ax10.set_xlabel('OR [-]')
        ax10.set_yticklabels([])
        ax10.set_ylabel('')
        ax11 = sns.histplot(error[:, 3], ax=axs1[1, 1], color=new_colors[3])
        ax11.set_xlabel('m choke [kg/s]')
        ax11.set_yticklabels([])
        ax11.set_ylabel('')

        fig1.tight_layout()
        fig1.savefig(self.jpeg_dir + '/error_distribution_obj.jpeg', dpi=400)
        fig1.savefig(self.tiff_dir + '/error_distribution_obj.tiff')
        plt.close(fig1)

        # plot relative percentage error distribution
        ax00 = sns.histplot(pct_error[:, 0], ax=axs2[0, 0], color=new_colors[0])
        ax00.set_xlabel('beta tt [%]')
        ax00.set_yticklabels([])
        ax00.set_ylabel('')
        ax00.set_xlim(-15, 15)
        ax01 = sns.histplot(pct_error[:, 1], ax=axs2[0, 1], color=new_colors[1])
        ax01.set_xlabel('eta tt [%]')
        ax01.set_yticklabels([])
        ax01.set_ylabel('')
        ax01.set_xlim(-15, 15)
        ax10 = sns.histplot(pct_error[:, 2], ax=axs2[1, 0], color=new_colors[2])
        ax10.set_xlabel('OR [%]')
        ax10.set_yticklabels([])
        ax10.set_ylabel('')
        ax10.set_xlim(-15, 15)
        ax11 = sns.histplot(pct_error[:, 3], ax=axs2[1, 1], color=new_colors[3])
        ax11.set_xlabel('m_choke [%]')
        ax11.set_yticklabels([])
        ax11.set_ylabel('')
        ax11.set_xlim(-15, 15)

        fig2.tight_layout()
        fig2.savefig(self.jpeg_dir + '/pct_error_distribution_obj.jpeg', dpi=400)
        fig2.savefig(self.tiff_dir + '/pct_error_distribution_obj.tiff')
        plt.close(fig2)

    def plot_eta_max(self, beta_tt, mass_flow, eta_max, eta_max_fit, levels=40):

        fig1, ax1 = plt.subplots(figsize=(9.6, 8.0))

        CS1 = ax1.contourf(beta_tt, mass_flow, eta_max, levels, cmap=self.cmap, origin='upper')
        ax1.set_xlabel(r'$\beta_\mathrm{tt}$ [-]')
        ax1.set_ylabel(r'$\dot{m}$ [kg/s]')
        cbar = fig1.colorbar(CS1, shrink=1.0)
        cbar.ax.set_xlabel(r'$\eta_\mathrm{tt,max}$ [\%]')

        fig1.savefig(self.jpeg_dir + '/eta_max.jpeg', dpi=400)
        fig1.savefig(self.tiff_dir + '/eta_max.tiff')
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(9.6, 8.0))

        CS2 = ax2.contourf(beta_tt, mass_flow, eta_max_fit, levels, cmap=self.cmap, origin='upper')
        ax2.set_xlabel(r'$\beta_\mathrm{tt}$ [-]')
        ax2.set_ylabel(r'$\dot{m}$ [kg/s]')
        cbar = fig2.colorbar(CS2, shrink=1.0)
        cbar.ax.set_xlabel(r'$\eta_\mathrm{tt,max,fit}$ [\%]')

        fig2.savefig(self.jpeg_dir + '/eta_max_fit.jpeg', dpi=400)
        fig2.savefig(self.tiff_dir + '/eta_max_fit.tiff')
        plt.close(fig2)
