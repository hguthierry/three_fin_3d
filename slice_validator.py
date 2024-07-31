import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

from modulus.sym.utils.io.plotter import ValidatorPlotter

# define custom class
class SliceValidatorPlotter(ValidatorPlotter):

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables
        x,y,z = invar["x"][:,0], invar["y"][:,0], invar["z"][:,0]

        bool_array = ((y > 0.065) < 0.067)

        x = x[bool_array]
        z = z[bool_array]

        extent = (x.min(), x.max(), z.min(), z.max())

        # get and interpolate output variable
        u_true, u_pred = true_outvar["u"][:,0], pred_outvar["u"][:,0]
        u_true = u_true[bool_array]
        u_pred = u_pred[bool_array]
        u_true, u_pred = self.interpolate_output(x, z,
                                                [u_true, u_pred],
                                                extent,
        )

        vmin_true= u_true.min()
        vmax_true = u_true.max()
        vmin_pred= u_pred.min()
        vmax_pred = u_pred.max()


        # make plot
        f = plt.figure(figsize=(14,4), dpi=100)
        plt.suptitle("Three Fin Flow: PINN vs true solution")
        plt.subplot(1,3,1)
        plt.title("True solution (u)")
        plt.imshow(u_true.T, origin="lower", extent=extent, vmin=-0.5, vmax=2.2)
        plt.xlabel("x"); plt.ylabel("z")
        plt.colorbar()
        #plt.vlines(-0.05, -0.05, 0.05, color="k", lw=10, label="No slip boundary")
        #plt.vlines( 0.05, -0.05, 0.05, color="k", lw=10)
        #plt.hlines(-0.05, -0.05, 0.05, color="k", lw=10)
        #plt.legend(loc="lower right")
        plt.subplot(1,3,2)
        plt.title("PINN solution (u)")
        plt.imshow(u_pred.T, origin="lower", extent=extent, vmin=-0.5, vmax=2.2)
        plt.xlabel("x"); plt.ylabel("z")
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.title("Difference")
        plt.imshow((u_true-u_pred).T, origin="lower", extent=extent, vmin=-2.2, vmax=2.2)
        plt.xlabel("x"); plt.ylabel("z")
        plt.colorbar()
        plt.tight_layout()

        return [(f, "custom_plot"),]

    @staticmethod
    def interpolate_output(x, z, us, extent):
        "Interpolates irregular points onto a mesh"

        # define mesh to interpolate onto
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], 1000),
            np.linspace(extent[2], extent[3], 1000),
            indexing="ij",
        )

        # linearly interpolate points onto mesh
        us = [scipy.interpolate.griddata(
            (x, z), u, tuple(xyi)
            )
            for u in us]

        return us