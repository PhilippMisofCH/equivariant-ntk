from neural_tangents import Kernel
from neural_tangents._src.stax.requirements import layer


@layer
def GroupPool():
    """Pooling over entire group orbit.
    Returns:
        Tuple[Callable, Callable, Callable]: Returns initialization, apply, and kernel function
            (init_fn, apply_fn, kernel_fn).
    """

    def init_fn(rng, input_shape):
        """No parameters to be initialized
        Args:
            input_shape (tuple): shape of input features of form `BGHWI`
        Returns:
            output (Tuple[int]): output shape
        """
        b, g, h, w, c = input_shape
        assert g == 4, "Input has to have `G` dimension of 4."

        output_shape = (b, c)

        return output_shape, ()

    def apply_fn(_, x, **kwargs):
        """Apply group pooling
        Args:
            x (jnp.array): Input features of form `BGHWI` where `I` is number of in-channels
        Returns:
            output (jnp.array): output of pooling of form BI.
        """
        return x.mean(axis=(1, 2, 3))

    def kernel_fn(k: Kernel, **kwargs):
        cov1, nngp, cov2, ntk, is_reversed = (k.cov1, k.nngp, k.cov2, k.ntk, k.is_reversed)

        def group_mean(kernel):
            if kernel is None:
                return None
            return kernel.mean(axis=(-1, -2, -3, -4, -5, -6))

        return k.replace(
            cov1=group_mean(cov1),
            nngp=group_mean(nngp),
            cov2=group_mean(cov2),
            ntk=group_mean(ntk),
            is_gaussian=True,
            is_reversed=is_reversed,
            batch_axis=0,
            channel_axis=1,
            is_input=False,
        )

    return init_fn, apply_fn, kernel_fn
