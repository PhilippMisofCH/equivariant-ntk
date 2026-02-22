import numpy as onp
import jax.numpy as jnp
from jax import vmap


def vmap_subarray_fn(fn, arr_axes, fn_axes, axes_sizes=None):
    """Create a multidimensional vmapped version of a function of a single array.

    Can be used to create a function that should act on each subarray. Ellipsis is not supported.

    Args:
        fn: function that acts on a single array
        arr_axes: string of the form 'ij..->ijk..' that describes input dimensions and output
            dimensions of the the function that will be created.
        fn_axes: string of the form 'ij..->jik..' that describes the current dimensions mapped by
            fn. WARNING: doesn't handle scalar output yet.
        axes_sizes: shape of the array that the function will be vectorized over. Only needed if
        fn has no input arguments to vectorize over.

    Returns:
        A vmapped function having input and output dimensions given by arr_axes.
    """

    # 'ijkl->ijmkl', 'lj->jlm'
    fn_in, fn_out = fn_axes.split("->")
    arr_in, arr_out = arr_axes.split("->")

    if fn_in == "":
        assert len(arr_in) == len(axes_sizes), (
            "If the function has no input arguments, "
            "axes_sizes must be provided with the same length "
            "as arr_axes."
        )

    remain_in_ind = arr_in
    remain_out_ind = arr_out
    remain_in_ind = remove_characters(remain_in_ind, fn_in)
    remain_out_ind = remove_characters(remain_out_ind, fn_out)
    assert sorted(remain_out_ind) == sorted(remain_in_ind), (
        "Dimensions that are not explicit "
        "dimensions created by the output "
        "of the function have to appear in "
        "both the input and output "
        "dimension of the array."
    )

    found_in_axes = []
    for ax in fn_in:
        ind = arr_in.find(ax)
        assert ind != -1, (
            "Not all function input dimensions are found in input array specifier."
        )
        found_in_axes.append(ind)
    found_in_axes = onp.array(found_in_axes)
    # example: found [3,1]

    found_out_axes = []
    for ax in fn_out:
        ind = arr_out.find(ax)
        assert ind != -1
        found_out_axes.append(ind)

    found_out_axes = onp.array(found_out_axes)
    # example: found [1,4,2]

    in_transp_order = found_in_axes.argsort().argsort()
    out_transp_order = found_out_axes.argsort().argsort()

    if fn_in == "":

        def fn_tr(**kwargs):
            return jnp.transpose(fn(**kwargs), out_transp_order)
    else:

        def fn_tr(array, **kwargs):
            y = fn(jnp.transpose(array, in_transp_order), **kwargs)
            return jnp.transpose(y, out_transp_order)
    # example: signature 'jl->jml'

    in_mapped = "".join([ax for ax in arr_in if ax in fn_in])
    out_mapped = "".join([ax for ax in arr_out if ax in fn_out])

    for i, ax in enumerate(arr_in):
        if ax not in fn_in:
            in_ind = get_relative_string_pos(ax, arr_in, in_mapped)
            out_ind = get_relative_string_pos(ax, arr_out, out_mapped)

            if fn_in == "":
                fn_tr = vmap(fn_tr, None, out_ind, axis_size=axes_sizes[i])
            else:
                fn_tr = vmap(fn_tr, in_ind, out_ind)

            in_mapped = in_mapped[:in_ind] + ax + in_mapped[in_ind:]
            out_mapped = out_mapped[:out_ind] + ax + out_mapped[out_ind:]

    return fn_tr


def get_relative_string_pos(x, final, current):
    assert x not in current
    assert x in final

    counter = 0
    for ch in final:
        if ch == x:
            return counter
        elif ch in current:
            counter += 1


def remove_characters(s, chs):
    for ch in chs:
        s = s.replace(ch, "")
    return s
