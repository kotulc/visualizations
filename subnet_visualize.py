"""
Visualize the network data in an animated or static plot
"""


import argparse
import matplotlib.gridspec as gs
import numpy as np
import subnet_plot as sp


# ---------------------- Plot definitions ----------------------


class ActivationPlot(sp.SubnetPlot):
    """Each layer column contains feedback, activations and readout rows"""
    def __init__(self, data_dict, layer_keys):
        step_count, substep_dim = data_dict[layer_keys[0]]['activations'].shape[:2]
        super().__init__(
            step_count * substep_dim, gs_rows=3, gs_cols=len(layer_keys) * 2,
            substep_dim=substep_dim
        )
        # For each layer, add a column plot mapping its step activations
        for layer_idx in range(len(layer_keys)):
            activation_readout_colplot(
                data_dict, layer_keys[layer_idx], self,
                self.grid_spec[:, layer_idx * 2:layer_idx * 2 + 2]
            )


class LayerPlot(sp.SubnetPlot):
    """Display a 3 row plot with line, matrix, and bar rowplots"""
    def __init__(self, data_dict, layer_keys, primary_key, selected_idx=0):
        plot_cols = len(layer_keys) + 1
        step_count, substep_dim = data_dict[layer_keys[0]]['activations'].shape[:2]
        super().__init__(
            step_count * substep_dim, gs_rows=3, gs_cols=plot_cols,
            substep_dim=substep_dim
        )
        # The first subplot spans the top row
        layer_activation_step_rowplot(
            data_dict, layer_keys, primary_key, selected_idx,
            self, self.grid_spec[0, :]
        )
        # The next subplot occupies the remaining left-half of the plot
        for key_idx in range(len(layer_keys)):
            layer_weight_step_colplot(
                data_dict, layer_keys[key_idx], self,
                self.grid_spec[1:, key_idx]
            )
        # The final subplot column occupies the right-half
        layer_image_readout_colplot(
            data_dict, primary_key, selected_idx, self, self.grid_spec[1:, -1]
        )


class LossPlot(sp.SubnetPlot):
    """Display a 3 row plot with line, matrix, and bar rowplots"""
    def __init__(self, data_dict, layer_key):
        step_count = data_dict[layer_key]['activations'].shape[0]
        super().__init__(
            step_count, gs_rows=1, gs_cols=3, pause_intervals=(0.0001, 0.00000001)
        )
        loss_array = data_dict[layer_key]['loss']
        loss_line_series = sp.line_series(
            {'loss': loss_array}, '{} Loss'.format(layer_key), 'sample', 'loss',
            color_map='RdBu', x_limit=loss_array.shape[0], y_line=0
        )
        inner_grid = gs.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=self.grid_spec[:, :], wspace=0.2, hspace=0.1
        )
        self.add_subplot(loss_line_series, inner_grid[0, 0])


class NodePlot(sp.SubnetPlot):
    """Display a 3 row plot with line, matrix, and bar rowplots"""
    def __init__(self, data_dict, layer_key, selected_idx=0):
        step_count, substep_dim = data_dict[layer_key]['activations'].shape[:2]
        super().__init__(
            step_count * substep_dim, gs_rows=3, gs_cols=3,
            substep_dim=substep_dim
        )
        node_potential_block_rowplot(
            data_dict, layer_key, self, self.grid_spec[0, :], selected_idx
        )
        node_input_block_rowplot(
            data_dict, layer_key, self, self.grid_spec[1, :], selected_idx
        )
        node_weight_block_rowplot(
            data_dict, layer_key, self, self.grid_spec[2, :], selected_idx
        )


# ----------------- Subplot row and column definitions ----------------


def activation_readout_colplot(data_dict, layer_key, subnet_plot, subplot_spec):
    """Add a 3 rowed column activation plot to subnet_plot"""
    inner_grid = gs.GridSpecFromSubplotSpec(
        5, 4, subplot_spec=subplot_spec, wspace=0.4, hspace=0.8
    )
    # Add the target image matrix to the middle-left
    image_array = data_dict['input_image']
    matrix_series = sp.matrix_series(
        image_array, 'Input Image', center_color_map='cividis'
    )
    subnet_plot.add_subplot(matrix_series, inner_grid[:2, :2])
    #
    '''# Add the weight matrix subplot to the second column
    matrix_array = data_dict[layer_key]['fb_image']
    matrix_series = sp.matrix_series(
        matrix_array, '{} Feedback'.format(layer_key),
        center_color_map='plasma'
    )
    subnet_plot.add_subplot(matrix_series, inner_grid[2, :2])'''
    #
    # Add the text step subplot to the final row
    table_dict, table_labels, column_dict = table_row_readout_dicts(
        data_dict, layer_key, selected_idx=0
    )
    step_text_series = sp.text_series(
        table_dict, '{} Readout'.format(layer_key),
        table_labels, text_col_dict=column_dict
    )
    subnet_plot.add_subplot(step_text_series, inner_grid[:2, 2:])
    #
    # Add the weight matrix subplot to the second column
    input_array = data_dict[layer_key]['inputs']
    mask_array = data_dict[layer_key]['mask']
    matrix_array = data_dict[layer_key]['activations']
    matrix_series = sp.matrix_series(
        matrix_array, '{} Activations'.format(layer_key),
        center_color_map='seismic', left_array=mask_array,
        left_color_map='viridis', substep_idx=True, top_array=input_array,
        top_color_map='cividis'
    )
    subnet_plot.add_subplot(matrix_series, inner_grid[2:, :3], layer_idx=2)
    #
    # Add the weight matrix subplot to the second column
    matrix_array = data_dict[layer_key]['encoding']
    matrix_series = sp.matrix_series(
        matrix_array, '{} Encoding'.format(layer_key),
        center_color_map='viridis'
    )
    subnet_plot.add_subplot(matrix_series, inner_grid[2:, 3:], layer_idx=1)


def layer_activation_step_rowplot(data_dict, layer_keys, primary_key,
                                  selected_idx, subnet_plot, subplot_spec):
    """Return a plot containing activation and potential line subplots"""
    inner_grid = gs.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=subplot_spec, wspace=0.2, hspace=0.1
    )
    #
    # Add the activation line plot with segmentation
    block_steps = data_dict['block_count'] * data_dict['block_steps']
    k = 'a.idx={}'.format(selected_idx)
    line_dict = {
        k: data_dict[primary_key]['activations'][:, :, selected_idx],
        'a.mean': np.mean(data_dict[primary_key]['activations'], axis=2),
        'p.mean': np.mean(data_dict[primary_key]['potentials'], axis=2)
    }
    seg_dict = segmentation_dict(line_dict, block_steps)
    actv_line_series = sp.line_bar_series(
        line_dict, '{} Activation and Potentials'.format(primary_key), 'step', 'value',
        bar_array_dict=seg_dict, color_map='gist_heat', substep_idx=True
    )
    subnet_plot.add_subplot(actv_line_series, inner_grid[:, 0])
    #
    # Add the potential line plot with segmentation
    line_dict = {
        'loss': data_dict[primary_key]['loss'],
        'weight delta': data_dict[primary_key]['update']
    }
    seg_dict = segmentation_dict(line_dict, block_steps)
    loss_line_series = sp.line_bar_series(
        line_dict, '{} Loss'.format(primary_key), 'step', 'loss',
        bar_array_dict=seg_dict, color_map='viridis'
    )
    subnet_plot.add_subplot(loss_line_series, inner_grid[:, 1])


def layer_image_readout_colplot(data_dict, layer_key,
                                selected_idx, subnet_plot, subplot_spec):
    """Return a plot containing input, weight and step subplots"""
    inner_grid = gs.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=subplot_spec, wspace=0.3, hspace=0.5
    )
    #
    # Add the weight matrix subplot to the second column
    '''image_array = data_dict[layer_key]['fb_image']
    image_matrix_series = sp.matrix_series(image_array, 'Weighted Feedback')
    subnet_plot.add_subplot(image_matrix_series, inner_grid[0, 0])'''
    #
    # Add the target image matrix to be reconstructed
    image_array = data_dict['input_image']
    image_matrix_series = sp.matrix_series(image_array, 'Input Image')
    subnet_plot.add_subplot(image_matrix_series, inner_grid[0, :])
    #
    # Add the text step subplot to the final column
    table_dict, table_labels, column_dict = table_row_readout_dicts(
        data_dict, layer_key, selected_idx
    )
    step_text_series = sp.text_series(
        table_dict, '{} Readout'.format(layer_key),
        table_labels, text_col_dict=column_dict
    )
    subnet_plot.add_subplot(step_text_series, inner_grid[1, :])


def layer_weight_step_colplot(data_dict, layer_key, subnet_plot, subplot_spec):
    """Return a 2-row column plot containing weight matrix and input plots"""
    inner_grid = gs.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=subplot_spec, wspace=0.3, hspace=0.5
    )
    #
    # Add the weight matrix subplot to the first row
    weight_array, left_array, top_array = weight_matrix_arrays(
        data_dict[layer_key]
    )
    weight_matrix_series = sp.matrix_series(
        weight_array, '{} Weights (input, output)'.format(layer_key),
        left_array=left_array, substep_idx=True, top_array=top_array,
        top_color_map='jet'
    )
    subnet_plot.add_subplot(weight_matrix_series, inner_grid[0, 0])
    #
    # Add the weighted input bar subplot to the second row
    primary_trace, secondary_trace = bar_overlaid_dicts(
        data_dict[layer_key], 'weighted', 'potentials'
    )
    layer_bar_series = sp.bar_series(
        primary_trace, '{} Input'.format(layer_key),
        'node', 'weighted input', secondary_array_dict=secondary_trace,
        substep_idx=True, y_log=True
    )
    subnet_plot.add_subplot(layer_bar_series, inner_grid[1, 0])


def node_potential_block_rowplot(data_dict, focus_key, subnet_plot,
                                 subplot_spec, selected_idx):
    """Return a plot containing activation and potential line subplots"""
    inner_grid = gs.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=subplot_spec, wspace=0.2, hspace=0.1
    )
    # Add the activation line plot with segmentation
    block_steps = data_dict['block_count'] * data_dict['block_steps']
    line_dict = {
        'activation': data_dict[focus_key]['activations'][:, :, selected_idx],
        'potential': data_dict[focus_key]['potentials'][:, :, selected_idx]
    }
    seg_dict = segmentation_dict(line_dict, block_steps)
    line_series = sp.line_bar_series(
        line_dict, 'Node Activation and Potential', 'step', 'value',
        bar_array_dict=seg_dict, color_map='gist_heat',
        substep_idx=True
    )
    subnet_plot.add_subplot(line_series, inner_grid[:, 0])
    # Add the potential line plot with segmentation
    line_dict = {
        'modifier': data_dict[focus_key]['modifier'],
        'threshold': data_dict[focus_key]['thresholds'][:, :, selected_idx]
    }
    seg_dict = segmentation_dict(line_dict, block_steps)
    line_series = sp.line_bar_series(
        line_dict, 'Node Thresholds', 'step', 'value',
        bar_array_dict=seg_dict, color_map='viridis',
        substep_idx=True, y_log=True
    )
    subnet_plot.add_subplot(line_series, inner_grid[:, 1])


def node_input_block_rowplot(data_dict, focus_key, subnet_plot,
                             subplot_spec, selected_idx):
    """Return a plot containing input, weight and step subplots"""
    inner_grid = gs.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=subplot_spec, wspace=0.3, hspace=0.1
    )
    #
    # Add the weighted input bar subplot to the first row
    gain_trace = {'gain': data_dict[focus_key]['gain']}
    potentials_trace = {'potentials': data_dict[focus_key]['potentials']}
    # Display bar with log y-axis
    bar_series = sp.bar_series(
        gain_trace, 'Layer {} Gain'.format(focus_key), 'node',
        'gain', secondary_array_dict=potentials_trace,
        y_log=True, substep_idx=True
    )
    subnet_plot.add_subplot(bar_series, inner_grid[0, 0:2])
    #
    # Add the weight matrix subplot to the second column
    weight_array, left_array, top_array = weight_matrix_arrays(
        data_dict[focus_key]
    )
    matrix_series = sp.matrix_series(
        weight_array, 'Layer {} Weights (input, output)'.format(focus_key),
        left_array=left_array, substep_idx=True, top_array=top_array,
        top_color_map='jet'
    )
    subnet_plot.add_subplot(matrix_series, inner_grid[0, 2])
    #
    # Add the text step subplot to the final column
    table_dict, table_labels, column_dict = table_row_readout_dicts(
        data_dict, focus_key, selected_idx
    )
    text_series = sp.text_series(
        table_dict, 'Node Readout', table_labels, text_col_dict=column_dict
    )
    subnet_plot.add_subplot(text_series, inner_grid[0, 3])


def node_weight_block_rowplot(data_dict, focus_key, subnet_plot,
                              subplot_spec, selected_idx, step_delta=50):
    """Return a plot containing a single weight subplot for the selected index"""
    inner_grid = gs.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=subplot_spec, wspace=0.1, hspace=0.1
    )
    #
    # Add the weighted input bar subplot to the second row
    input_dict, weight_dict = bar_delta_overlaid_dicts(
        data_dict[focus_key], 'inputs', 'weights', selected_idx, step_delta
    )
    # Display bar with log y-axis
    bar_series = sp.bar_series(
        input_dict, 'Node Weights [idx={}]'.format(selected_idx),
        'weight', 'strength', secondary_array_dict=weight_dict, y_log=True,
        substep_idx=True
    )
    subnet_plot.add_subplot(bar_series, inner_grid[0, 0])


# --------------------- Data plot trace definitions -------------------


def bar_delta_overlaid_dicts(array_dict, primary_key,
                             secondary_key, selected_idx, step_delta):
    """Return a bar graph with secondary_array overlaid with the primary_array"""
    primary_dict = {primary_key: array_dict[primary_key]}
    # Base array mask, offset secondary arrays by delta
    samples, block_steps, elements = array_dict[primary_key].shape
    secondary_array = array_dict[secondary_key][:, :, selected_idx]
    block_array = np.ones((block_steps, *secondary_array.shape))
    block_array *= secondary_array
    block_array = block_array.reshape(samples, block_steps, -1)
    delta_indices = np.arange(-1 * step_delta, block_array.shape[0] - step_delta)
    delta_indices = delta_indices * (delta_indices > 0).astype(np.int)
    delta_weights = block_array[delta_indices, :]
    secondary_dict = {
        '{} (i-{})'.format(secondary_key, step_delta): delta_weights,
        '{} (i)'.format(secondary_key): block_array
    }
    return primary_dict, secondary_dict


def bar_overlaid_dicts(array_dict, primary_key, secondary_key):
    """Return an overlaid bar graph with input, activations and potentials"""
    primary_array = array_dict[primary_key]
    primary_dict = {'{}'.format(primary_key): primary_array}
    # Display secondary overlaid bars (grey)
    secondary_array = array_dict[secondary_key]
    secondary_dict = {'{}'.format(secondary_key): secondary_array}
    return primary_dict, secondary_dict


def line_weight_delta_dict(array_dict, array_keys, array_subkey, delta_steps=50):
    """Calculate the difference between array_subkey offset by block_steps"""
    array_steps, line_dict = 0, {}
    for array_key in array_keys:
        delta_array = np.array(array_dict[array_key][array_subkey])
        # Shift the base array to the left by repeating the first block
        delta_array[delta_steps:] = delta_array[:-1 * delta_steps]
        delta_array[:delta_steps] = 0
        # Delta array is the mean difference over block_steps
        delta_array = array_dict[array_key][array_subkey] - delta_array
        line_dict[array_key] = np.max(np.abs(delta_array), axis=(1, 2))
    return line_dict


def line_mean_dict(array_dict, primary_key, mean_line=True, selected_idx=0):
    """Return a dictionary of lines and a step segmentation dict"""
    line_dict = {}
    # Reshape data for visualization
    primary_array = array_dict[primary_key]
    selected_key = '{} idx={}'.format(primary_key, selected_idx)
    if len(primary_array.shape) == 3:
        selected_array = primary_array[:, :, selected_idx]
    else:
        selected_array = primary_array[:, selected_idx]
    line_dict[selected_key] = selected_array
    if mean_line:
        # Display the mean of the primary array
        mean_key = '{} mean'.format(primary_key)
        if len(primary_array.shape) == 3:
            line_dict[mean_key] = np.mean(primary_array, axis=2)
        else:
            line_dict[mean_key] = np.mean(primary_array, axis=1)
    return line_dict


def line_dict(array_dict, array_key, selected_idx=0):
    """Return a dictionary of lines and a step segmentation dict"""
    line_array = array_dict[array_key].astype(np.float)
    assert len(line_array.shape) == 2
    # Package with segmentation plots
    mean_str = '{} mean'.format(array_key)
    select_str = '{} idx={}'.format(array_key, selected_idx)
    line_dict = {
        select_str: line_array[:, selected_idx],
        mean_str: np.mean(line_array, axis=1)
    }
    return line_dict


def table_row_readout_dicts(data_dict, focus_key, selected_idx):
    """Return the text dictionary and column labels for the text subplot"""
    table_dict = basic_stats_dict(data_dict[focus_key], selected_idx)
    table_labels = ['selected', 'min', 'max', '||x||']
    column_dict = {'target': data_dict['target']}
    return table_dict, table_labels, column_dict


# -------------------------- Data transforms --------------------------


def basic_stats_dict(data_dict, selected_idx):
    """Return the selected, min, max, mean of all data_dict array values"""
    stat_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            v = v.astype(np.float)
            # Index selection using the final array axis
            if len(v.shape) >= 3:
                mean_axis = tuple([i for i in range(1, len(v.shape) - 1)])
                # Average over all remaining axes besides the first (step)
                selected_array = np.mean(v[:, :, selected_idx], axis=mean_axis)
            elif len(v.shape) == 2:
                selected_array = v[:, selected_idx]
            else:
                selected_array = v
            axis = tuple(range(1, len(v.shape)))
            #
            selected_array = selected_array.reshape(v.shape[0], 1)
            min_array = np.min(v, axis=axis).reshape(v.shape[0], 1)
            max_array = np.max(v, axis=axis).reshape(v.shape[0], 1)
            mean_array = np.mean(np.abs(v), axis=axis).reshape(v.shape[0], 1)
            stat_dict[k] = np.hstack(
                (selected_array, min_array, max_array, mean_array)
            )
    return stat_dict


def block_to_step_dict(block_steps, data_dict):
    """Convert all arrays from block-based arrays to step-based"""
    for k, v in data_dict.items():
        if isinstance(v, dict):
            block_to_step_dict(block_steps, v)
        elif isinstance(v, np.ndarray):
            if len(v.shape) >= 3 and v.shape[1] == block_steps:
                array_rows = v.shape[0] * v.shape[1]
                data_dict[k] = v.reshape((array_rows, *v.shape[2:]))
            else:
                data_dict[k] = extend_block_array(v, block_steps)
    return data_dict


def extend_block_array(block_array, block_steps):
    step_rows, step_columns = block_array.shape[0], block_array.shape[1:]
    new_array = np.zeros((block_steps * step_rows, *step_columns))
    for row_idx in range(block_array.shape[0]):
        start_idx = row_idx * block_steps
        end_idx = start_idx + block_steps
        new_array[start_idx:end_idx] = block_array[row_idx]
    return new_array


def ooc_target_array(data_dict):
    """Return a mask array with single out-of-class target selections"""
    target_array = data_dict['targets']
    target_idx = np.argmax(target_array, axis=1)
    ooc_targets = np.zeros(target_array.shape)
    ooc_targets[:, 0] = target_idx > 0
    ooc_targets[:, 1] = target_idx == 0
    return ooc_targets


def sample_seg_array(array_steps, block_steps):
    """Return a binary segmentation array"""
    seg_array, seg_flag = np.ones(array_steps), True
    for step_idx in range(1, array_steps):
        if step_idx % block_steps == 0:
            seg_flag = not seg_flag
        seg_array[step_idx] = float(seg_flag)
    return seg_array


def sample_sigma_arrays(data_dict):
    """Return the in-class and mean gain arrays"""
    if len(data_dict['targets'].shape) == 3:
        target_array = data_dict['targets'][:, 0, :]
        weighted_array = np.mean(data_dict['weighted'][:, :, 0, :], axis=1)
    else:
        target_array = data_dict['targets']
        weighted_array = data_dict['weighted'][:, 0, :]
    ic_sigma = np.sum(weighted_array * target_array, axis=1)
    avg_sigma = np.mean(weighted_array, axis=1)
    return ic_sigma, avg_sigma


def segmentation_dict(array_dict, segment_steps):
    """Return a segmented array of array_steps length with matching dimensions"""
    abs_max = np.max([np.abs(v) for v in array_dict.values()])
    if abs_max == 0:
        abs_max = 1
    min_val = np.min([v for v in array_dict.values()])
    first_array = next(iter(array_dict.values()))
    array_steps = first_array.shape[0]
    segment_sign = 1
    # Sign array contains segment_steps runs of 1 and -1
    if len(first_array.shape) == 2:
        sign_array = np.ones(first_array.shape[:2])
        for step_idx in range(array_steps):
            sign_array[step_idx] *= segment_sign
            segment_sign *= -1
    else:
        sign_array = np.ones(array_steps)
        for step_idx in range(1, array_steps):
            if step_idx % segment_steps == 0:
                segment_sign *= -1
            sign_array[step_idx] *= segment_sign
    seg_dict = {'+': (sign_array > 0) * abs_max}
    if min_val < 0:
        seg_dict['-'] = (sign_array > 0) * abs_max * -1
    return seg_dict


def weight_matrix_arrays(data_dict):
    """Return the weight matrix and arrays for the matrix subplot"""
    weight_array = data_dict['weights']
    left_array = data_dict['mask']
    if 'activations' not in data_dict:
        top_array = data_dict['gain']
    else:
        top_array = data_dict['activations']
    # If block dimension (4d arrays) average over block value
    if len(left_array.shape) == 3:
        left_array = np.mean(left_array, axis=1)
    return weight_array, left_array, top_array


# -------------------- Display plot from arguments --------------------


def visualize_data(data_dict, plot_type,
                   selected_idx, selected_key, static_plot=False):
    """Visualize all elements in data_dict in a single combined plot"""
    if plot_type == 'node':
        subnet_plot = NodePlot(data_dict, selected_key, selected_idx)
    elif plot_type == 'layer':
        # Get list of layer dictionary keys to plot
        subnet_plot = LayerPlot(
            data_dict, data_dict['layer_keys'], selected_key, selected_idx
        )
    elif plot_type == 'loss':
        subnet_plot = LossPlot(data_dict, selected_key)
    else:
        subnet_plot = ActivationPlot(data_dict, data_dict['layer_keys'])
    sp.display_plot(subnet_plot, static_plot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-file', default=io.SAVE_PATH + 'ckernel_log.pkl', required=False,
        help='Data dictionary file path'
    )
    parser.add_argument(
        '-plot_type', default='node', required=False,
        help='node, layer, loss, or activation plot'
    )
    parser.add_argument(
        '-selected_idx', default='0', required=False, help='focus node index'
    )
    parser.add_argument(
        '-selected_key', default='L0', required=False, help='focus layer key'
    )
    parser.add_argument(
        '-static_plot', default='0', required=False,
        help='0: Dynamic plot, 1: Static plot'
    )
    input_args = parser.parse_args()
    data_dict = io.load_pickle(input_args.file)
    if data_dict is None:
        print("Error: File {} not found".format(input_args.file))
    else:
        visualize_data(
            data_dict=data_dict,
            plot_type=input_args.plot_type,
            selected_idx=int(input_args.selected_idx),
            selected_key=input_args.selected_key,
            static_plot=bool(int(input_args.static_plot))
        )


if __name__ == '__main__':
    main()
