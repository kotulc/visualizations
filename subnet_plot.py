"""
Collection of data visualizations for neural networks
"""


from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# ----------------- Define mixed subplot visualizations ---------------


class SubnetPlot(object):
    """Define base template row-column grid plot of subplot objects"""
    def __init__(self, step_count, figure_size=None, gs_dim=4, gs_cols=1,
                 gs_rows=1, pause_intervals=(0.25, 0.0001),
                substep_dim=1, window_title='Subnet Plot'):
        # Size of drawn plot as (width_units, height_units)
        if figure_size is None:
            # Determine dimensions by plot rows and columns
            fig_width = min(16, gs_dim * gs_cols)
            fig_height = min(9, gs_dim * gs_rows)
            figure_size = (fig_width, fig_height)
        self.figure = plt.figure(figsize=figure_size)
        self.cols, self.rows = gs_cols, gs_rows
        self.pause_intervals = pause_intervals
        # Setup template GridSpec based layout
        self.grid_spec = GridSpec(
            ncols=gs_cols, nrows=gs_rows, figure=self.figure, left=0.06,
            right=0.94, top=0.92, bottom=0.08, hspace=0.50, wspace=0.20
        )
        self.step_count = step_count
        self.substep_dim = substep_dim
        self.figure.canvas.set_window_title(window_title)
        # Plot dynamic components must be redrawn on each call to update()
        self.dynamic_components, self.layer_steps = [], []
        self.subplot_layers, self.component_layers = [], []

    def add_subplot(self, series_subplot, subplot_gspec, layer_idx=0):
        """Initialize subplot_class with the series dict and add to plot"""
        # Series tuples include the subplot data, class and args
        series_dict, subplot_class, subplot_kwargs = series_subplot
        # Add the plot-wide substep dimension to the subplot dict
        subplot_kwargs['substep_dim'] = self.substep_dim
        subplot_ax = self.figure.add_subplot(subplot_gspec)
        # Initialize subplot_class within this figure at subplot_ax
        sp_object = subplot_class(series_dict, subplot_ax, subplot_kwargs)
        # Expand layers if layer index is not within list bounds
        if layer_idx >= len(self.subplot_layers):
            self.component_layers.append([])
            self.subplot_layers.append([])
            # Layer specific update counts
            self.layer_steps.append(0)
            layer_idx = len(self.subplot_layers) - 1
        # Add subplot and the subplots dynamic components to the layer
        self.subplot_layers[layer_idx].append(sp_object)
        self.component_layers[layer_idx].extend(sp_object.dynamic_components)
        self.dynamic_components.extend(sp_object.dynamic_components)

    def update(self, step_idx):
        """Call the update method of each subplot in the selected layer"""
        # Update the subplots contained in each layer
        if step_idx == 0:
            for layer in self.subplot_layers:
                for subplot in layer:
                    subplot.update(0)
            return self.dynamic_components
        else:
            substep_flag = False
            step_offset = step_idx // self.substep_dim
            layer_idx = step_offset % len(self.subplot_layers)
            for subplot in self.subplot_layers[layer_idx]:
                subplot.update(self.layer_steps[layer_idx])
                if subplot.substep_idx:
                    substep_flag = True
            if substep_flag:
                plt.pause(self.pause_intervals[1])
            if step_idx % self.substep_dim == 0:
                plt.pause(self.pause_intervals[0])
            self.layer_steps[layer_idx] += 1
            return self.component_layers[layer_idx]

    def draw(self, step_idx):
        """Update all subplot components to display step_idx data points"""
        for layer in self.subplot_layers:
            for subplot in layer:
                if isinstance(subplot, SubplotLine):
                    # Update line buffer sequentially for proper display
                    start_idx = 0
                    if step_idx > subplot.x_limit:
                        start_idx = step_idx - subplot.x_limit
                    step_array = np.arange(start_idx, step_idx)
                    for s in step_array:
                        subplot.update(s)
                else:
                    subplot.update(step_idx)


class SampleColumnPlot(SubnetPlot):
    """Sample column-prime composite plot with substep indexing"""
    def __init__(self, plot_steps):
        # Substep subplots are updated on a substep_dim frequency
        substep_dim = 32
        step_count = plot_steps * substep_dim
        super().__init__(
            step_count, gs_cols=3, gs_rows=1, substep_dim=substep_dim
        )
        # Add the sample bar subplot to be updated on the standard frequency
        add_sample_bar_subplot(self, self.grid_spec[0, :2], plot_steps)
        # Initialize a combined line/bar subplot with substep indexing
        line_dict = {'line series': np.random.randn(plot_steps, substep_dim)}
        bar_dict = {'bar series': np.random.randn(plot_steps, substep_dim)}
        series_subplot = line_bar_series(
            line_dict, 'Line bar plot', 'Step x', 'Rand y',
            bar_array_dict=bar_dict, substep_idx=True
        )
        # Add the line subplot to the plot grid spec and pass the data series
        self.add_subplot(series_subplot, self.grid_spec[0, 2])


class SampleRowPlot(SubnetPlot):
    """Sample row-prime composite plot"""
    def __init__(self, plot_steps):
        substep_dim = 32
        step_count = plot_steps * substep_dim
        super().__init__(
            step_count, gs_cols=3, gs_rows=3, substep_dim=substep_dim
        )
        # First subplot row
        # Initialize a combined line/bar subplot with substep indexing
        line_dict = {'line series': np.random.randn(plot_steps, substep_dim)}
        bar_dict = {'bar series': np.random.randn(plot_steps, substep_dim)}
        series_subplot = line_bar_series(
            line_dict, 'Line bar plot', 'Step x', 'Rand y',
            bar_array_dict=bar_dict, substep_idx=True
        )
        # Add the line subplot to the plot grid spec and pass the data series
        self.add_subplot(series_subplot, self.grid_spec[0, :], layer_idx=0)
        #
        # Second subplot row
        sample_matrix = np.random.rand(plot_steps, substep_dim, 25)
        left_array = np.random.rand(plot_steps, substep_dim, 25)
        top_array = np.random.rand(plot_steps, substep_dim, 25)
        series_subplot = matrix_series(
            sample_matrix, 'Matrix plot (row, col)',
            center_color_map='viridis',
            left_color_map='cividis', left_array=left_array,
            top_array=top_array, top_color_map='cividis', substep_idx=True
        )
        self.add_subplot(series_subplot, self.grid_spec[1, 0], layer_idx=0)
        add_sample_matrix_subplot(
            self, self.grid_spec[1, 1], plot_steps, layer_idx=1
        )
        add_sample_text_subplot(
            self, self.grid_spec[1, 2], plot_steps, layer_idx=1
        )
        #
        # Last subplot row
        add_sample_bar_subplot(
            self, self.grid_spec[2, :], plot_steps, layer_idx=1
        )


# ------------------- Subplot component definitions -------------------


class SubnetSubplot(object):
    """Base axes style, layout, and options for component subplots"""
    def __init__(self, series_arrays, subplot_axes, axis='on', grid=True,
                 substep_dim=1, substep_idx=False, title=None, x_label=None,
                 y_label=None, y_lim=True, y_line=None, y_log=False):
        self.series_arrays = series_arrays
        self.subplot_axes = subplot_axes
        self.substep_dim = substep_dim
        self.substep_idx = substep_idx
        if substep_idx:
            # If substep indexing is enabled, verify array dimension match
            assert series_arrays[0].shape[1] >= self.substep_dim
        # Set base style from kwargs here
        subplot_axes.autoscale(True, axis='y')
        subplot_axes.axis(axis)
        subplot_axes.set_anchor('SW')
        subplot_axes.set_frame_on(False)
        subplot_axes.tick_params(labelsize=5)
        if grid:
            subplot_axes.grid(True, axis='y', color='lightgray', which='both')
        if title is not None:
            subplot_axes.set_title(title, loc='left', pad=14)
        if x_label is not None:
            subplot_axes.set_xlabel(x_label, fontsize='x-small')
        if y_label is not None:
            subplot_axes.set_ylabel(y_label, va='bottom', fontsize='x-small')
        # Calculate plot y limits based on data source
        if y_lim is not False:
            if isinstance(y_lim, tuple):
                y_max, y_min = y_lim
                subplot_axes.set_ylim(y_lim)
            else:
                # Estimate the y limit values
                y_max, y_min = 0.0, 1.0
                for series_array in series_arrays:
                    y_max = max((y_max, float(series_array.max())))
                    y_min = min((y_min, float(series_array.min())))
                if y_min < 0:
                    # Center the y axis if plot contains negative values
                    y_max = max(abs(y_max), abs(y_min))
                    y_min = -1 * y_max
                    subplot_axes.set_ylim([y_min, y_max])
                else:
                    subplot_axes.set_ylim([y_min, y_max])
            subplot_axes.axhline(y_max, color='gray')
            subplot_axes.axhline(y_min, color='gray')
        if y_line is not None:
            subplot_axes.axhline(
                y_line, alpha=0.25, color='steelblue', dashes=(4, 3)
            )
        if y_log:  # Scale the y-axis on a symmetric log
            subplot_axes.set_yscale('symlog')
        # Components to be redrawn after each update() call
        self.dynamic_components = []

    def add_legend(self, label_columns=None):
        if label_columns is None:
            label_columns = len(self.series_arrays)
        # bbox_to_anchor=(1.01, 1.14)
        self.subplot_axes.legend(
            fontsize=6, frameon=False, loc='upper right',
            ncol=label_columns
        )

    def step_array(self, series_array, step_idx):
        """Return the selected array with the given indexing pattern"""
        if self.substep_idx:
            # If substep indexing is enabled, adjust step array indexing
            substep_idx = step_idx % self.substep_dim
            return series_array[step_idx // self.substep_dim, substep_idx]
        return series_array[step_idx // self.substep_dim]

    def update(self, step_idx):
        """Update all dynamic elements in this subplot"""
        # Process and store current step array as self.step_array
        return self.dynamic_components


class SubplotBar(SubnetSubplot):
    """Display array data points from data_series_dict as a bar plot"""
    def __init__(self, data_series_dict, subplot_axes, subplot_kwargs):
        series_arrays = data_series_dict['arrays']
        super().__init__(series_arrays, subplot_axes, **subplot_kwargs)
        # Verify the array dimensions match the plot type
        array_dim = len(series_arrays[0].shape)
        assert array_dim == 2 or (self.substep_idx and array_dim == 3)
        if self.substep_idx:
            # If substep indexing is enabled, adjust array axis accordingly
            self.bar_series_count = series_arrays[0].shape[2]
        else:
            self.bar_series_count = series_arrays[0].shape[1]
        x = np.arange(self.bar_series_count)
        for series_idx, series_array in enumerate(series_arrays):
            bar_container = subplot_axes.bar(
                x, np.zeros(self.bar_series_count),
                alpha=data_series_dict['alphas'][series_idx],
                color=data_series_dict['colors'][series_idx],
                label=data_series_dict['labels'][series_idx]
            )
            self.dynamic_components.extend(bar_container)
        self.add_legend()

    def update(self, step_idx):
        """Update all dynamic elements in this subplot"""
        for array_idx, series_array in enumerate(self.series_arrays):
            step_array = self.step_array(series_array, step_idx)
            for x_idx in range(self.bar_series_count):
                rect_idx = self.bar_series_count * array_idx + x_idx
                rect = self.dynamic_components[rect_idx]
                rect.set_height(step_array[x_idx])
        return self.dynamic_components


class SubplotLine(SubnetSubplot):
    def __init__(self, data_series_dict, subplot_axes, subplot_kwargs):
        self.bar_series_count = data_series_dict['bar_count']
        self.series_dict = data_series_dict
        self.x_limit = data_series_dict['x_limit']
        series_arrays = data_series_dict['arrays']
        # Initialize data buffers for each series array in the subplot
        self.line_arrays = [np.zeros(self.x_limit) for a in series_arrays]
        super().__init__(series_arrays, subplot_axes, **subplot_kwargs)
        # Verify the array dimensions match the plot type
        array_dim = len(series_arrays[0].shape)
        assert array_dim == 1 or (self.substep_idx and array_dim == 2)
        for series_idx, series_array in enumerate(series_arrays):
            # Add bar arrays to the plot background
            if series_idx < self.bar_series_count:
                base_fill = subplot_axes.fill_between(
                    np.arange(self.x_limit), self.line_arrays[series_idx],
                    alpha=data_series_dict['alphas'][series_idx],
                    color=data_series_dict['colors'][series_idx],
                    label=data_series_dict['labels'][series_idx],
                    step="pre"
                )
                self.dynamic_components.append(base_fill)
            else:
                plot_lines = subplot_axes.plot(
                    np.arange(self.x_limit), self.line_arrays[series_idx],
                    alpha=data_series_dict['alphas'][series_idx],
                    color=data_series_dict['colors'][series_idx],
                    label=data_series_dict['labels'][series_idx],
                    linewidth=1.6
                )
                self.dynamic_components.extend(plot_lines)
        self.add_legend()

    def update(self, step_idx):
        """Update all dynamic elements in this subplot"""
        array_idx, start_idx = min(step_idx, self.x_limit - 1), 0
        # Calculate the first index of the x-axis
        if step_idx > self.x_limit:
            start_idx = step_idx - self.x_limit
        end_idx = start_idx + self.x_limit
        self.subplot_axes.set_xlim(start_idx, end_idx)
        x = np.arange(start_idx, end_idx)
        # Background bar plots need to be redrawn after each update
        if self.bar_series_count > 0:
            self.subplot_axes.collections.clear()
        for series_idx, series_array in enumerate(self.series_arrays):
            if array_idx == self.x_limit - 1:
                # Shift line array to the left by a single element
                shift_array = self.line_arrays[series_idx][1:]
                self.line_arrays[series_idx][:-1] = shift_array
            # Extract the next step array value and update line array
            step_value = self.step_array(series_array, step_idx)
            self.line_arrays[series_idx][array_idx] = step_value
            if series_idx < self.bar_series_count:
                input_fill = self.subplot_axes.fill_between(
                    x, self.line_arrays[series_idx],
                    alpha=self.series_dict['alphas'][series_idx],
                    color=self.series_dict['colors'][series_idx],
                    label=self.series_dict['labels'][series_idx],
                    step="pre"
                )
                self.dynamic_components[series_idx] = input_fill
            else:
                self.dynamic_components[series_idx].set_data(
                    x, self.line_arrays[series_idx]
                )
        return self.dynamic_components


class SubplotMatrix(SubnetSubplot):
    def __init__(self, data_series_dict, subplot_axes, subplot_kwargs):
        self.center_array = data_series_dict['center_array']
        self.left_array = data_series_dict['left_array']
        self.top_array = data_series_dict['top_array']
        series_arrays = (self.center_array, self.left_array, self.top_array)
        super().__init__(series_arrays, subplot_axes, **subplot_kwargs)
        # Verify the array dimensions match the plot type
        array_dim = len(series_arrays[0].shape)
        assert array_dim == 3 or (self.substep_idx and array_dim == 4)
        self.substep_count = 1
        if self.substep_idx:
            # Calculate the number of animation steps in each substep
            self.substep_count = series_arrays[0].shape[2] // self.substep_dim
        subplot_axes.set_title(
            subplot_kwargs['x_label'], fontdict={'fontsize': 'small'}, pad=20
        )
        val_min, val_max = np.min(self.center_array), np.max(self.center_array)
        # Buffer arrays for substep animation
        self.center_buffer = np.zeros(self.center_array.shape[1:])
        center_im = self.subplot_axes.imshow(
            self.center_buffer, cmap=data_series_dict['center_color_map'],
            interpolation='nearest', vmin=val_min, vmax=val_max, aspect='auto'
        )
        self.dynamic_components = [center_im]
        # Add bottom, left, and top sub-axes to the primary axes
        divider = make_axes_locatable(self.subplot_axes)
        bottom_ax = divider.append_axes("bottom", size="5%", pad=0.05)
        left_ax = divider.append_axes("left", size="6%", pad=0.05)
        top_ax = divider.append_axes("top", size="6%", pad=0.04)
        # Add a vertical image vector to the left of the image
        if self.left_array is not None:
            val_min, val_max = np.min(self.left_array), np.max(self.left_array)
            self.left_buffer = np.zeros(self.left_array.shape[-1]).reshape(-1, 1)
            left_im = left_ax.imshow(
                self.left_buffer, cmap=data_series_dict['left_color_map'],
                interpolation='nearest', aspect='auto',
                vmin=val_min, vmax=val_max
            )
            self.dynamic_components.append(left_im)
        left_ax.axis('off')
        # Add a horizontal image vector to the top of the subplot
        if self.top_array is not None:
            val_min, val_max = np.min(self.top_array), np.max(self.top_array)
            self.top_buffer = np.zeros(self.top_array.shape[-1]).reshape(1, -1)
            # Add top activation bar
            top_im = top_ax.imshow(
                self.top_buffer, cmap=data_series_dict['top_color_map'],
                interpolation='nearest', aspect='auto',
                vmin=val_min, vmax=val_max
            )
            self.dynamic_components.append(top_im)
        top_ax.axis('off')
        # Add color bar scale to the bottom of the matrix
        color_bar = plt.colorbar(center_im, cax=bottom_ax, orientation='horizontal')
        color_bar.ax.tick_params(labelsize=5)

    def update(self, step_idx):
        """Update all dynamic components of this object"""
        if self.substep_idx:
            # If substep indexing is enabled, adjust step array indexing
            substep_idx = step_idx % self.substep_dim
            outerstep_idx = step_idx // self.substep_dim
            start_idx = substep_idx * self.substep_count
            end_idx = start_idx + self.substep_count
            step_array = self.center_array[outerstep_idx, :, start_idx:end_idx]
            if step_idx % self.substep_dim == 0:
                self.center_buffer *= 0
            if step_idx % self.substep_dim == self.substep_dim - 1:
                # Fill in the rest of the matrix on the final step
                step_array = self.center_array[outerstep_idx, :, start_idx:]
                self.center_buffer[:, start_idx:] = step_array
            else:
                self.center_buffer[:, start_idx:end_idx] = step_array
        else:
            self.center_buffer = self.step_array(self.center_array, step_idx)
        self.dynamic_components[0].set_data(self.center_buffer)
        # Update left and top bar array values if enabled
        bar_idx = 1
        if self.left_array is not None:
            # Do not use step_array() for the left array, always display static
            step_array = self.left_array[step_idx // self.substep_dim]
            self.dynamic_components[bar_idx].set_data(step_array.reshape(-1, 1))
            bar_idx += 1
        if self.top_array is not None:
            step_array = self.step_array(self.top_array, step_idx)
            self.dynamic_components[bar_idx].set_data(step_array.reshape(1, -1))
        return self.dynamic_components


class SubplotText(SubnetSubplot):
    def __init__(self, data_series_dict, subplot_axes, subplot_kwargs):
        self.table_row_count = data_series_dict['table_row_count']
        series_arrays = data_series_dict['arrays']
        super().__init__(series_arrays, subplot_axes, **subplot_kwargs)
        # Verify the array dimensions match the plot type
        array_dim = len(series_arrays[0].shape)
        assert array_dim == 2 or (self.substep_idx and array_dim == 3)
        subplot_axes.set_title(
            subplot_kwargs['x_label'], fontdict={'fontsize': 'small'},
            loc='center', pad=10
        )
        column_labels, row_labels = data_series_dict['labels']
        # Verify length of column and rows match label lists
        assert len(series_arrays) == len(row_labels)
        assert series_arrays[0].shape[1] == len(column_labels)
        # Final row contains iteration count
        row_labels.insert(self.table_row_count, '')
        row_labels.append('iteration')
        self.columns, self.rows = len(column_labels), len(row_labels)
        # Build cell text table and format labels
        cell_text = [[''] * self.columns for _ in range(self.rows)]
        plot_table = self.subplot_axes.table(
            cell_text, colLabels=column_labels, colLoc='right',
            edges='open', bbox=[0.2, 0.0, 0.8, 1.0], rowLabels=row_labels
        )
        for (row, col), cell in plot_table.get_celld().items():
            if row == 0 or col == -1:
                cell.set_text_props(
                    fontproperties=FontProperties(weight='bold')
                )
        self.dynamic_components = [plot_table]

    def update(self, step_idx):
        """Update all row and column cell data for this subplot"""
        cell_dict = self.dynamic_components[0].get_celld()
        # Update "row-col" table entries
        for row_idx in range(self.table_row_count):
            series_array = self.series_arrays[row_idx]
            step_array = self.step_array(series_array, step_idx)
            for col_idx in range(self.columns):
                cell_dict[row_idx + 1, col_idx].get_text().set_text(
                    '{:> 8.5f}'.format(step_array[col_idx])
                )
        # All remaining rows contain a single column
        if self.table_row_count < self.rows:
            # Update remaining "column" table entries
            for row_idx in range(self.table_row_count, self.rows - 2):
                series_array = self.series_arrays[row_idx]
                step_value = self.step_array(series_array, step_idx)
                cell_dict[row_idx + 2, 0].get_text().set_text(
                    '{:> 8.5f}'.format(step_value)
                )
        # Update the iteration text
        cell_dict[self.rows, 0].get_text().set_text('{:>8d}'.format(step_idx))
        return self.dynamic_components


# ------------------ Subplot data source definitions ------------------


def bar_series(bar_array_dict, title, x_label, y_label,
               color_map='gnuplot_r', secondary_array_dict=None,
               substep_idx=False, y_line=None, y_log=False):
    """Return a bar plot data series of the form (data_dict, class, kwargs)"""
    alphas, colors, labels, series_arrays = [], [], [], []
    # Get the number of bar elements from the first data array
    first_value = next(iter(bar_array_dict.values()))
    color_list = color_map_list(first_value.shape[1], color_map)
    # Secondary dictionary displays overlaid background bar entries
    if secondary_array_dict is not None:
        alpha_list = np.linspace(0.2, 0.4, len(secondary_array_dict))
        for idx, (k, v) in enumerate(secondary_array_dict.items()):
            series_arrays.append(v)
            alphas.append(alpha_list[idx])
            colors.append('grey')
            labels.append(k)
    # Data from the primary bar array is plotted in the foreground
    for idx, (k, v) in enumerate(bar_array_dict.items()):
        series_arrays.append(v)
        alphas.append(0.5)
        colors.append(color_list)
        labels.append(k)
    # Define bar subplot series data and formatting dictionaries
    series_dict = {
        'alphas': alphas, 'arrays': series_arrays, 'colors': colors,
        'labels': labels
    }
    subplot_dict = {
        'substep_idx': substep_idx, 'title': title, 'x_label': x_label,
        'y_label': y_label, 'y_line': y_line, 'y_log': y_log
    }
    return series_dict, SubplotBar, subplot_dict


def line_bar_series(line_array_dict, title, x_label, y_label,
                    bar_array_dict=None, color_map='viridis',
                    substep_idx=False, x_limit=100, y_line=None, y_log=False):
    """Return a overlaid line plot of the form (data_dict, class, kwargs)"""
    alphas, colors, labels, series_arrays = [], [], [], []
    if isinstance(bar_array_dict, dict):
        # Add a bar subplot behind the line subplot
        bar_count = len(bar_array_dict)
        total_count = len(line_array_dict) + bar_count
        color_list = color_map_list(total_count, cmap_id=color_map)
        for idx, (k, v) in enumerate(bar_array_dict.items()):
            series_arrays.append(v)
            alphas.append(0.1)
            colors.append(color_list[idx])
            labels.append(k)
    else:
        bar_count = 0
        total_count = len(line_array_dict)
        color_list = color_map_list(total_count, cmap_id=color_map)
    # Add the line subplot to the foreground
    for idx, (k, v) in enumerate(line_array_dict.items()):
        series_arrays.append(v)
        alphas.append(0.5)
        colors.append(color_list[idx + bar_count])
        labels.append(k)
    # Define line subplot series data and formatting dictionaries
    series_dict = {
        'alphas': alphas, 'arrays': series_arrays, 'bar_count': bar_count,
        'colors': colors, 'labels': labels, 'x_limit': x_limit
    }
    subplot_dict = {
        'substep_idx': substep_idx, 'title': title, 'x_label': x_label,
        'y_label': y_label, 'y_line': y_line, 'y_log': y_log
    }
    return series_dict, SubplotLine, subplot_dict


def line_series(line_array_dict, title, x_label, y_label, color_map='viridis',
                secondary_array_dict=None, secondary_color_map=None,
                substep_idx=False, x_limit=100, y_line=None, y_log=False):
    """Return a line plot data series of the form (data_dict, class, kwargs)"""
    alphas, colors, labels, series_arrays = [], [], [], []
    if isinstance(secondary_array_dict, dict):
        if secondary_color_map is None:
            secondary_color_map = color_map
        # Add a secondary group of line plots with alternate colors
        color_list = color_map_list(
            len(secondary_array_dict), cmap_id=secondary_color_map
        )
        for idx, (k, v) in enumerate(secondary_array_dict.items()):
            series_arrays.append(v)
            alphas.append(0.3)
            colors.append(color_list[idx])
            labels.append(k)
    # Add the line plots to the foreground
    color_list = color_map_list(len(line_array_dict), cmap_id=color_map)
    for idx, (k, v) in enumerate(line_array_dict.items()):
        series_arrays.append(v)
        alphas.append(0.5)
        colors.append(color_list[idx])
        labels.append(k)
    # Define line subplot series data and formatting dictionaries
    series_dict = {
        'alphas': alphas, 'arrays': series_arrays, 'bar_count': 0,
        'colors': colors, 'labels': labels, 'x_limit': x_limit
    }
    subplot_dict = {
        'substep_idx': substep_idx, 'title': title, 'x_label': x_label,
        'y_label': y_label, 'y_line': y_line, 'y_log': y_log
    }
    return series_dict, SubplotLine, subplot_dict


def matrix_series(matrix_array, title, center_color_map='plasma',
                  left_array=None, left_color_map='magma',
                  substep_idx=False, top_array=None, top_color_map='magma'):
    """Return a matrix image series with optional left and top bar arrays"""
    # Matrix subplot displays the m x n dimensional matrix in the plot center
    series_dict = {
        'center_array': matrix_array, 'center_color_map': center_color_map,
        'left_array': left_array, 'left_color_map': left_color_map,
        'top_array': top_array, 'top_color_map': top_color_map
    }
    subplot_dict = {
        'axis': 'off', 'grid': False, 'substep_idx': substep_idx,
        'x_label': title, 'y_lim': False
    }
    return series_dict, SubplotMatrix, subplot_dict


def text_series(text_table_dict, title, labels,
                substep_idx=False, text_col_dict=None):
    """Return a text plot data series of the form (data_dict, class, kwargs)"""
    table_row_count = len(text_table_dict)
    row_labels, series_arrays = [], []
    for k, v in text_table_dict.items():
        series_arrays.append(v)
        row_labels.append(k)
    if text_col_dict is not None:
        for k, v in text_col_dict.items():
            series_arrays.append(v)
            row_labels.append(k)
    # Define text subplot series data and formatting dictionaries
    series_dict = {
        'arrays': series_arrays, 'labels': (labels, row_labels),
        'table_row_count': table_row_count,
    }
    subplot_dict = {
        'axis': 'off', 'grid': False, 'substep_idx': substep_idx,
        'x_label': title, 'y_lim': False
    }
    return series_dict, SubplotText, subplot_dict


# ------------------- Data transforms and utilities -------------------


def color_map_list(value_max, cmap_id='gnuplot', value_min=0):
    """Return a list of color values from value_min to value_max"""
    cmap = mpl.cm.get_cmap(cmap_id)
    value_array = np.arange(value_min, value_max)
    norm = mpl.colors.Normalize(vmin=value_min, vmax=value_max)
    return [cmap(norm(v)) for v in value_array]


def input_delta_sum(data_dict, layer_key, input_keys):
    """Return the step-wise normalized delta input sum array"""
    input_array = data_dict[layer_key][input_keys[0]][input_keys[1]]
    # Offset the array by a single step and take the sum of the abs difference
    input_offset = np.vstack((input_array[0, :], input_array[:-1, :]))
    input_delta = np.sum(np.abs(input_offset - input_array), axis=1)
    return input_delta / input_array.shape[1]


# ----------------------- Display sample plots ------------------------


def add_sample_bar_subplot(parent_plot, plot_gs, plot_steps,
                           bar_count=25, layer_idx=0):
    """Add a sample bar subplot to parent_plot"""
    # Display a bar plot that overlays an optional bar plot
    primary_dict = {'primary bar series': np.random.randn(plot_steps, bar_count)}
    secondary_dict = {
        'bar series 1': np.random.randn(plot_steps, bar_count),
        'bar series 2': np.random.randn(plot_steps, bar_count)
    }
    # Add data to a new bar series
    series_subplot = bar_series(
        primary_dict, 'Bar plot', 'Bar x', 'Rand y',
        secondary_array_dict=secondary_dict
    )
    # Add the bar subplot to the parent plot at grid spec location
    parent_plot.add_subplot(series_subplot, plot_gs, layer_idx=layer_idx)


def add_sample_line_subplot(parent_plot, plot_gs, plot_steps, layer_idx=0):
    """Add a sample line subplot to parent_plot"""
    line_dict = {'line series': np.random.randn(plot_steps)}
    bar_dict = {'bar series': np.random.randn(plot_steps)}
    # Add input input, matrix and text plots
    series_subplot = line_bar_series(
        line_dict, 'Line bar plot', 'Step x', 'Rand y',
        bar_array_dict=bar_dict
    )
    # Add the line subplot to the plot grid spec and pass the data series
    parent_plot.add_subplot(series_subplot, plot_gs, layer_idx=layer_idx)


def add_sample_matrix_subplot(parent_plot, plot_gs, plot_steps, layer_idx=0):
    """Add a sample matrix subplot to parent_plot"""
    sample_matrix = np.random.rand(plot_steps, 25, 25)
    left_array = np.random.rand(plot_steps, 25)
    top_array = np.random.rand(plot_steps, 25)
    series_subplot = matrix_series(
        sample_matrix, 'Matrix plot (row, col)', left_array=left_array,
        top_array=top_array
    )
    # Add the line subplot to the plot grid spec and pass the data series
    parent_plot.add_subplot(series_subplot, plot_gs, layer_idx=layer_idx)


def add_sample_text_subplot(parent_plot, plot_gs, plot_steps, layer_idx=0):
    """Add a sample text subplot to parent_plot"""
    text_dict = {
        'row 1': np.random.rand(plot_steps, 3),
        'row 2': np.random.rand(plot_steps, 3),
        'row 3': np.random.rand(plot_steps, 3)
    }
    # Add input input, matrix and text plots
    series_subplot = text_series(
        text_dict, 'Sample text plot',
        labels=['col 1', 'col 2', 'col 3']
    )
    # Add the line subplot to the plot grid spec and pass the data series
    parent_plot.add_subplot(series_subplot, plot_gs, layer_idx=layer_idx)


def display_plot(subnet_plot, static_plot=True):
    """Display and optionally animate subnet_plot"""
    if static_plot:
        display_plot_static(subnet_plot)
    else:
        display_plot_steps(subnet_plot)


def display_plot_static(subnet_plot):
    """Manually update all plot components to step_idx"""
    subnet_plot.draw(step_idx=subnet_plot.step_count - 1)
    plt.show()


def display_plot_steps(subnet_plot):
    """Iterate through step_count plot component updates"""
    step_array = np.arange(0, subnet_plot.step_count)
    for step_idx in step_array:
        subnet_plot.update(step_idx=step_idx)
        plt.draw()
    plt.show()


def display_sample_plots(plot_steps, static=True):
    """Display a dynamic plot from data_dict"""
    # Display a bar plot
    sample_plot = SubnetPlot(plot_steps)
    add_sample_bar_subplot(
        sample_plot, sample_plot.grid_spec[0, 0], plot_steps
    )
    display_plot(sample_plot, static)
    #
    # Display a line plot that overlays an optional bar plot
    sample_plot = SubnetPlot(plot_steps)
    add_sample_line_subplot(
        sample_plot, sample_plot.grid_spec[0, 0], plot_steps
    )
    display_plot(sample_plot, static)
    #
    # Display a matrix-array plot with left and top array-fed bars
    sample_plot = SubnetPlot(plot_steps)
    add_sample_matrix_subplot(
        sample_plot, sample_plot.grid_spec[0, 0], plot_steps
    )
    display_plot(sample_plot, static)
    #
    # Display a grid-aligned text plot with 'column labels' as the first row
    sample_plot = SubnetPlot(plot_steps)
    add_sample_text_subplot(
        sample_plot, sample_plot.grid_spec[0, 0], plot_steps
    )
    display_plot(sample_plot, static)
    #
    # Display column and row dominant composite plots
    sample_plot = SampleRowPlot(plot_steps)
    display_plot(sample_plot, static)
    #
    sample_plot = SampleColumnPlot(plot_steps)
    display_plot(sample_plot, static)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-static', default='0', required=False,
        help='(0, 1): Display static plot containing steps data points'
    )
    parser.add_argument(
        '-steps', default='10', required=False,
        help='int: Random sample data points to plot'
    )
    input_args = parser.parse_args()
    print('subnet_plot: display plot...')
    # Display a sample plot with the given arguments
    display_sample_plots(int(input_args.steps), bool(int(input_args.static)))
    print('subnet_plot: process complete.')


if __name__ == '__main__':
    main()
