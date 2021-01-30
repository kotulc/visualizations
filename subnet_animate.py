"""
Display matplotlib based animations from neural network data
"""


from matplotlib import animation
import subnet_io as io
import matplotlib.pyplot as plt
import subnet_visualize as sv


class SubnetPlotAnimator(object):
    def __init__(self, subnet_plot_class, data_dict):
        plt.close('all')
        self.subnet_plot = subnet_plot_class(data_dict)
        self.figure = self.subnet_plot.figure

    def plot_animate(self, frame_idx):
        return self.subnet_plot.update(frame_idx)

    def plot_init(self):
        self.subnet_plot.update(0)
        return self.subnet_plot.dynamic_components

    def run_animation(self, frames, interval, save_path=None):
        fig_animation = animation.FuncAnimation(
            fig=self.figure, func=self.plot_animate, frames=frames,
            init_func=self.plot_init, blit=True, interval=interval,
            repeat=False
        )
        if save_path is not None:
            fig_animation.save(
                save_path, fps=5,
                extra_args=['-vcodec', 'libx264']
            )
        plt.show()


if __name__ == '__main__':
    print('Preparing animation...')
    plot_type = 'node'
    save_path = io.SAVE_PATH + '{}_av7.3_slow.mp4'.format(plot_type)
    # Path to pickled subnet data dict export
    if plot_type == 'layer':
        data_dict = io.load_pickle(io.SAVE_PATH + 'layer_log.pkl')
        data_dict = sv.block_to_step_dict(data_dict['block_steps'], data_dict)
        plot_obj = sv.LayerPlot
    else:
        data_dict = io.load_pickle(io.SAVE_PATH + 'node_log.pkl')
        plot_obj = sv.NodePlot
    subnet_animator = SubnetPlotAnimator(plot_obj, data_dict)
    subnet_animator.run_animation(500, interval=500, save_path=save_path)
    print('Operation completed successfully.')
