Examples
========

This section will cover running the included scripts from the command line and some basic use cases. See :doc:`cli` for command line options.

**Note:** Each plot type will be rendered after the previous plot has finished animating. If you close a plot before it has finished animating the program will terminate.


Display sample plots
--------------------

Call subnet_plot without arguments to use the default settings and generate a series of sample plot objects.

Simply run::

    python subnet_plot.py

to call display_sample_plots.


Display static plots
--------------------

To display the sample plots without animations,
use :option:`-a`:

.. code-block:: 

    python subnet_plot.py -a 0

This will disable animations, rendering the plots at the final step.


Display a variable number of steps
----------------------------------

You may use the :option:`-s` flag to change the number of animation steps displayed for each plot.

    python subnet_plot.py -s 100

This will run 100 animation steps per plot.
