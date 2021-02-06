Getting Started
===============

Before using this framework, it is strongly suggested that you start by familiarizing yourself with Matplotlib's pyplot interface using the following `Matplotlib <https://matplotlib.org/3.1.1/tutorials/introductory/pyplot.html>`_ tutorial.

For ease of use, it is recommended that you setup a virtual environment to install and run this framework. 

If you haven't worked with virtual envionrmnets before, you may use a platform like `Anaconda or Miniconda <https://docs.anaconda.com/>`_


Installation
------------

Use the included requirements.txt to install all required dependencies::

    pip install -r requirements.txt
    
Alternatively, if you want to run just the base subnet_plot module, you may be able to simply install Matplotlib::
    
    pip install Matplotlib
    
Once you have installed the dependencies you can run some sample animations by calling subnet_plot. See :doc:`examples` for more details.  


Subplot Components
------------------

A small selection of custom components are included to serve as an example of how to implement Matplotlib objects into this framework.  See below for a brief description of each class along with links to the relevant Matplotlib documentation.  

.. image:: /images/line_bar_plot.png

Above, a simple combined bar and line plot from :meth:`display_sample_plots`.



Bar subplot 
~~~~~~~~~~~
The bar subplot displays primary elements using the supplied colormap while displaying the optional secondary elements in grayscale.

.. image:: /images/bar_plot.png

The :class:`SubplotBar` component is based on the `Matplotlib bar <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.bar.html>`_ object.


Line subplot 
~~~~~~~~~~~~
The line subplot combines the basic line and bar plots. The optional bar data source is rendered behind the primary line data.

.. image:: /images/line_plot.png

The :class:`SubplotLine` component is based on the `Matplotlib plot <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html>`_ object.


Matrix subplot 
~~~~~~~~~~~~~~
The matrix subplot renders a 2-dimensional array in sequential steps with optional 1-dimensional arrays affixed to the left and top sides of the central matrix.

.. image:: /images/matrix_plot.png

The :class:`SubplotMatrix` component is based on the `Matplotlib image <https://matplotlib.org/api/image_api.html>`_ object. Also see `imshow <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.imshow.html>`_.


Text subplot 
~~~~~~~~~~~~
The text subplot displays data using a text-based readout. The primary data source is rendered within a table, while a secondary data source is displayed in rows below the primary table.

.. image:: /images/text_plot.png

The :class:`SubplotText` component is based on the `Matplotlib table <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.table.html>`_ object.


Layouts and gridspec
~~~~~~~~~~~~~~~~~~~~
The figure's grid-based layout is defined by the gridspec object passed to :meth:`add_subplot` when adding subplot objects to the parent SubnetPlot.  

.. image:: /images/combined_plot.png

For a complete description of Matplotlib's gridspec please take a look at the `tutorial <https://matplotlib.org/3.1.1/tutorials/intermediate/gridspec.html>`_.


Styles and color maps
~~~~~~~~~~~~~~~~~~~~~
Each subplot component's style is defined by the tempalte SubnetSubplot class. Colormaps are used extensively to render plot elements. See the `colormap <https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html>`_ reference for valid values.
