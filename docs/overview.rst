Project Overview
================

Introduction
------------

The *visualizations* project is a framework written in python for animating Matplotlib based components with data primarily from neural network models.  


Motivation
----------

There exists a wide variety of visualization libraries that allow for the creation of moving or animated plots (such as Plotly Dash), however I found that many of these libraries were limited for my particular use case. I needed to display multiple custom plots simultaneously while having the ability to control animation rate, component updates, a various display properties.  

The goal of this project is to create a convienent framework for **combining** and **animating** matplotlib components concurrently. The product is akin to a dashboard with the single purpose of displaying multiple plots, each with many different sources of **sequential** data.  


Features
--------

Component subplots may be updated on a step or sub-step schedule, meaning that select subplots can display information at different rates or resolutions. The sub-step animation support provides an additional layer of flexibilty when displaying sequential data that may have been collected at different intervals.  

See the :doc:`examples` section for more videos and screenshots generated using this framework.  

subnet_plot
~~~~~~~~~~~

This module contains the :class:`SubnetPlot` and :class:`SubnetSubplot` template classes that encapsulate the base functionality of this framework. The SubnetPlot class contains and manages child SubnetSubplots. All custom plot components (Bar, Line, Matrix, etc.) inherit from SubnetSubplot.  

The :doc:`usage` guide covers the included subplot component.  

subnet_visualize
~~~~~~~~~~~~~~~~

The visualize module contains a collection of combined visualizations for neural networks implemented with the subnet_plot module. This module may be used as an example of ways to combine and animate the base components from subnet_plot.  

subnet_animate
~~~~~~~~~~~~~~

The subnet_animate module enables animations to be rendered and saved as .mp4 video files.  