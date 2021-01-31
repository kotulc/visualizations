# Visualizations

The *visualizations* project is a framework written in python for animating Matplotlib based components with data primarily from neural network models.  

## Description

The goal of this project is to create a convienent framework for combining and animating matplotlib components concurrently. The product is akin to a dashboard with the single purpose of displaying multiple plots, each with many different sources of sequential information. Component subplots may be updated on a step or sub-step schedule, meaning that select subplots can display information at different rates or resolutions. The sub-step animation support provides an additional layer of flexibilty when displaying sequential data that may have been collected at different intervals. For more information please see the project wiki page.  

## Installation

This project was developed and tested with Python 3.7.3 and 3.7.9. Compatibility with any other version is not guaranteed. See the included requirements.txt file for package dependencies.  

## Usage

To run the basic collection of sample plots simply call subnet_plot.py and each component plot will be displayed and animated one at a time using the default arguments static=0 and steps=10. static=1 generates static plots while static=0 generates animated plots. the 'steps' argument determines the number of animation steps each plot iterates through. All data points are from a random distribution.  

## Credits

This project was developed and tested by Clayton Kotulak.  

## License

This project is licensed under the terms of the GNU General Public License v3.0. See [gnu.org](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.  