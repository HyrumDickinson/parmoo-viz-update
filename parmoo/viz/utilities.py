""" This module contains utilities (helper functions) that are used throughout
the viz tool.

"""

import pandas as pd
import logging
from .statics import (
    valid_db_input,
    valid_output_input,
    valid_points_input,
    valid_file_type_input,
    valid_screenshot_input,
    valid_image_export_format_input,
    valid_data_export_format_input,
)


def export_file(fig, plot_name, file_type):
    """ Export image of figure to working directory.

        Args:
            fig (plotly.graph_objects.Figure): The figure to export.

            plot_name (string): Set the filename of the image file.

            file_type (string): Set the image file type.
             - 'html' - Export as .html file.
             - 'pdf' - Export as .pdf file.
             - 'svg' - Export as .svg file.
             - 'eps' - Export as .eps file
               if the poppler dependency is installed.
             - 'jpeg' - Export as .jpeg file.
             - 'png' - Export as .png file.
             - 'webp' - Export as .webp file.

    """

    # validate input
    validate_input(valid_file_type_input, file_type, "file_type")

    # export graph to file
    if file_type == 'html':
        fig.write_html(plot_name + ".html")
        logging.info("exported graph as .html")
    else:
        fig.write_image(f"{plot_name}.{file_type}")
        logging.info("exported graph as .%s", file_type)


def set_plot_name(db):
    """ Provide a default graph title.

        Args:
            db (string): Graph contents inform title.
             - 'pf' - Set plot name to "Pareto Front"
             - 'obj' - Set plot name to "Objective Data"

        Returns:
            plot_name (string): The default plot name.

    """

    # validate input
    validate_input(valid_db_input, db, "db")

    # set plot name
    if db == 'pf':
        plot_name = "Pareto Front"
    elif db == 'obj':
        plot_name = "Objective Data"
    return plot_name


def set_database(moop, db, points):
    """ Choose which points from MOOP object to plot.

        Args:
            db (string): Set dataset.
             - 'pf' - Set Pareto Front as dataset.
             - 'obj' - Set objective data as dataset.

            points (string): Filter traces from dataset by constraint score.
             - 'constraint_satisfying' - Include only points that
               satisfy every constraint.
             - 'constraint_violating' - Include only points that
               violate any constraint.
             - 'all' - Include all points in dataset.
             - 'none' - Include no points in dataset.

        Returns:
            df (Pandas dataframe): A 2D dataframe containing post-filter
            data from the MOOP.

    """

    # validate input
    validate_input(valid_db_input, db, "db")
    validate_input(valid_points_input, points, "points")

    # select database
    if db == 'pf':
        database = pd.DataFrame(moop.getPF())
    elif db == 'obj':
        database = pd.DataFrame(moop.getObjectiveData())

    # choose points from database
    if moop.getConstraintType() is None:
        return database
    if points == 'all':
        return database
    if points == 'none':
        return database[0:0]

    constraints = moop.getConstraintType().names
    df = database.copy(deep=True)
    for constraint in constraints:
        if points == 'constraint_satisfying':
            indices = df[df[constraint] > 0].index
        elif points == 'constraint_violating':
            indices = df[df[constraint] <= 0].index
        df.drop(indices, inplace=True)
        df.reset_index(inplace=True)
    return df


def set_hover_info(database, i):
    """ Customize information in hover label for trace i.

        Args:
            database (Pandas dataframe): A 2D dataframe containing the
                traces to be graphed.

            i (int): An index indicating the row where the trace
                we're labeling is located.

        Returns:
            hover_info (string): An HTML-format string to display when
            users hover over trace i.

    """
    # since plotly is JavaScript-based it uses HTML string formatting
    return "<br>".join([f"{key}: {database[key][i]}" for key in database.columns])


def validate_input(
        valid_inputs: set,
        actual_input,
        parameter_name: str = "input",
):
    if actual_input not in valid_inputs:
        raise ValueError(
            f"Unsupported parameter value for {parameter_name}: {actual_input}. "
            f"{parameter_name} must be one of {valid_inputs}"
        )


def check_inputs(
        db,
        output,
        points,
        height,
        width,
        font,
        fontsize,
        background_color,
        screenshot,
        image_export_format,
        data_export_format,
        dev_mode,
        pop_up,
        port,
):
    """ Check keyword inputs to user-facing functions for validity

        Args:
            db: The item passed to the 'db' keyword in a user-facing function.
                If db cannot be cast to a string valued 'pf' or 'obj',
                a ValueError is raised.

            output: The item passed to the 'output' keyword in a
                user-facing function.
                If output cannot be cast to a string corresponding to one of
                the supported output filetypes, a ValueError is raised.

            points: The item passed to the 'points' keyword in a
                user-facing function.
                If points cannot be cast to a string corresponding to one of
                the supported constraint filters, a ValueError is raised.

            height: The item passed to the 'height' keyword in a user-facing
                function.
                If height is not the default string 'auto' or cannot be cast
                to an int
                of value greater than one, a ValueError is raised.

            width: The item passed to the 'width' keyword in a user-facing
                function.
                If width is not the default string 'auto' or cannot be cast
                to an int
                of value greater than one, a ValueError is raised.

            font: The item passed to the 'font' keyword in a user-facing
                function.
                If font cannot be cast to a string, a ValueError is raised.

            fontsize: The item passed to the 'fontsize' keyword in a
                user-facing function.
                If fontsize is not the default value 'auto' or cannot be cast
                to an int
                of value between 1 and 100 inclusive, a ValueError is raised.

            background_color: The item passed to the 'background_color'
                keyword in a user-facing function.
                If background_color cannot be cast to a string, a ValueError
                is raised.

            screenshot: The item passed to the 'screenshot' keyword in a
                user-facing function.
                If screenshot cannot be cast to a string corresponding to one
                of the supported
                screenshot filetypes, a ValueError is raised.

            image_export_format: The item passed to the 'image_export_format'
                keyword in a user-facing function.
                If image_export_format cannot be cast to a string
                corresponding to one of the supported
                image_export_format filetypes, a ValueError is raised.

            data_export_format: The item passed to the 'data_export_format'
                keyword in a user-facing function.
                If data_export_format cannot be cast to a string corresponding
                to one of the supported
                data_export_format filetypes, a ValueError is raised.

            data_export_format: The item passed to the 'data_export_format'
                keyword in a user-facing function.
                If data_export_format cannot be cast to a string corresponding
                to one of the supported
                data_export_format filetypes, a ValueError is raised.

            dev_mode: The item passed to the 'dev_mode' keyword in a
                user-facing function.
                If dev_mode cannot be cast to one of the boolean values True
                and False, a ValueError is raised.

            pop_up: The item passed to the 'pop_up' keyword in a user-facing
                function.
                If pop_up cannot be cast to one of the boolean values True and
                False, a ValueError is raised.

            port: The item passed to the 'port' keyword in a user-facing
                function.
                If port cannot be cast to a string beginning with 'http', a
                ValueError is raised.

        Raises:
            A ValueError if any of the values passed by a user to a keyword in
            a user-facing function are judged invalid.

    TODO update this documentation

    """

    validate_input(valid_db_input, db, "db")

    validate_input(valid_output_input, output, "output")

    validate_input(valid_points_input, points, "points")

    assert (height == 'auto') or (isinstance(height, int) and height >= 1)

    assert (width == 'auto') or (isinstance(width, int) and width >= 1)

    assert isinstance(font, str)

    assert (fontsize == 'auto') or (isinstance(fontsize, int) and 1 <= int(fontsize) <= 100)

    assert isinstance(background_color, str)

    validate_input(valid_screenshot_input, screenshot, "screenshot")

    validate_input(valid_image_export_format_input, image_export_format, "image_export_format")

    validate_input(valid_data_export_format_input, data_export_format, "data_export_format")

    assert isinstance(dev_mode, bool)

    assert isinstance(pop_up, bool)

    assert port[0:4] == 'http'
