from .dashboard import (
    Dash_App
)
from .utilities import (
    export_file,
    set_plot_name,
)
from .graph import (
    generate_scatter,
    generate_parallel,
    generate_radar,
)

"""
For all interactve browser plot functions, there is a known issue
where Plotly images may not export from Safari correctly. If you
encounter this issue, change your default browser to Chrome, Firefox,
or Edge.
"""

""" Display MOOP results as matrix of 2D scatterplots.

Create an interactive plot that displays in the browser.

For ``n`` objectives, generate an ``n x n`` matrix of 2D scatterplots.

Users can hover above an output point to see input information pop up.

Args:
    moop (MOOP): A ParMOO MOOP containing the MOOP results to plot.
    db (String): Choose database to plot
                    'pf' (default) plot Pareto Front
                    'obj' plot objective data
    output (String):
                    'dash' (default) display plot in dash app
                    'no_dash' display plot in browser without dash
                    'html' export plot as html to working directory
                    'pdf' export plot as pdf to working directory
                    'svg' export plot as svg to working directory
                    'webp' export plot as webp to working directory
                    'jpeg' export plot as jpeg to working directory
                    'png' export plot as png to working directory
    browser (boolean): Display interactive plot in browser window.
                True: (default) display interactive plot in browser window
                False: don't display interactive plot in browser window
                It is recommended that this setting be left on True
                The 'browser' and 'export' keywords will not
                interfere with each other. If you choose to export an image
                of the plot by using the 'export' keyword, and leave
                'browser' to True, you will BOTH export an image file to
                the current working directory AND open an interactive
                figure in the browser

Returns:
    None

"""


def scatter(
    moop,
    db='pf',
    output='dash',
    axes='objectives',
    specificaxes='auto',
    height='auto',
    width='auto',
    font='auto',
    fontsize='auto',
    background_color='auto',
    margins='auto',
    dummy1='auto',
    dummy2='auto',
    dummy3='auto',
    dummy4='auto',
    dummy5='auto',
    dummy6='auto',
    verbose=True,
    hot_reload=False,
    pop_up=True,
    port='http://127.0.0.1:8050/',
):
    if output == 'dash':
        Dash_App(
            plot_type='scatter',
            moop=moop,
            db=db,
            axes=axes,
            specificaxes=specificaxes,
            height=height,
            width=width,
            font=font,
            fontsize=fontsize,
            background_color=background_color,
            margins=margins,
            dummy1=dummy1,
            dummy2=dummy2,
            dummy3=dummy3,
            dummy4=dummy4,
            dummy5=dummy5,
            dummy6=dummy6,
            verbose=verbose,
            hot_reload=hot_reload,
            pop_up=pop_up,
            port=port,
        )
    else:
        plot_name = set_plot_name(db=db)
        export_file(
            fig=generate_scatter(
                moop,
                db=db,
                axes=axes,
                specificaxes=specificaxes,
                height=height,
                width=width,
                font=font,
                fontsize=fontsize,
                background_color=background_color,
                margins=margins,
                dummy1=dummy1,
                dummy2=dummy2,
                dummy3=dummy3,
                dummy4=dummy4,
                dummy5=dummy5,
                dummy6=dummy6,
                verbose=verbose,
            ),
            plot_name=plot_name,
            file_type=output,
        )


""" Display MOOP results as parallel coordinates plot.

    Create an interactive plot that displays in the display.

    Users can select item(s) in a parallel coordinates plot
    by selecting an axis section which item(s) pass through.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.
        db (String): Choose database to plot
                     'pf' (default) plot Pareto Front
                     'obj' plot objective data
        export (String): Export plot to working directory.
                     'none' (default) don't export image file
                     'html' export plot as html
                     'pdf' export plot as pdf
                     'svg' export plot as svg
                     'webp' export plot as webp
                     'jpeg' export plot as jpeg
                     'png' export plot as png
        browser (boolean): Display interactive plot in browser window.
                    True: (default) display interactive plot in browser window
                    False: don't display interactive plot in browser window
                    It is recommended that this setting be left on True
                    The 'browser' and 'export' keywords will not
                    interfere with each other. If you choose to export an image
                    of the plot by using the 'export' keyword, and leave
                    'browser' to True, you will BOTH export an image file to
                    the current working directory AND open an interactive
                    figure in the browser.
        objectives_only (boolean): display all data, or objectives only
                    True: (default) plot objectives as axes only
                    False: plot inputs as axes as well

    Returns:
        None

    """


def parallel_coordinates(
    moop,
    db='pf',
    output='dash',
    axes='objectives',
    specificaxes='auto',
    height='auto',
    width='auto',
    font='auto',
    fontsize='auto',
    background_color='auto',
    margins='auto',
    dummy1='auto',
    dummy2='auto',
    dummy3='auto',
    dummy4='auto',
    dummy5='auto',
    dummy6='auto',
    verbose=True,
    hot_reload=False,
    pop_up=True,
    port='http://127.0.0.1:8050/',
):

    if output == 'dash':
        Dash_App(
            plot_type='parallel',
            moop=moop,
            db=db,
            axes=axes,
            specificaxes=specificaxes,
            height=height,
            width=width,
            font=font,
            fontsize=fontsize,
            background_color=background_color,
            margins=margins,
            dummy1=dummy1,
            dummy2=dummy2,
            dummy3=dummy3,
            dummy4=dummy4,
            dummy5=dummy5,
            dummy6=dummy6,
            verbose=verbose,
            hot_reload=hot_reload,
            pop_up=pop_up,
            port=port,
        )
    else:
        plot_name = set_plot_name(db=db)
        export_file(
            fig=generate_parallel(
                moop=moop,
                db=db,
                axes=axes,
                specificaxes=specificaxes,
                height=height,
                width=width,
                font=font,
                fontsize=fontsize,
                background_color=background_color,
                margins=margins,
                dummy1=dummy1,
                dummy2=dummy2,
                dummy3=dummy3,
                dummy4=dummy4,
                dummy5=dummy5,
                dummy6=dummy6,
                verbose=verbose,
            ),
            plot_name=plot_name,
            file_type=output
        )


""" Display MOOP results as radar plot.

    Create an interactive plot that displays in the browser.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.
        db (String): Choose database to plot
                     'pf' (default) plot Pareto Front
                     'obj' plot objective data
        export (String): Export plot to working directory.
                     'none' (default) don't export image file
                     'html' export plot as html
                     'pdf' export plot as pdf
                     'svg' export plot as svg
                     'webp' export plot as webp
                     'jpeg' export plot as jpeg
                     'png' export plot as png
        browser (boolean): Display interactive plot in browser window.
                    True: (default) display interactive plot in browser window
                    False: don't display interactive plot in browser window
                    It is recommended that this setting be left on True
                    The 'browser' and 'export' keywords will not
                    interfere with each other. If you choose to export an image
                    of the plot by using the 'export' keyword, and leave
                    'browser' to True, you will BOTH export an image file to
                    the current working directory AND open an interactive
                    figure in the browser.

    Returns:
        None

    """


def radar(
    moop,
    db='pf',
    output='dash',
    axes='objectives',
    specificaxes='auto',
    height='auto',
    width='auto',
    font='auto',
    fontsize='auto',
    background_color='auto',
    margins='auto',
    dummy1='auto',
    dummy2='auto',
    dummy3='auto',
    dummy4='auto',
    dummy5='auto',
    dummy6='auto',
    verbose=True,
    hot_reload=False,
    pop_up=True,
    port='http://127.0.0.1:8050/',
):

    if output == 'dash':
        Dash_App(
            plot_type='radar',
            moop=moop,
            db=db,
            axes=axes,
            specificaxes=specificaxes,
            height=height,
            width=width,
            font=font,
            fontsize=fontsize,
            background_color=background_color,
            margins=margins,
            dummy1=dummy1,
            dummy2=dummy2,
            dummy3=dummy3,
            dummy4=dummy4,
            dummy5=dummy5,
            dummy6=dummy6,
            verbose=verbose,
            hot_reload=hot_reload,
            pop_up=pop_up,
            port=port,
        )
    else:
        plot_name = set_plot_name(db=db)
        export_file(
            fig=generate_radar(
                moop,
                db=db,
                axes=axes,
                specificaxes=specificaxes,
                height=height,
                width=width,
                font=font,
                fontsize=fontsize,
                background_color=background_color,
                margins=margins,
                dummy1=dummy1,
                dummy2=dummy2,
                dummy3=dummy3,
                dummy4=dummy4,
                dummy5=dummy5,
                dummy6=dummy6,
                verbose=verbose,
            ),
            plot_name=plot_name,
            file_type=output
        )
