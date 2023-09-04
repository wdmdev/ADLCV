import pandas as pd
import plotly.graph_objs as go

def plot_all():
    # Read the data from the .csv file
    df = pd.read_csv('batch_results.csv')

    # Melt the data to get it in a format suitable for plotting
    df_melted = df.melt(id_vars=['epoch'], var_name='model_config', value_name='validation_accuracy')

    # Define a list of marker symbols
    marker_symbols = ['circle', 'square', 'diamond', 'cross']

    # Define a list of line styles
    line_styles = ['solid', 'dash', 'dot', 'dashdot']

    # Create a dictionary that maps model configurations to marker symbols and line styles in a cyclic manner
    style_dict = {
        model_config: {
            'marker_symbol': marker_symbols[i % len(marker_symbols)],
            'line_style': line_styles[i % len(line_styles)]
        }
        for i, model_config in enumerate(df_melted['model_config'].unique())
    }

    # Create an empty figure
    fig = go.Figure()

    # Add a scatter trace to the figure for each model configuration
    for model_config, group in df_melted.groupby('model_config'):
        fig.add_trace(
            go.Scatter(
                x=group['epoch'],
                y=group['validation_accuracy'],
                name=model_config,
                mode='lines+markers',
                marker=dict(symbol=style_dict[model_config]['marker_symbol']),
                line=dict(dash=style_dict[model_config]['line_style'])
            )
        )

    #set x and y titles
    fig.update_xaxes(title_text='Epoch')
    fig.update_yaxes(title_text='Validation accuracy')

    #font sizes
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=14))
    fig.update_layout(legend_title_font=dict(size=18), legend_font=dict(size=16))

    # Save the figure to a file
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    fig.write_image('plots/all_validation_accuracy_plot.png')

def plot_seperate():
    # Read the data from the .csv file
    df = pd.read_csv('batch_results.csv')

    # Define the parameter groups
    param_groups = {
        'embed_dim': ['embed_dim_64', 'embed_dim_128', 'embed_dim_256', 'embed_dim_512'],
        'num_heads': ['num_heads_2', 'num_heads_4', 'num_heads_8', 'num_heads_16'],
        'num_layers': ['num_layers_2', 'num_layers_4', 'num_layers_6', 'num_layers_8'],
        'pos_enc': ['pos_enc_fixed', 'pos_enc_learnable'],
        'pool': ['pool_mean', 'pool_max']
    }

    # Define a list of marker symbols
    marker_symbols = ['circle', 'square', 'diamond', 'cross']

    # Define a list of line styles
    line_styles = ['solid', 'dash', 'dot', 'dashdot']

    # Create a plot for each parameter group
    for group_name, params in param_groups.items():
        # Select the columns corresponding to the current parameter group and the epoch column
        df_group = df[['epoch'] + params]

        # Melt the data to get it in a format suitable for plotting
        df_melted = df_group.melt(id_vars=['epoch'], var_name='model_config', value_name='validation_accuracy')

        # Create a dictionary that maps model configurations to marker symbols and line styles in a cyclic manner
        style_dict = {
            model_config: {
                'marker_symbol': marker_symbols[i % len(marker_symbols)],
                'line_style': line_styles[i % len(line_styles)]
            }
            for i, model_config in enumerate(df_melted['model_config'].unique())
        }

        # Create an empty figure
        fig = go.Figure()

        # Add a scatter trace to the figure for each model configuration
        for model_config, group in df_melted.groupby('model_config'):
            fig.add_trace(
                go.Scatter(
                    x=group['epoch'],
                    y=group['validation_accuracy'],
                    name=model_config,
                    mode='lines+markers',
                    marker=dict(symbol=style_dict[model_config]['marker_symbol']),
                    line=dict(dash=style_dict[model_config]['line_style'])
                )
            )

        #set x and y titles
        fig.update_xaxes(title_text='Epoch')
        fig.update_yaxes(title_text='Validation accuracy')

        #font sizes
        fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=14))
        fig.update_layout(legend_title_font=dict(size=18), legend_font=dict(size=16))

        # Save the figure to a file
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
        fig.write_image(f'plots/{group_name}_plot.png')

if __name__ == '__main__':
    plot_all()
    plot_seperate()