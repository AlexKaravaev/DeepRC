from plotly.subplots import make_subplots

import json
from plotly import graph_objs as go
import plotly.express as px
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot results.')

    parser.add_argument('--log-file', type=str, 
                    help='Input log')
    
    args = parser.parse_args()
    with open(args.log_file, 'r') as f:
        json_log = json.load(f)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Path for car", 
                        "Histogram for track error"))
    fig.append_trace(go.Scatter(
         x = [lat_lon_h[0] for lat_lon_h in json_log['positions']],
         y = [lat_lon_h[2] for lat_lon_h in json_log['positions']],
         mode='markers'), row=1, col=1)
    

    fig.append_trace(go.Histogram(
        x = json_log['ctes'],
        name = 'Mean track error'
    ), row=2, col=1) 

    fig.show()
