import os, click
import pandas as pd, numpy as np
import plotly.graph_objs as go
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event

from functools import reduce
from glob import glob
from statistics import mode

DISTINCT_COLORS = [(140,14,14), (255,64,64), (191,96,96), (255,77,0), (229,141,103), (178,107,0), (178,161,0), (180,217,33), (113,128,57), (87,242,48), (18,179,82), (77,255,201), (14,140,140), (73,242,242), (8,109,153), (73,191,242), (8,71,166), (89,156,255), (0,25,255), (33,0,166), (143,115,255), (217,22,197), (166,41,116), (242,61,115)]

def MovingAverageSmoothing(data, window):
    kernel = np.ones(window)
    data = np.asarray(data)
    nfactor = np.ones(len(data))
    smooth_data = np.convolve(data, kernel, 'same') / np.convolve(nfactor, kernel, 'same')
    return smooth_data

def ContinuousErrorBands(figbox, color, xdata, ydata_mean, ydata_stdv, yname):
    # MEAN LINE
    figbox.add_trace(
        go.Scatter(
            name=yname,
            x=xdata,
            y=ydata_mean,
            mode='lines',
            line=dict(color=f'rgb({color[0]},{color[1]},{color[2]})'),
        )
    )
    # UPPER BOUND LINE
    figbox.add_trace(
        go.Scatter(
            name='Upperbound',
            x=xdata,
            y=ydata_mean+ydata_stdv,
            mode='lines',
            marker=dict(color=f'rgb({color[0]},{color[1]},{color[2]})'),
            line=dict(width=0),
            showlegend=False
        )
    )
    # LOWER BOUND LINE
    figbox.add_trace(
        go.Scatter(
            name='Lowerbound',
            x=xdata,
            y=ydata_mean-ydata_stdv,
            marker=dict(color=f'rgb({color[0]},{color[1]},{color[2]})'),
            line=dict(width=0),
            mode='lines',
            fillcolor=f'rgba({color[0]},{color[1]},{color[2]},0.20)',
            fill='tonexty',
            showlegend=False
        )
    )
    return figbox

def GetTensorboardData(algpath):
    seedpaths = glob(f"{algpath}/*/")
    seedbook = []
    for seedpath in seedpaths:
        algdata = EventAccumulator(seedpath).Reload().scalars 
        algdata = [pd.DataFrame(algdata.Items(key))[['step', 'value']].rename(columns={'value':key}) for key in algdata.Keys()]
        smoothalgdata = []
        for algseed in algdata:
            for key in algseed.columns:
                if key not in ['Environment/num_steps', 'Environment/step']:
                    algseed[key] = MovingAverageSmoothing(algseed[key], window=11)
            smoothalgdata.append(algseed)

        smoothalgdata = reduce(lambda d1,d2: pd.merge(d1,d2,on='step',validate='1:1'), smoothalgdata)
        seedbook.append(smoothalgdata)
    
    avgbook = pd.concat(seedbook).groupby(level=0).mean()
    stdbook = pd.concat(seedbook).groupby(level=0).std()
    stdbook['step'] = avgbook['step']
    avgbook.rename(columns={name:f"{name}_mean" for name in avgbook.columns if name!='step'}, inplace=True)
    stdbook.rename(columns={name:f"{name}_stdv" for name in stdbook.columns if name!='step'}, inplace=True)
    seedbook = pd.merge(avgbook,stdbook,on='step',validate='1:1')
    return seedbook

def GetEnvironmentBenchmark(envpath, benchmarkname, figpath, saveformat='png'):
    algpaths = glob(f"{envpath}/*")
    figbox = go.Figure()
    for idno, algpath in enumerate(algpaths):
        algname = os.path.basename(os.path.normpath(algpath))
        algdata = GetTensorboardData(algpath)
        algdata.columns = [name.replace('Environment/','') for name in algdata.columns]

        x = algdata['step'].values
        y_mean = algdata[f'{benchmarkname}_mean'].values
        y_stdv = algdata[f'{benchmarkname}_stdv'].values
        colorid = DISTINCT_COLORS[idno] 
        figbox = ContinuousErrorBands(figbox, colorid, x, y_mean, y_stdv, algname)
        
    envname = os.path.basename(os.path.normpath(envpath))
    figbox.update_layout(
        yaxis_title='Average Reward',
        xaxis_title='Iterations',
        title=f'Benchmark on {envname}',
        hovermode="x"
    )   
    
    os.makedirs(figpath, exist_ok=True)
    if saveformat=='png':
        figbox.write_image(f"{figpath}/{envname}.png")
    else:
        with open(f"{figpath}/{envname}.html", 'w') as f:
            f.write(figbox.to_html(full_html=False, include_plotlyjs='cdn'))
    

@click.command()
@click.option("--env_id",       type=str,   default="BipedalWalker-v3",       help="Environment Id")
@click.option("--benchname",    type=str,   default="average_reward",         help="Benchmark value Id")
@click.option("--logpath",      type=str,   default="./results/logs",         help="logs directory")
@click.option("--figpath",      type=str,   default="./results/benchmarks",   help="figures directory")
@click.option("--format",       type=str,   default="png",                    help="saving format")
def main(env_id, benchname, logpath, figpath, format):
    GetEnvironmentBenchmark(f'{logpath}/{env_id}', benchname, figpath, format)

if __name__ == '__main__':
    main()
