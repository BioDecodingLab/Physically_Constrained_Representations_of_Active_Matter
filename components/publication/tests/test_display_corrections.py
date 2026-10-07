import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ampub.figures import decoder_task_records
from ampub.videos import reference_frame


def test_decoder_tasks_are_not_overlaid():
    rows=[]
    for kind,predictor,val in [('reconstruction','none',1e-7),('rollout','linear',2e-7),('rollout','mlp',8.0)]:
        rows.append(dict(dimension=512,seed=31415,method='rpc',kind=kind,predictor=predictor,value=val))
    df=pd.DataFrame(rows)
    assert decoder_task_records(df,'reconstruction','none').value.tolist()==[1e-7]
    assert decoder_task_records(df,'rollout','mlp').value.tolist()==[8.0]


def test_reference_movie_coordinate_labels_clear_colorbars():
    fig=reference_frame(np.ones((3,16,16)),10,{'length':[10,10]},[(0,1),(-1,1),(0,1)])
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for i in range(3):
        image_ax,color_ax=fig.axes[2*i:2*i+2]
        box=image_ax.xaxis.label.get_window_extent(renderer)
        ticks=[t.get_window_extent(renderer) for t in color_ax.get_xticklabels() if t.get_visible()]
        assert not box.overlaps(color_ax.get_window_extent(renderer))
        assert all(not box.overlaps(t) for t in ticks)
    plt.close(fig)
