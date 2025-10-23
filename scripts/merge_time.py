import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_total_runtime_from_events(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    
    # Get all scalar tags (modify to select your specific metric)
    scalar_tags = event_acc.Tags()['scalars']
    
    if not scalar_tags:
        return None
        
    # We'll use the first scalar tag to get timestamps (modify as needed)
    sample_tag = 'train/return'
    events = event_acc.Scalars(sample_tag)
    
    if not events:
        return None
        
    # Calculate runtime as the difference between first and last event timestamps
    start_time = events[0].wall_time
    end_time = events[-1].wall_time
    total_runtime = end_time - start_time
    
    return total_runtime

if __name__ == '__main__':
    results_tot = {}
    for method in ['rl_leiden']:
        folder = os.path.join('results/data_large2', method, 'CNV', 'leiden')
        datafolder = os.path.join('data', 'data_large2')
        results = {}
        for noise_level in [0.25, 0.3, 0.35,0.4]:
            results[noise_level] = []
            for dir in os.listdir(folder):
                if dir.endswith(str(noise_level)):
                    tb_file = [f for f in os.listdir(os.path.join(folder, dir)) if f.startswith('events')][0]
                    run_time = get_total_runtime_from_events(os.path.join(folder, dir, tb_file))
                    results[noise_level].append(run_time)
            results[noise_level] = np.array(results[noise_level])

        times = {k: [results[k].mean(), results[k].std()] for k in results.keys()}
        all_results = np.concatenate([results[k] for k in results.keys()])
        print(all_results.mean(), all_results.std())
        
