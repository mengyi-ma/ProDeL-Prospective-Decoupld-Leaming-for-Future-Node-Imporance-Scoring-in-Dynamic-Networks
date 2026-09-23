import pandas as pd
import numpy as np
import pickle
import os

def process_network_data(input_file, output_path):
    """
    parameter:
        input_file: inputfilepath
        output_path: outputfilepath
    """
    try:
        df = pd.read_csv(input_file, header=None, sep=r'\s+', engine='python')

        if df.shape[1] >= 3:
            # df = df.iloc[:, [1, 2, 0]].copy()
            df.columns = ['source', 'target', 'time_stamps/seconds']
            df['source'] = df['source'].astype(str)
            df['target'] = df['target'].astype(str)
        else:
            raise ValueError(f"error：dataneedtext3text，text{df.shape[1]}text")

        df = df[df['time_stamps/seconds'] != 0]
        df = df[df['source'] != df['target']].copy()

        min_timestamp = df['time_stamps/seconds'].min()
        print(f"text: {min_timestamp}")
        max_timestamp = df['time_stamps/seconds'].max()
        print(f"text: {max_timestamp}")
        df['time_stamps/seconds'] = df['time_stamps/seconds'] - min_timestamp

        df['time_stamps/seconds'] = pd.to_numeric(df['time_stamps/seconds'], errors='raise')

        min_timestamp = df['time_stamps/seconds'].min()
        bucket_size = 432000
        df['time_stamps/seconds'] = ((df['time_stamps/seconds'] - min_timestamp) // bucket_size).astype(np.int64)

        ts_min_after = int(df['time_stamps/seconds'].min())
        if ts_min_after != 0:
            raise RuntimeError(f"text {ts_min_after}，text0text。")

        time_window_size = 1
        df['time_window'] = (df['time_stamps/seconds'] / time_window_size).astype(int)

        original_windows = df['time_window'].nunique()
        print(f"textwindownumber: {original_windows}")

        window_mapping = {old: new for new, old in enumerate(sorted(df['time_window'].unique()))}
        df['time_window'] = df['time_window'].map(window_mapping)

        df = df[['source', 'target', 'time_stamps/seconds', 'time_window']].copy()

        df = df.sort_values(by=['time_window', 'time_stamps/seconds'])

        # df = df[df['time_window'] >= 2].copy()
        # df['time_window'] = df['time_window'] - 2

        window_attributes = {}
        split_to_windows = {}

        total_windows = df['time_window'].nunique()
        print(f"textwindownumber: {total_windows}")

        train_start, train_end = 0, 30
        val_start, val_end = 30, 43
        test_start = 43

        def generate_sliding_splits(start_window, end_window):
            """textwindowsplit"""
            splits = []
            current_window = start_window
            while current_window + 4 <= end_window:
                split_windows = list(range(current_window, current_window + 4))
                splits.append(split_windows)
                current_window += 1
            return splits

        train_splits = generate_sliding_splits(train_start, train_end)
        val_splits = generate_sliding_splits(val_start, val_end)
        test_splits = generate_sliding_splits(test_start, total_windows)

        splits = {
            "train": train_splits,
            "val": val_splits,
            "test": test_splits
        }

        print(f"Trainsplitnumber: {len(splits['train'])}")
        print(f"Valsplitnumber: {len(splits['val'])}")
        print(f"Testsplitnumber: {len(splits['test'])}")

        split_index_counter = 0
        for split_type, split_list in splits.items():
            for windows in split_list:
                for window in windows:
                    if window in df['time_window'].unique():
                        window_attributes[window] = (split_type, split_index_counter)

                if split_index_counter not in split_to_windows:
                    split_to_windows[split_index_counter] = []
                split_to_windows[split_index_counter].extend(windows)

                split_index_counter += 1

        print("\neachsplittext:")
        split_info = {}
        for split_idx, windows in split_to_windows.items():
            split_type = None
            for window in windows:
                if window in window_attributes:
                    split_type = window_attributes[window][0]
                    break

            actual_windows = [w for w in windows if w in df['time_window'].unique()]

            split_info[split_idx] = {
                'type': split_type,
                'total_windows': len(actual_windows),
                'window_ids': actual_windows
            }

            print(f"split {split_idx}: type={split_type}, windownumber={len(actual_windows)}, windowID={actual_windows}")

        print("\ntexttypetext:")
        type_summary = {}
        for split_idx, info in split_info.items():
            split_type = info['type']
            if split_type not in type_summary:
                type_summary[split_type] = {'count': 0, 'total_windows': 0, 'split_ids': []}
            type_summary[split_type]['count'] += 1
            type_summary[split_type]['total_windows'] += info['total_windows']
            type_summary[split_type]['split_ids'].append(split_idx)

        for split_type, summary in type_summary.items():
            print(f"{split_type}type: splitnumber={summary['count']}, textwindowtext={summary['total_windows']}, splitID={summary['split_ids']}")

        base_name = os.path.splitext(os.path.basename(output_path))[0]
        pkl_filename = f"{base_name}_window_attributes.pkl"
        attributes_path = os.path.join(pkl_output_dir, pkl_filename)
        with open(attributes_path, 'wb') as f:
            pickle.dump({
                'window_attributes': window_attributes,
                'split_to_windows': split_to_windows,
                'split_info': split_info
            }, f)
        print(f"windowtextsavetext: {attributes_path}")

        unique_nodes = set(df['source'].unique()) | set(df['target'].unique())
        total_nodes = len(unique_nodes)
        print(f"nodetotal number: {total_nodes}")

        total_edges = len(df)
        print(f"texttotal number: {total_edges}")

        num_windows = df['time_window'].nunique()
        print(f"mergetextwindownumber: {num_windows}")

        print("___________________________")

        window_stats = []
        window_nodes_sets = []
        for window in range(num_windows):
            window_df = df[df['time_window'] == window]

            source_nodes = set(window_df['source'].unique())
            target_nodes = set(window_df['target'].unique())
            connected_nodes = source_nodes | target_nodes

            window_nodes_sets.append(connected_nodes)
            window_edges = len(window_df)
            window_stats.append({
                'window': window,
                'nodes': len(connected_nodes),
                'edges': window_edges
            })

        print("\neachwindowtext:")
        print("window\tnumber of nodes\ttext")
        for stat in window_stats:
            print(f"{stat['window']}\t{stat['nodes']}\t{stat['edges']}")

        lines = []
        header = f"{'ts_i':>12}\t{'win_i':>6}\t{'ts_j':>12}\t{'win_j':>6}\t{'nodes_i':>8}\t{'nodes_j':>8}\t{'common':>8}\n"
        lines.append(header)

        for split_idx, split_info in split_info.items():
            split_type = split_info['type']
            split_windows = split_info['window_ids']

            group_windows = split_windows[:3]

            ref_window = split_windows[3]

            nodes_i = set().union(*(window_nodes_sets[w] for w in group_windows))
            nodes_j = window_nodes_sets[ref_window]
            common = len(nodes_i & nodes_j)

            ts_i_vals = df.loc[df['time_window'].isin(group_windows), 'time_stamps/seconds'].unique()
            ts_j_vals = df.loc[df['time_window'] == ref_window, 'time_stamps/seconds'].unique()
            ts_i = int(np.min(ts_i_vals)) if len(ts_i_vals) > 0 else -1
            ts_j = int(ts_j_vals[0]) if len(ts_j_vals) > 0 else -1

            win_i = f"{group_windows[0]}-{group_windows[2]}"
            win_j = ref_window

            lines.append(
                f"{ts_i:>12}\t{win_i:>6}\t{ts_j:>12}\t{win_j:>6}\t{len(nodes_i):>8}\t{len(nodes_j):>8}\t{common:>8}\n")

        txt_path = os.path.join(pkl_output_dir, "Node_Status_in_Window.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"textwindownodetextsavetext TXT：{txt_path}")

        df.to_csv(output_path, index=False)
        print(f"datatextcompleted，textsavetext: {output_path}")

        print("\ntextdatatext:")
        print(df.head())

    except Exception as e:
        print(f"textfiletext: {e}")
        print("textcheckinputfiletext")

if __name__ == "__main__":
    input_file = "./data/raw/Email.txt"
    output_path = "./data/raw/Email.csv"
    pkl_output_dir = "./data/processed_data"
    os.makedirs(pkl_output_dir, exist_ok=True)

    process_network_data(input_file, output_path)
