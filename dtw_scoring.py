import numpy as np
import glob
import torch
import os
import math, argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dtw import dtw
from lxml import etree
from scipy.spatial import distance
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import PosixPath


def match(query, doc, query_name, doc_name, dist_fn, minmax_norm, dtwrc):
    """Match between a query and a doc."""
    dist = dist_fn(query, doc)

    if minmax_norm:
        dist_min = dist.min(1)[:, np.newaxis]
        dist_max = dist.max(1)[:, np.newaxis]
        # dist_min = np.nan_to_num(dist_min, nan=0.0, posinf=0.0, neginf=0.0)
        # dist_max = np.nan_to_num(dist_max, nan=1.0, posinf=1.0, neginf=1.0)
        dist = (dist - dist_min) / np.clip(dist_max - dist_min, 1e-9, np.inf)

    dtw_result = dtw(x=dist, **dtwrc)
    cost = dtw_result.normalizedDistance
    return query_name, doc_name, -1 * cost


def cosine_exp(query, doc):
    dist = distance.cdist(query, doc, "cosine")
    dist = np.exp(dist) - 1
    return dist


def cosine_neg_log(query, doc):
    dist = distance.cdist(query, doc, "cosine")
    dist = -1 * np.log(1 - dist)
    return dist



def main(reference_dir, reference_output, queries_dir, queries_output, expdir):

    doc_entries = glob.glob(args.reference_dir + "/*.wav")
    eval_entries = glob.glob(args.queries_dir + "/*.wav")  

    docs = []
    for item in doc_entries:
        item = item.split("/")[-1]
        fn = os.path.join(reference_output, item.replace(".wav", "_emissions.pt"))
        docs.append(torch.load(fn).squeeze())
    
    doc_names = []
    for item in doc_entries:
        doc_names.append(item.split("/")[-1].replace(".wav", ""))

    queries = []
    for item in eval_entries:
        item = item.split("/")[-1]
        fn = os.path.join(queries_output, item.replace(".wav", "_emissions.pt"))
        queries.append(torch.load(fn).squeeze())
    
    query_names = []
    for item in eval_entries:
        query_names.append(item.split("/")[-1].replace(".wav", ""))
    
    dist_fn = partial(distance.cdist, metric='cosine')
    dtwrc = {'step_pattern': 'asymmetric', 'keep_internals': False, 'distance_only': False, 'open_begin': True, 'open_end': True}


    # Calculate matching scores
    results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=24) as executor:
        futures = []
        for query, query_name in zip(queries, query_names):
            if len(query) < 5:  # Do not consider too short queries
                results[query_name] = [(doc_name, 0) for doc_name in doc_names]
                continue
            for doc, doc_name in zip(docs, doc_names):
                futures.append(
                    executor.submit(
                        match,
                        query,
                        doc,
                        query_name,
                        doc_name,
                        dist_fn,
                        True, #minmax_norm
                        dtwrc,
                    )
                )
        for future in tqdm(
            as_completed(futures), total=len(futures), ncols=0, desc="DTW"
        ):
            query_name, doc_name, score = future.result()
            results[query_name].append((doc_name, score))

    # Normalize scores with regard to each query
    for query_name, doc_scores in results.items():
        names, scores = zip(*doc_scores)
        scores = np.array(scores)
        scores = (scores - scores.mean()) / np.clip(scores.std(), 1e-9, np.inf)
        results[query_name] = list(zip(names, scores))

    # Scores above 2 STDs are seen as detected (top 2.5% as YES)
    score_thresh = 2

    # Build XML tree
    root = etree.Element(
        "stdlist",
        termlist_filename="benchmark.stdlist.xml",
        indexing_time="1.00",
        language="english",
        index_size="1",
        system_id="benchmark",
    )
    for query_name, doc_scores in results.items():
        term_list = etree.SubElement(
            root,
            "detected_termlist",
            termid=query_name.replace(".wav", ""),
            term_search_time="1.0",
            oov_term_count="1",
        )
        for doc_name, score in doc_scores:
            etree.SubElement(
                term_list,
                "term",
                file=doc_name.replace(".wav", ""),
                channel="1",
                tbeg="0.000",
                dur="0.00",
                score=f"{score:.4f}",
                decision="YES" if score > score_thresh else "NO",
            )

    # Output XML
    etree.ElementTree(root).write(
        str(expdir / "benchmark.stdlist.xml"),
        encoding="UTF-8",
        pretty_print=True,
    )



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference-dir")
    parser.add_argument("--reference-output")
    parser.add_argument("--queries-dir")
    parser.add_argument("--queries-output")
    parser.add_argument("--exp-dir")

    args = parser.parse_args()

    expdir = PosixPath(args.exp_dir)
    os.makedirs(expdir)
    main(args.reference_dir, args.reference_output, args.queries_dir, args.queries_output, expdir)
