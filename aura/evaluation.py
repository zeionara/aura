from logging import getLogger

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from .Subset import Subset


logger = getLogger(__name__)


def compute_ap(ranked_list, relevant_set):
    """Compute Average Precision (AP) for binary relevance"""
    if not relevant_set:
        return 0.0

    relevant_item_positions = []

    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(ranked_list):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / (i + 1)
            relevant_item_positions.append(i)

    return precision_sum / len(relevant_set)


def compute_mir(ranked_list, relevant_set):
    """Compute Mean Inverted Rank (MIR) for binary relevance"""
    if not relevant_set:
        return 0.0

    inverted_ranks = []

    for i, doc_id in enumerate(ranked_list):
        if doc_id in relevant_set:
            inverted_ranks.append(len(ranked_list) - i - 1)


    if len(inverted_ranks) < 1:
        return 0.0

    if len(inverted_ranks) < 2:
        return inverted_ranks[0]

    return sum(inverted_ranks) / len(inverted_ranks)


def compute_mrr(ranked_list, relevant_set):
    """Compute Mean Reciprocal Rank (MRR) for binary relevance"""
    if not relevant_set:
        return 0.0

    for i, doc_id in enumerate(ranked_list):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(ranked_list, relevance_scores, k=None):
    """Compute Normalized Discounted Cumulative Gain (NDCG) with graded relevance"""
    if k is None:
        k = len(ranked_list)

    # Handle case with no relevant documents
    if not relevance_scores or sum(relevance_scores.values()) == 0:
        return 0.0

    actual = [relevance_scores.get(doc_id, 0) for doc_id in ranked_list[:k]]
    ideal = sorted(actual, reverse=True)
    return ndcg_score([ideal], [actual], k=k)


def evaluate(elements):
    test_paragraph_ids = set()
    paragraphs = []
    tables = []

    for elem in elements:
        if elem['type'] == 'paragraph':
            paragraphs.append(elem)

            if (subset := elem['subset']) is not None and Subset(subset) == Subset.TEST:
                test_paragraph_ids.add(elem['id'])
        elif elem['type'] == 'table':
            tables.append(elem)

    n_tables = len(tables)

    # logger.info('N tables: %d', n_tables)

    para_embeddings = {}
    for para in paragraphs:
        pid = para['id']
        para_embeddings[pid] = para['embeddings']['flat']

    # Determine all available models from paragraphs

    all_models = set()
    for para in paragraphs:
        if 'embeddings' in para and 'flat' in para['embeddings']:
            all_models.update(para['embeddings']['flat'].keys())

    # Initialize results storage
    # Structure: results[model][emb_type][dist_metric][eval_metric] = list of scores

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Process each table
    for table in tables:
        context = {paragraph_id: score for paragraph_id, score in table.get('context', {}).items() if paragraph_id in test_paragraph_ids}

        # Skip tables without context
        if not context:
            continue

        # Create relevance scores (first = highest relevance)

        relevance_scores = {}
        binary_relevance = set()
        max_relevance = len(context)
        for i, pid in enumerate(context):
            relevance_scores[pid] = max_relevance - i
            binary_relevance.add(pid)

        # Process each embedding type in the table
        for emb_type in ['flat', 'structured']:
            if emb_type not in table['embeddings']:
                continue

            # Process each model that exists in both table and paragraphs
            for model in all_models:
                # Skip if model not in table embeddings
                if model not in table['embeddings'][emb_type]:
                    continue

                # Get table embedding
                try:
                    table_emb = np.array(table['embeddings'][emb_type][model]).reshape(1, -1)
                except (KeyError, TypeError):
                    continue

                # Prepare paragraph embeddings for this model
                para_ids = []
                para_embs = []
                for pid, emb_dict in para_embeddings.items():
                    if model in emb_dict:
                        try:
                            emb = np.array(emb_dict[model])
                            para_ids.append(pid)
                            para_embs.append(emb)
                        except (KeyError, TypeError):
                            continue

                # Skip if no paragraphs found
                if not para_ids:
                    continue

                para_embs = np.array(para_embs)

                # Compute similarity metrics
                similarity_metrics = {}

                # Cosine similarity
                try:
                    cosine_sim = cosine_similarity(table_emb, para_embs).flatten()
                    similarity_metrics['cosine'] = cosine_sim
                except Exception:
                    similarity_metrics['cosine'] = None

                # Euclidean distance (convert to similarity)
                try:
                    euclidean_dist = euclidean_distances(table_emb, para_embs).flatten()
                    similarity_metrics['euclidean'] = -euclidean_dist  # Higher is better
                except Exception:
                    similarity_metrics['euclidean'] = None

                # Dot product
                try:
                    dot_product = np.dot(para_embs, table_emb.T).flatten()
                    similarity_metrics['dot'] = dot_product
                except Exception:
                    similarity_metrics['dot'] = None

                # Compute evaluation metrics for each similarity metric
                for dist_metric, scores in similarity_metrics.items():
                    if scores is None:
                        continue

                    # Create ranked list of paragraph IDs
                    ranked_indices = np.argsort(scores)[::-1]  # Descending order
                    ranked_list = [para_ids[i] for i in ranked_indices]

                    # Compute metrics
                    try:
                        ndcg = compute_ndcg(ranked_list, relevance_scores)
                    except Exception:
                        ndcg = np.nan

                    try:
                        ap = compute_ap(ranked_list, binary_relevance)
                    except Exception:
                        ap = np.nan

                    try:
                        mrr = compute_mrr(ranked_list, binary_relevance)
                    except Exception:
                        mrr = np.nan

                    try:
                        mir = compute_mir(ranked_list, binary_relevance)
                    except Exception:
                        mir = np.nan

                    # Store results
                    if not np.isnan(ndcg):
                        results[model][emb_type][dist_metric]['NDCG'].append(ndcg)
                    if not np.isnan(ap):
                        results[model][emb_type][dist_metric]['MAP'].append(ap)
                    if not np.isnan(mrr):
                        results[model][emb_type][dist_metric]['MRR'].append(mrr)
                    if not np.isnan(mir):
                        results[model][emb_type][dist_metric]['MIR'].append(mir)

    # Prepare final dataframe
    rows = []
    columns = set()

    # Collect all unique evaluation metrics and distance metrics
    eval_metrics = ['NDCG', 'MAP', 'MRR', 'MIR']
    dist_metrics = ['cosine', 'euclidean', 'dot']

    for model in all_models:
        for emb_type in ['flat', 'structured']:
            # Skip if no results for this combination
            if model not in results or emb_type not in results[model]:
                continue

            row_data = {'Model': model, 'Embedding Type': emb_type}

            # Initialize with NaN for all possible combinations
            for eval_metric in eval_metrics:
                for dist_metric in dist_metrics:
                    col_name = (eval_metric, dist_metric)
                    row_data[col_name] = np.nan
                    columns.add(col_name)

            # Fill in available results
            for dist_metric, metrics_dict in results[model][emb_type].items():
                for eval_metric, values in metrics_dict.items():
                    if values:
                        avg_value = np.mean(values)
                        col_name = (eval_metric, dist_metric)
                        row_data[col_name] = avg_value

            rows.append(row_data)

    # Create multi-index dataframe if we have results
    if not rows:
        if n_tables > 0:
            raise ValueError()

        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.set_index(['Model', 'Embedding Type'], inplace=True)

    columns = sorted(columns, key=lambda x: (x[0], x[1]))
    df = df[columns]

    df.columns = pd.MultiIndex.from_tuples(
        columns,
        names=['Evaluation Metric', 'Distance Metric']
    )

    logger.info(df)

    return df


def average(dataframes):
    if not dataframes:
        return pd.DataFrame()

    # Validate that all DataFrames have the same structure
    reference_idx = dataframes[0].index
    reference_cols = dataframes[0].columns

    while len(dataframes) > 0:
        for i, df in enumerate(dataframes):
            if not df.index.equals(reference_idx) or not df.columns.equals(reference_cols):
                # raise ValueError(f"DataFrame at position {i} has different structure than the first DataFrame")
                logger.warning("DataFrame at position %d has different structure than the first DataFrame. Trying to skip it...", i)
                dataframes = dataframes[0:i] + dataframes[i + 1:]
                continue
        break

    non_empty_dataframes = []

    for df in dataframes:
        if not df.empty:
            non_empty_dataframes.append(df)

    print('N total dfs:', len(dataframes))
    print('N non-empty dfs:', len(non_empty_dataframes))

    # Create 3D array of all DataFrame values
    all_values = np.array([df.values for df in non_empty_dataframes])

    # Compute mean along the first axis (across DataFrames)
    averaged_values = np.nanmean(all_values, axis = 0)

    # Create new DataFrame with same index and columns
    return pd.DataFrame(averaged_values, index = reference_idx, columns = reference_cols)
