import numpy as np
import torch

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    metrics['R100'] = float(np.sum(ind < 100)) * 100 / len(ind)
    metrics['R500'] = float(np.sum(ind < 500)) * 100 / len(ind)
    metrics['R1000'] = float(np.sum(ind < 1000)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def tensor_text_to_video_metrics(sim_tensor, top_k = [1,5,10,50]):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
    second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1 = 1, dim2 = 2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    #assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
      valid_ranks = torch.tensor(valid_ranks)
    
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]
    return results

def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T

def print_metrics(t_len, v_len, t2v, v2t, t2v_dsl, v2t_dsl):
    print(f'\t Length-T: {t_len}, Length-V:{v_len}')
    # dsl output
    print("------------------------------------------------------------")
    print("DSL Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(t2v_dsl['R1'], t2v_dsl['R5'], t2v_dsl['R10'], t2v_dsl['R50'], t2v_dsl['MR'], t2v_dsl['MeanR']))
    print("DSL Video-to-Text:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(v2t_dsl['R1'], v2t_dsl['R5'], v2t_dsl['R10'], v2t_dsl['R50'], v2t_dsl['MR'], v2t_dsl['MeanR']))

    print("------------------------------------------------------------")
    print("Text-to-Video:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(t2v['R1'], t2v['R5'], t2v['R10'], t2v['R50'], t2v['MR'], t2v['MeanR']))
    print("Video-to-Text:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(v2t['R1'], v2t['R5'], v2t['R10'], v2t['R50'], v2t['MR'], v2t['MeanR']))

def log_metrics(results, f):
    t_len, v_len, t2v, v2t, t2v_dsl, v2t_dsl = results
    f.write("-------------------------------------------------------------------------------------------------------\n")
    f.write(f'Length-T: {t_len}, Length-V:{v_len}\n')
    # dsl output
    f.write("-------------------------------------------------------------------------------------------------------\n")
    f.write("DSL Text-to-Video:")
    f.write('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}\n'.
                format(t2v_dsl['R1'], t2v_dsl['R5'], t2v_dsl['R10'], t2v_dsl['R50'], t2v_dsl['MR'], t2v_dsl['MeanR']))
    f.write("DSL Video-to-Text:")
    f.write('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}\n'.
                format(v2t_dsl['R1'], v2t_dsl['R5'], v2t_dsl['R10'], v2t_dsl['R50'], v2t_dsl['MR'], v2t_dsl['MeanR']))

    f.write("-------------------------------------------------------------------------------------------------------\n")
    f.write("Text-to-Video:")
    f.write('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}\n'.
                format(t2v['R1'], t2v['R5'], t2v['R10'], t2v['R50'], t2v['MR'], t2v['MeanR']))
    f.write("Video-to-Text:")
    f.write('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}\n'.
                format(v2t['R1'], v2t['R5'], v2t['R10'], v2t['R50'], v2t['MR'], v2t['MeanR']))
    f.write("-------------------------------------------------------------------------------------------------------\n")
    
def multi_setence_retrieval(logits, cut_off_points_):
  cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
  max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
  new_logits = []
  for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
    new_logits.append(np.concatenate((logits[s_:e_], np.full((max_length - e_ + s_, logits.shape[1]), -np.inf)), axis=0))
  logits = np.stack(tuple(new_logits), axis=0)
  print("after reshape, sim matrix size: {} x {} x {}".format(logits.shape[0], logits.shape[1], logits.shape[2]))
  tv_metrics = tensor_text_to_video_metrics(logits)
  vt_metrics = compute_metrics(tensor_video_to_text_sim(logits))
  return tv_metrics, vt_metrics