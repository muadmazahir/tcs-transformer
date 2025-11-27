import torch
from scipy import sparse
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class PASTruthCollator(object):
    def __init__(self, pred2ix):
        self.pred2ix = pred2ix

    def __call__(self, instance_batch):
        # each instance in the batch looks like:
        # [
        #     list of lists ...: pred_func_nodes_ctxt_preds
        #     list of lists ...: pred_func_nodes_ctxt_args
        #     list of lists ...: logic_expr
        #     list of lists: args_vars
        # ]
        # print (instance_batch)
        batch_size = len(instance_batch)
        instance = instance_batch[0]

        pred_func_nodes_ctxt_predargs = [
            torch.tensor(predargs, dtype=torch.int32) for predargs in instance[0]
        ]
        pred_func_nodes_ctxt_predargs_len = torch.tensor(
            [len(predargs) for predargs in instance[0]]
        )
        for a in [len(predargs) for predargs in instance[0]]:
            if a == 0:
                print(instance)
                input()
        pred_func_nodes_ctxt_predargs_padded = pad_sequence(
            pred_func_nodes_ctxt_predargs, batch_first=True
        )

        logic_expr = torch.tensor(instance[2], dtype=torch.int32).unsqueeze(0)
        args_vars_batch = torch.tensor(instance[3], dtype=torch.int32).unsqueeze(0)

        collated = {
            "encoder": [
                pred_func_nodes_ctxt_predargs_padded,
                pred_func_nodes_ctxt_predargs_len,
            ],
            "decoder": [logic_expr, args_vars_batch],
        }

        return collated


class MyTruthCollator(object):
    def __init__(self, pred2ix):
        self.pred2ix = pred2ix

    def __call__(self, instance_batch):
        # each instance in the batch looks like:
        # [
        #     list of lists ...: pred_func_nodes_ctxt_preds
        #     list of lists ...: pred_func_nodes_ctxt_args
        #     list of lists ...: logic_expr
        #     list of lists: args_vars
        # ]
        batch_size = len(instance_batch)
        instance = instance_batch[0]

        pred_func_nodes_ctxt_preds = torch.tensor(instance[0], dtype=torch.int32)
        pred_func_nodes_ctxt_args = torch.tensor(instance[1], dtype=torch.int32)

        logic_expr = torch.tensor(instance[2], dtype=torch.int32).unsqueeze(0)
        args_vars_batch = torch.tensor(instance[3], dtype=torch.int32).unsqueeze(0)

        collated = {
            "encoder": [pred_func_nodes_ctxt_preds, pred_func_nodes_ctxt_args],
            "decoder": [logic_expr, args_vars_batch],
        }

        return collated


class PASGenCollator(object):
    def __init__(self, pred2ix):
        self.pred2ix = pred2ix

    def __call__(self, instance_batch):
        # each instance in the batch looks like:
        # [
        #     list of lists ...: pred_func_nodes_ctxt_preds
        #     list of lists ...: pred_func_nodes_ctxt_args
        #     list of lists ...: arg2pred_funcs_ix
        #     list of lists: arg2vars
        #     list of list: args_num_sum
        # ]
        batch_size = len(instance_batch)
        instance = instance_batch[0]

        pred_func_nodes_ctxt_predargs = [
            torch.tensor(predargs, dtype=torch.int32) for predargs in instance[0]
        ]
        pred_func_nodes_ctxt_predargs_len = torch.tensor(
            [len(predargs) for predargs in instance[0]]
        )
        pred_func_nodes_ctxt_predargs_padded = pad_sequence(
            pred_func_nodes_ctxt_predargs, batch_first=True
        )
        for a in [len(predargs) for predargs in instance[0]]:
            if a == 0:
                print(instance)
                input()

        pred_funcs_ix_list = instance[2]
        vars_list = instance[3]
        args_num_sum_list = instance[4]

        max_len = max([len(pred_funcs_ix) for pred_funcs_ix in pred_funcs_ix_list])

        len2pred_funcs_ix_list = [[] for i in range(max_len)]
        len2vars_list = [[] for i in range(max_len)]
        len2args_num_sum_list = [[] for i in range(max_len)]
        len2num_pred_funcs = [0 for i in range(max_len)]

        for pred_idx, pred_funcs_ix in enumerate(pred_funcs_ix_list):
            len2pred_funcs_ix_list[len(pred_funcs_ix) - 1].append(pred_funcs_ix)
            len2vars_list[len(pred_funcs_ix) - 1].append(vars_list[pred_idx])
            len2args_num_sum_list[len(pred_funcs_ix) - 1].append(
                args_num_sum_list[pred_idx]
            )
            len2num_pred_funcs[len(pred_funcs_ix) - 1] = len(pred_funcs_ix)

        num_pred_funcs_list = [
            len(pred_funcs_ix)
            for pred_funcs_ix in pred_funcs_ix_list
            if len(pred_funcs_ix) != 0
        ]  # wrong if we consider expanded corod):

        terms_pred_funcs_ix_list = [
            torch.tensor(pred_funcs_ix_list)
            for pred_funcs_ix_list in len2pred_funcs_ix_list
            if pred_funcs_ix_list != []
        ]
        terms_vars_list = [
            torch.tensor(vars_list) for vars_list in len2vars_list if vars_list != []
        ]
        terms_args_num_sum_list = [
            torch.tensor(args_num_sum_list)
            for args_num_sum_list in len2args_num_sum_list
            if args_num_sum_list != []
        ]
        terms_num_pred_funcs = torch.tensor(
            [
                num_pred_funcs
                for num_pred_funcs in len2num_pred_funcs
                if num_pred_funcs != 0
            ]
        )

        collated = {
            "encoder": [
                pred_func_nodes_ctxt_predargs_padded,
                pred_func_nodes_ctxt_predargs_len,
            ],
            "decoder": [
                terms_pred_funcs_ix_list,
                terms_vars_list,
                terms_args_num_sum_list,
            ],
        }

        return collated


class MyGenCollator(object):
    def __init__(self, pred2ix):
        self.pred2ix = pred2ix

    def __call__(self, instance_batch):
        # each instance in the batch looks like:
        # [
        #     list of lists ...: pred_func_nodes_ctxt_preds
        #     list of lists ...: pred_func_nodes_ctxt_args
        #     list of lists ...: arg2pred_funcs_ix
        #     list of lists: arg2vars
        #     list of list: args_num_sum
        # ]
        batch_size = len(instance_batch)
        instance = instance_batch[0]

        pred_func_nodes_ctxt_preds = torch.tensor(instance[0], dtype=torch.int32)
        pred_func_nodes_ctxt_args = np.array(instance[1])

        data, col, row = pred_func_nodes_ctxt_args
        pred_func_nodes_ctxt_args = torch.tensor(
            sparse.csr_array(
                (data, (row, col)), shape=list(pred_func_nodes_ctxt_preds.shape)
            ).toarray()
        )

        pred_funcs_ix_list = instance[2]
        vars_list = instance[3]
        args_num_sum_list = instance[4]

        max_len = max([len(pred_funcs_ix) for pred_funcs_ix in pred_funcs_ix_list])

        len2pred_funcs_ix_list = [[] for i in range(max_len)]
        len2vars_list = [[] for i in range(max_len)]
        len2args_num_sum_list = [[] for i in range(max_len)]
        len2num_pred_funcs = [0 for i in range(max_len)]

        for pred_idx, pred_funcs_ix in enumerate(pred_funcs_ix_list):
            len2pred_funcs_ix_list[len(pred_funcs_ix) - 1].append(pred_funcs_ix)
            len2vars_list[len(pred_funcs_ix) - 1].append(vars_list[pred_idx])
            len2args_num_sum_list[len(pred_funcs_ix) - 1].append(
                args_num_sum_list[pred_idx]
            )
            len2num_pred_funcs[len(pred_funcs_ix) - 1] = len(pred_funcs_ix)

        num_pred_funcs_list = [
            len(pred_funcs_ix)
            for pred_funcs_ix in pred_funcs_ix_list
            if len(pred_funcs_ix) != 0
        ]  # wrong if we consider expanded corod):

        terms_pred_funcs_ix_list = [
            torch.tensor(pred_funcs_ix_list)
            for pred_funcs_ix_list in len2pred_funcs_ix_list
            if pred_funcs_ix_list != []
        ]
        terms_vars_list = [
            torch.tensor(vars_list) for vars_list in len2vars_list if vars_list != []
        ]
        terms_args_num_sum_list = [
            torch.tensor(args_num_sum_list)
            for args_num_sum_list in len2args_num_sum_list
            if args_num_sum_list != []
        ]
        terms_num_pred_funcs = torch.tensor(
            [
                num_pred_funcs
                for num_pred_funcs in len2num_pred_funcs
                if num_pred_funcs != 0
            ]
        )

        collated = {
            "encoder": [pred_func_nodes_ctxt_preds, pred_func_nodes_ctxt_args],
            "decoder": [
                terms_pred_funcs_ix_list,
                terms_vars_list,
                terms_args_num_sum_list,
            ],
        }

        return collated


class EvalRelpronPASCollator(object):
    def __init__(self):
        pass

    def __call__(self, instance_batch):
        # each instance in the batch looks like:
        # [
        #     list of lists ...: pred_func_nodes_ctxt_preds
        #     list of lists ...: pred_func_nodes_ctxt_args
        #     list of lists ...: logic_expr
        #     list of lists: args_vars
        #     labels  # for eval and display of reulsts
        #     full_props  # for eval and display of reulsts
        # ]
        # only support batch_size = 1
        instance = instance_batch[0]
        pred_func_nodes_ctxt_predargs = [
            torch.tensor(predargs, dtype=torch.int32) for predargs in instance[0]
        ]
        pred_func_nodes_ctxt_predargs_len = torch.tensor(
            [len(predargs) for predargs in instance[0]]
        )
        pred_func_nodes_ctxt_predargs_padded = pad_sequence(
            pred_func_nodes_ctxt_predargs, batch_first=True
        )

        # pred_func_nodes_ctxt_args = instance[1]
        logic_expr = torch.tensor(instance[2], dtype=torch.int32)
        args_vars_batch = torch.tensor(instance[3], dtype=torch.int32)
        term = instance[4]
        labels = instance[5]
        full_props = instance[6]

        return {
            "encoder": [
                pred_func_nodes_ctxt_predargs_padded,
                pred_func_nodes_ctxt_predargs_len,
            ],
            "decoder": [logic_expr, args_vars_batch],
            "term": term,
            "labels": labels,
            "full_props": full_props,
        }


# class EvalRelpronMyCollator(object):

#     def __init__(self):
#         pass

#     def __call__(self, instance_batch):

#         # each instance in the batch looks like:
#             # [
#             #     list of lists ...: pred_func_nodes_ctxt_preds
#             #     list of lists ...: pred_func_nodes_ctxt_args
#             #     list of lists ...: logic_expr
#             #     list of lists: args_vars
#             #     labels  # for eval and display of reulsts
#             #     full_props  # for eval and display of reulsts
#             # ]
#         # only support batch_size = 1
#         instance = instance_batch[0]
#         pred_func_nodes_ctxt_preds = torch.tensor(instance[0], dtype = torch.int32)
#         pred_func_nodes_ctxt_args = torch.tensor(instance[1], dtype = torch.int32)
#         logic_expr = torch.tensor(instance[2], dtype = torch.int32)
#         args_vars_batch = torch.tensor(instance[3], dtype = torch.int32)
#         term = instance[4]
#         labels = instance[5]
#         full_props = instance[6]

#         return {
#             "encoder": [pred_func_nodes_ctxt_preds, pred_func_nodes_ctxt_args],
#             "decoder": [logic_expr, args_vars_batch],
#             "term": term,
#             "labels": labels,
#             "full_props": full_props
#         }


class EvalGS2011PASCollator(object):
    def __init__(self):
        pass

    def __call__(self, instance_batch):
        # each instance in the batch looks like:
        # sample = [
        #     self.pred_func_nodes_ctxt_preds_list[idx], self.pred_func_nodes_ctxt_preds_list[idx],
        #     self.logic_expr_list, self.vars_unzipped_list,
        #      self.scores_list[idx]
        # ]
        # only support batch_size = 1
        instance = instance_batch[0]

        pred_func_nodes_ctxt_predargs = [
            torch.tensor(predargs, dtype=torch.int32) for predargs in instance[0]
        ]
        pred_func_nodes_ctxt_predargs_len = torch.tensor(
            [len(predargs) for predargs in instance[0]]
        )
        pred_func_nodes_ctxt_predargs_padded = pad_sequence(
            pred_func_nodes_ctxt_predargs, batch_first=True
        )
        logic_expr_list = instance[2]
        args_vars_batch = torch.tensor(instance[3])
        scores_list = instance[4]
        ix2svo = instance[5]

        return {
            "encoder": [
                pred_func_nodes_ctxt_predargs_padded,
                pred_func_nodes_ctxt_predargs_len,
            ],
            "decoder": [logic_expr_list, args_vars_batch],
            "eval": [scores_list, ix2svo],
        }


# class EvalGS2011MyCollator(object):

#     def __init__(self):
#         pass

#     def __call__(self, instance_batch):

#         # each instance in the batch looks like:
#             # sample = [
#             #     self.pred_func_nodes_ctxt_preds_list[idx], self.pred_func_nodes_ctxt_preds_list[idx],
#             #     self.logic_expr_list, self.vars_unzipped_list,
#             #      self.scores_list[idx]
#             # ]
#         # only support batch_size = 1
#         instance = instance_batch[0]

#         pred_func_nodes_ctxt_predORpredargs = torch.tensor(instance[0], dtype = torch.int32)
#         pred_func_nodes_ctxt_args = torch.tensor(instance[1], dtype = torch.int32)
#         logic_expr_list = instance[2]
#         args_vars_batch = torch.tensor(instance[3], dtype = torch.int32)
#         scores_list = instance[4]
#         ix2svo = instance[5]

#         return {
#             "encoder": [pred_func_nodes_ctxt_predORpredargs, pred_func_nodes_ctxt_args],
#             "decoder": [logic_expr_list, args_vars_batch],
#             "eval": [scores_list, ix2svo]
#         }


class EvalWeeds2014PASCollator(object):
    def __init__(self):
        pass

    def __call__(self, instance_batch):
        # each instance in the batch looks like:
        # sample = [
        #     self.pred_func_nodes_ctxt_preds_list[idx], self.pred_func_nodes_ctxt_preds_list[idx],
        #     self.logic_expr_list, self.vars_unzipped_list,
        #      self.scores_list[idx]
        # ]
        # only support batch_size = 1
        instance = instance_batch[0]

        pred_func_nodes_ctxt_predargs = [
            torch.tensor(predargs, dtype=torch.int32) for predargs in instance[0]
        ]
        pred_func_nodes_ctxt_predargs_len = torch.tensor(
            [len(predargs) for predargs in instance[0]]
        )
        pred_func_nodes_ctxt_predargs_padded = pad_sequence(
            pred_func_nodes_ctxt_predargs, batch_first=True
        )
        logic_expr_list = instance[2]
        args_vars_batch = torch.tensor(instance[3])
        scores_list = instance[4]
        ix2svo = instance[5]

        return {
            "encoder": [
                pred_func_nodes_ctxt_predargs_padded,
                pred_func_nodes_ctxt_predargs_len,
            ],
            "decoder": [logic_expr_list, args_vars_batch],
            "eval": [scores_list, ix2svo],
        }


class TCSTransformerCollator(object):
    """
    Collate function to convert TCS samples into transformer-friendly format

    Converts structured predicate-argument data into:
    - Flattened sequences with separator tokens between nodes
    - Node boundary indices for pooling
    - Target semantic function labels
    """

    def __init__(self, sep_token_id=1, pad_token_id=0):
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Batch size should be 1 for TCS data (as per original implementation)
        sample = batch[0]

        # Extract predicate-argument context for each node
        pred_func_nodes_ctxt_predargs = sample[0]  # List of lists

        # Flatten into sequence with separators
        input_sequence = []
        node_boundaries = [0]  # Track where each node starts/ends

        for node_predargs in pred_func_nodes_ctxt_predargs:
            # Add predicate-argument tokens for this node
            input_sequence.extend(node_predargs)
            # Add separator token
            input_sequence.append(self.sep_token_id)
            node_boundaries.append(len(input_sequence))

        # Convert to tensor
        input_ids = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(
            0
        )  # [1, L]

        # Extract target information
        # For training, we want to predict which semantic functions are present
        # We'll use the pred_funcs_ix_list as targets
        pred_funcs_ix_list = sample[2] if len(sample) > 2 else []

        # Create target tensor (multi-label classification for each node)
        num_nodes = len(pred_func_nodes_ctxt_predargs)

        # For simplicity, we'll use a reconstruction target:
        # Try to predict the predicate-argument structure itself
        targets = []
        for node_predargs in pred_func_nodes_ctxt_predargs:
            if node_predargs:
                # Use first predicate-argument as target (simplified)
                targets.append(node_predargs[0])
            else:
                targets.append(self.pad_token_id)

        targets_tensor = torch.tensor(targets, dtype=torch.long).unsqueeze(
            0
        )  # [1, num_nodes]

        return {
            "input_ids": input_ids,
            "node_boundaries": [node_boundaries],
            "targets": targets_tensor,
            "num_nodes": num_nodes,
        }
