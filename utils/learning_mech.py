"""
Version : 1
Version Date : 30th Nov 2021
Description : Utils file for learning mechanisms
"""
from tensorflow.keras.layers import Add, Multiply


def attention_residual_learning(trunk_output, soft_mask_output):
    """
    Function to implement attention residual learning module
    Args:
        trunk_output: Output after passing through the trunk branch which is used for feature processing
        soft_mask_output: Output after passing through the soft mask branch

    Returns:
        output: Combines outputs from trunk & soft mask branches also using the 
        identity mapping/residual replication
    """
    # In attention residual learning, the output obtained by multiplying
    # trunk & soft mask outputs is added to the trunk output for identity mapping if the value
    # obtained after direct multiplication of trunk & mask branches is low
    output = Multiply()([trunk_output, soft_mask_output])
    output = Add()([output, trunk_output])
    return output


def naive_attention_learning(trunk_output, soft_mask_output):
    """
    Function to implement naive attention learning module
    trunk branch & soft mask ouputs will be multiplied for the naive attention learning

    Args:
        trunk_output: Output after passing through the trunk branch which is used for feature processing
        soft_mask_output: Output after passing through the soft mask branch

    Returns:
        output: Combines outputs from trunk & soft mask branches without using the identity mapping/residual replication
    """
    output = Multiply()([trunk_output, soft_mask_output])
    return output
