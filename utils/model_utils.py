"""
Version : 3
Authors : Rahul Lokesh, Shivam Ojha, Sushant Tiwari
Version Date : 16th Dec 2021
Description : Utils file for Residual Attention Network Implementation
"""
from enum import Enum
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import BatchNormalization, Add, Flatten, Dense, LayerNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, AveragePooling2D
from utils.res_unit import residual_unit
from utils.learning_mech import attention_residual_learning, naive_attention_learning

class AttentionModule(Enum):
    """
    Enum class to define attention model (56 or 92)
    """
    ATTENTION56 = 1
    ATTENTION92 = 2

class LearningOutput(Enum):
    """
    Enum class to define learning output
    """
    NAIVE_LEARNING = 1
    ATTENTION_RESIDUAL_LEARNING = 2

class AttentionActivationType(Enum):
    """
    Enum class to define attention activation type
    """
    MIXED_ATTENTION = 1
    SPATIAL_ATTENTION = 2

class AttentionModuleStage(Enum):
    """
    Enum class to define attention module stage
    """
    ATTN_MOD_STG1 = 1
    ATTN_MOD_STG2 = 2
    ATTN_MOD_STG3 = 3


class ResidualAttentionNetwork():
    """
    This class is for Residual Attention Network implementation. It uses all the units & functions required for constructing the attention network like attention units, soft mask branch
    trunk branch, attention module stages etc.
    """
    def __init__(self, input_shape, output_size, attention_model=AttentionModule.ATTENTION56,
                    learning=LearningOutput.ATTENTION_RESIDUAL_LEARNING,
                        attention_activation_type=AttentionActivationType.MIXED_ATTENTION,
                            p=1, t=2, r=1):
        """
        Init function

        Args:
            input_shape: 3 elements tuple (width, height, channel)
            output_size (int): number of categories
            attention_model (enum, optional): [description]. Defaults to AttentionModule.ATTENTION56.
            learning (enum, optional): [description]. Defaults to LearningOutput.ATTENTION_RESIDUAL_LEARNING.
            attention_activation_type (enum, optional): [description]. Defaults to AttentionActivationType.MIXED_ATTENTION.
            p (int, optional): number of residual units in each stage. Defaults to 1.
            t (int, optional): number of residual units in trunk branch. Defaults to 2.
            r (int, optional): number of residual units in soft mask branch. Defaults to 1.
        """
        self.input_shape = input_shape
        self.output_size = output_size
        self.p = p
        self.t = t
        self.r = r
        self.attention_model = attention_model
        self.learning = learning
        self.attention_activation_type = attention_activation_type


    def attention_mod(self):
        """
        Attention-56 & 92
        For attention-56, one attention module is used per stage.
        For Attention-92, 1 attention module is used in stage-1, 2 attention modules
        are used for stage-2 & 3 attention modules are used in stage-3
        """

        attention_model = self.attention_model
        input_data = Input(shape=self.input_shape) # CIFAR (32,32)

        #adding convolution layer for stage-1 keeping the 'same' padding for equal dimensions
        stage_1_conv = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_data) # CIFAR (32,32)
        max_pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(stage_1_conv) # CIFAR (16,16)

        # Adding the Residual unit for stage-1
        stage_1_residual = residual_unit(max_pool_1, filters=[16, 16, 64]) # CIFAR (16,16)

        # Calling attention module for stage-1 with learning mechanism (which is the input & can be Attention Residual or Naive Learning).
        # default learning mechanism is the Attention Residual Mechanism
        stage_1_attention = self.attention_mod_stg(stage_1_residual, filters=[16, 16, 64],
                                                    learning_mechanism=self.learning,
                                                        attention_mod_stg=AttentionModuleStage.ATTN_MOD_STG1) # CIFAR (16,16)

        # Residual-Attention Module for stage 2
        # For a trunk layer depth of 56, only 1 attention unit per stage will be used. For trunk layer depth of 96, 1 attention unit will be used for stage-1, 2 attention units for stage 2 & 3 attention units for stage 3
        # ATTENTION56 is taken as default
        #adding the common attention & residual units for Attention-56
        stage_2_residual = residual_unit(stage_1_attention, filters=[32, 32, 128], conv_stride=2) # CIFAR (8,8)
        stage_2_attention = self.attention_mod_stg(stage_2_residual, filters=[32, 32, 128],
                                                       learning_mechanism=self.learning,
                                                       attention_mod_stg=AttentionModuleStage.ATTN_MOD_STG2) # CIFAR (8,8)

        #Adding extra attenion unit for Attention-92 for stage-2
        if attention_model is AttentionModule.ATTENTION92:
            stage_2_attention = self.attention_mod_stg(stage_2_residual, filters=[32, 32, 128],
                                                           learning_mechanism=self.learning,
                                                            attention_mod_stg=AttentionModuleStage.ATTN_MOD_STG2) # CIFAR (8,8)

        # Residual-Attention Module for stage 3
        #Adding residual & attention units for stage 3
        stage_3_residual = residual_unit(stage_2_attention, filters=[64, 64, 256], conv_stride=2) # CIFAR (4,4)
        stage_3_attention = self.attention_mod_stg(stage_3_residual, filters=[64, 64, 256],
                                                       learning_mechanism=self.learning,
                                                        attention_mod_stg=AttentionModuleStage.ATTN_MOD_STG3) # CIFAR (4,4)


        #Adding 2 more attention units for Attention-92
        if attention_model is AttentionModule.ATTENTION92:
            stage_3_attention = self.attention_mod_stg(stage_3_attention, filters=[128, 128, 512],
                                                           learning_mechanism=self.learning,
                                                           attention_mod_stg=AttentionModuleStage.ATTN_MOD_STG3) # CIFAR (4,4)
            stage_3_attention = self.attention_mod_stg(stage_3_attention, filters=[128, 128, 512], learning_mechanism=self.learning,
                                                           attention_mod_stg=AttentionModuleStage.ATTN_MOD_STG3) # CIFAR (4,4)

        #Adding another out module
        stage_3_attention = residual_unit(stage_3_attention, filters=[128,128,512])  #4x4

        #adding residual units after the 3rd stage
        stage_3_attention = residual_unit(stage_3_attention, filters=[128, 128, 512]) #4x4
        stage_3_attention = residual_unit(stage_3_attention, filters=[128, 128, 512]) #4x4

        #doing batch normalisation for faster computation or speeding up the training process
        norm_var_2 = BatchNormalization()(stage_3_attention) #4x4

        #adding "relu" activation for non-linearity
        activation_2 = Activation('relu')(norm_var_2) #4x4

        #doing average pooling, taking average of the samples in the pool size
        avg_pool = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')(activation_2) #1x1

        #flattening & passing to the dense layer for obtaining the output. Passing the output into our model
        flatten_layer = Flatten()(avg_pool)
        output = Dense(self.output_size, activation='softmax')(flatten_layer)

        model = Model(inputs=input_data, outputs=output)

        return model


    def trunk_branch(self, input_data, filters):
        """
        Trunk branch is used for feature processing. It has 't' number of residual units in
        total and this can be adapted to any architecture like "ResNet"

        Args:
            input_data: output after p number of pre-processing Residual Units before splitting into trunk branch and mask branch
            filters: filter for the trunk branch

        Returns:trunk_output - returns trunk output corresponding to the number of residual units,t in the trunk branch
        """

        # 't' number of residual units are used for the trunk branch taking output of the pth pre-processing residual unit & filter as inputs
        trunk_output = input_data
        for t in range(self.t):
            trunk_output = residual_unit(trunk_output, filters)
        return trunk_output


    def soft_mask_branch(self, input_data, filters, attention_mod_stg):
        """
        Soft mask branch is used to mask the redundant environment in the image. 
        After p no. of pre-processing residual units, the output is given to trunk branch
        for feature processing & mask branch for masking the attention environment

        Args:
            input_data : output after p number of pre-processing Residual Units before splitting into trunk branch and mask branch
            filters: filter for the mask branch
            attention_mod_stg: attention module stage for the soft mask branch

        Returns:
            soft_mask_unit: returns output for the soft mask unit which along with the trunk output can be used for Naive Learning or Residual Attention Learning
        """
        #max pooling  is done to increase the receptive ﬁeld rapidly after a small number of Residual Units.
        #max pooling layer performs downsampling as it divides the input into rectangular pooling regions, then computes the maximum of each region.
        downsample_data_1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_data)

        #'r' no. of residual units between the adjacent pooling layers in each branch
        for _ in range(self.r):
            downsample_data_1 = residual_unit(downsample_data_1, filters)

        #From the Enum class to define attention module stage, 1st stage condition is specified for skip connections, downsampling - Maxpooling, upsampling & identity function implementation
        if attention_mod_stg == AttentionModuleStage.ATTN_MOD_STG1:

            #for stage 1, residual unit function is called and skip connections are assigned in the soft mask branch
            skip_connection = residual_unit(downsample_data_1, filters)
            #max pooling is done after calling residual_unit & data is down sampled to increase the receptive field
            downsample_data_2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(downsample_data_1)

            # downsampling and adding residual units "r" times
            for _ in range(self.r):
                downsample_data_2 = residual_unit(downsample_data_2, filters)
            #for stage 1, residual unit function is called and skip connections are assigned in the soft mask branch
            skip_init_inside = residual_unit(downsample_data_2, filters)
            #max pooling is done after calling residual_unit & data is down sampled to increase the receptive field
            downsample_data_3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(downsample_data_2)

           #doing further downsampling (2 times the number of residual units) by performing maxpooling
            for _ in range(self.r * 2):
                downsample_data_3 = residual_unit(downsample_data_3, filters)

           #up sampling  to make the spatial dimension in accordance to the ouput of the next stage
            upsampled_unit_1 = UpSampling2D(size=(2, 2))(downsample_data_3)

            #implementing residual skip connection for function identity to ensure performance replication for upsampled_unit_1
            add_unit_1 = Add()([upsampled_unit_1, skip_init_inside])
            for _ in range(self.r):
                add_unit_1 = residual_unit(add_unit_1, filters)

            #up sampling to make the spatial dimension in accordance to the ouput of the next stage
            upsampled_unit_2 = UpSampling2D(size=(2, 2))(add_unit_1)

            #implementing residual skip connection for function identity to ensure performance replication for upsampled_unit_2
            add_unit_last = Add()([upsampled_unit_2, skip_connection])
            for _ in range(self.r):
                add_unit_last = residual_unit(add_unit_last, filters)

            #up sampling to make the spatial dimension in accordance to the ouput of the next stage
            upsampled_unit_last = UpSampling2D(size=(2, 2))(add_unit_last)

        #From the Enum class to define attention module stage, 2nd stage condition is specified for skip connections, downsampling - Maxpooling, upsampling & identity function implementation
        if attention_mod_stg == AttentionModuleStage.ATTN_MOD_STG2:
            #for stage 2, residual unit function is called and skip connections are assigned in the soft mask branch
            skip_connection = residual_unit(downsample_data_1, filters)

            #max pooling layer to perform downsampling as it divides the input into rectangular pooling regions, then computes the maximum of each region. It extracts maximum information from the region
            downsample_data_3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(downsample_data_1)
            for _ in range(self.r * 2):
                downsample_data_3 = residual_unit(downsample_data_3, filters)

            #up sampling to make the spatial dimension in accordance to the ouput of the next stage
            upsampled_unit_1 = UpSampling2D(size=(2, 2))(downsample_data_3)

            #implementing residual skip connection for function identity to ensure performance replication for upsampled_unit_1
            add_unit_last = Add()([upsampled_unit_1, skip_connection])
            for _ in range(self.r):
                add_unit_last = residual_unit(add_unit_last, filters)

             #implementing residual skip connection for function identity to ensure performance replication for upsampled_unit_2
            upsampled_unit_last = UpSampling2D(size=(2, 2))(add_unit_last)

        #From the Enum class to define attention module stage, 3rd stage condition is specified for skip connections, downsampling - Maxpooling, upsampling & identity function implementation
        if attention_mod_stg == AttentionModuleStage.ATTN_MOD_STG3:
            #for stage 3, residual unit function is called and skip connections are assigned in the soft mask branch
            upsampled_unit_last = UpSampling2D(size=(2, 2))(downsample_data_1)

        filter_conv = upsampled_unit_last.shape[-1]
        #adding in 2 convolution layers
        convolution_1 = Conv2D(filters=filter_conv, kernel_size=(1, 1), padding='same')(upsampled_unit_last)
        #convolution_1 = Conv2D(filters=filter_conv, kernel_size=(1, 1), padding='same')(upsampled_unit_last)
        convolution_2 = Conv2D(filters=filter_conv, kernel_size=(1, 1), padding='same')(convolution_1)

        #if the attention type of activation is chosen as Mixed (also a default), then sigmoid is taken as the activation function
        if self.attention_activation_type is AttentionActivationType.MIXED_ATTENTION:
            soft_mask_unit = Activation('sigmoid')(convolution_2)

        #if the attention type of activation is chosen as Spatial , then layer normalisation is done followed by the sigmoid activation
        elif self.attention_activation_type is AttentionActivationType.SPATIAL_ATTENTION:
            layer_norm_1 = LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True,
                                          beta_initializer="zeros", gamma_initializer="ones",
                                          beta_regularizer=None, gamma_regularizer=None,
                                          beta_constraint=None, gamma_constraint=None)(convolution_2)
            soft_mask_unit = Activation('sigmoid')(layer_norm_1)
        #returns soft mask unit as the ouput after implementation of masking, residual network implementation (identity replication, down & up upsampling ) & convolution layers
        return soft_mask_unit


    def attention_mod_stg(self, input_unit, filters, learning_mechanism, attention_mod_stg):
        """
        Function to define attention module stage

        Args:
            input_unit: residual units for stage 1 or 2 or 3
            filters: filter used for the respective attention module stage
            learning_mechanism: type of learning. Naive Learning or Attention Residual Learning. Naive Learning will result in performance drop as it doesn't involve identity replication of
            trunk output to the product of the output of mask & trunk units. In case of Attention Residual Learning, soft mask unit is taken as an identical mapping leading to replicaiton of
            performance in the worst case scenario
            attention_mod_stg: To select the stage of the attention module: Stage-1, Stage-2 or Stage-3

        Returns:
            output_unit: returns output to the residual for the next step
        """

        #adding 'p' attention module units where 'p' is number of residual units before mask & trunk branch splits
        for _ in range(self.p):
            am_unit = residual_unit(input_unit, filters)

        #After splitting, computing trunk output after passing through trunk branch
        trunk_output = self.trunk_branch(am_unit, filters)

        #taking output of the softmask after passing through the soft mask branch
        soft_mask_output = self.soft_mask_branch(am_unit, filters, attention_mod_stg)

        #Creating different conditions for two learning mechanisms. For Naive Learning, direct multiplication of trunk & soft mask output is done without any identity mapping
        #For Residual Learning, attention_residual_learning function is called to compute identity mapping
        if learning_mechanism == LearningOutput.NAIVE_LEARNING:
            output_unit = naive_attention_learning(trunk_output, soft_mask_output)
        elif learning_mechanism == LearningOutput.ATTENTION_RESIDUAL_LEARNING:
            output_unit = attention_residual_learning(trunk_output, soft_mask_output)
        #adding in 'p' no. of residual units & returning the output to the residual to be used for the next step/stage
        for _ in range(self.p):
            output_unit = residual_unit(output_unit, filters)
        return output_unit

