��
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	"serve*2.4.32unknown8��

NoOpNoOp
�	
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�	
value�	B�	 B�	
}
	lbcpe

modulation
trainable_variables
	variables
regularization_losses
	keras_api

signatures
_
product
	trainable_variables

	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
 
 
 
�
non_trainable_variables

layers
layer_regularization_losses
trainable_variables
metrics
	variables
regularization_losses
layer_metrics
 
R
trainable_variables
	variables
regularization_losses
	keras_api
 
 
 
�
non_trainable_variables

layers
layer_regularization_losses
	trainable_variables
metrics

	variables
regularization_losses
layer_metrics
 
 
 
�
non_trainable_variables

 layers
!layer_regularization_losses
trainable_variables
"metrics
	variables
regularization_losses
#layer_metrics
 

0
1
 
 
 
 
 
 
�
$non_trainable_variables

%layers
&layer_regularization_losses
trainable_variables
'metrics
	variables
regularization_losses
(layer_metrics
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
z
serving_default_input_2Placeholder*'
_output_shapes
:��������� *
dtype0*
shape:��������� 
�
PartitionedCallPartitionedCallserving_default_input_1serving_default_input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_498227
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_498589
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_498599��
�`
m
C__inference_encoder_layer_call_and_return_conditional_losses_498315
input_1
input_2
identity�
7linear_block_code_product_encoder_with_external_g/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @29
7linear_block_code_product_encoder_with_external_g/mul/y�
5linear_block_code_product_encoder_with_external_g/mulMulinput_1@linear_block_code_product_encoder_with_external_g/mul/y:output:0*
T0*'
_output_shapes
:���������27
5linear_block_code_product_encoder_with_external_g/mul�
7linear_block_code_product_encoder_with_external_g/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?29
7linear_block_code_product_encoder_with_external_g/sub/x�
5linear_block_code_product_encoder_with_external_g/subSub@linear_block_code_product_encoder_with_external_g/sub/x:output:09linear_block_code_product_encoder_with_external_g/mul:z:0*
T0*'
_output_shapes
:���������27
5linear_block_code_product_encoder_with_external_g/sub�
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/ShapeShape9linear_block_code_product_encoder_with_external_g/sub:z:0*
T0*
_output_shapes
:2W
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape�
clinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2e
clinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack�
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2g
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1�
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2�
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_sliceStridedSlice^linear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape:output:0llinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack:output:0nlinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1:output:0nlinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2_
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice�
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2a
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1�
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shapePackflinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice:output:0hlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2_
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape�
Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ReshapeReshapeinput_2flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape:output:0*
T0*'
_output_shapes
:��������� 2Y
Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape�
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2`
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim�
Zlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims
ExpandDims9linear_block_code_product_encoder_with_external_g/sub:z:0glinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2\
Zlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims�
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2W
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/stack�
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/TileTileclinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims:output:0^linear_block_code_product_encoder_with_external_g/product_with_external_weights/stack:output:0*
T0*+
_output_shapes
:���������2V
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile�
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2`
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm�
Ylinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose	Transpose`linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0glinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm:output:0*
T0*'
_output_shapes
: ���������2[
Ylinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose�
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCallPartitionedCall]linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_f_4980262a
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall�
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/MulMul]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile:output:0hlinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2U
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul�
`linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2b
`linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm�
[linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1	Transpose`linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0ilinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm:output:0*
T0*'
_output_shapes
: ���������2]
[linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1�
alinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1PartitionedCall_linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_g_4980362c
alinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1�
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/addAddV2Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul:z:0jlinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1:output:0*
T0*+
_output_shapes
:��������� 2U
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/add�
flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2h
flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices�
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ProdProdWlinear_block_code_product_encoder_with_external_g/product_with_external_weights/add:z:0olinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:��������� 2V
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod�
5linear_block_code_product_encoder_with_external_g/NegNeg]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod:output:0*
T0*'
_output_shapes
:��������� 27
5linear_block_code_product_encoder_with_external_g/Neg�
9linear_block_code_product_encoder_with_external_g/SigmoidSigmoid9linear_block_code_product_encoder_with_external_g/Neg:y:0*
T0*'
_output_shapes
:��������� 2;
9linear_block_code_product_encoder_with_external_g/Sigmoid�
*differentiable_bpsk_modulation_layer/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*differentiable_bpsk_modulation_layer/sub/y�
(differentiable_bpsk_modulation_layer/subSub=linear_block_code_product_encoder_with_external_g/Sigmoid:y:03differentiable_bpsk_modulation_layer/sub/y:output:0*
T0*'
_output_shapes
:��������� 2*
(differentiable_bpsk_modulation_layer/sub�
.differentiable_bpsk_modulation_layer/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.differentiable_bpsk_modulation_layer/Greater/y�
,differentiable_bpsk_modulation_layer/GreaterGreater,differentiable_bpsk_modulation_layer/sub:z:07differentiable_bpsk_modulation_layer/Greater/y:output:0*
T0*'
_output_shapes
:��������� 2.
,differentiable_bpsk_modulation_layer/Greater�
/differentiable_bpsk_modulation_layer/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  �?21
/differentiable_bpsk_modulation_layer/SelectV2/t�
/differentiable_bpsk_modulation_layer/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ��21
/differentiable_bpsk_modulation_layer/SelectV2/e�
-differentiable_bpsk_modulation_layer/SelectV2SelectV20differentiable_bpsk_modulation_layer/Greater:z:08differentiable_bpsk_modulation_layer/SelectV2/t:output:08differentiable_bpsk_modulation_layer/SelectV2/e:output:0*
T0*'
_output_shapes
:��������� 2/
-differentiable_bpsk_modulation_layer/SelectV2�
-differentiable_bpsk_modulation_layer/IdentityIdentity6differentiable_bpsk_modulation_layer/SelectV2:output:0*
T0*'
_output_shapes
:��������� 2/
-differentiable_bpsk_modulation_layer/Identity�
.differentiable_bpsk_modulation_layer/IdentityN	IdentityN6differentiable_bpsk_modulation_layer/SelectV2:output:0,differentiable_bpsk_modulation_layer/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-498305*:
_output_shapes(
&:��������� :��������� 20
.differentiable_bpsk_modulation_layer/IdentityN�
IdentityIdentity7differentiable_bpsk_modulation_layer/IdentityN:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�

~
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_498156

inputs

identity_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sub/y[
subSubinputssub/y:output:0*
T0*'
_output_shapes
:��������� 2
sub[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yl
GreaterGreatersub:z:0Greater/y:output:0*
T0*'
_output_shapes
:��������� 2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ��2

SelectV2/e�
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:��������� 2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:��������� 2

Identity�
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-498146*:
_output_shapes(
&:��������� :��������� 2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
+
__inference_f_498553
w
identityU
IdentityIdentityw*
T0*'
_output_shapes
: ���������2

Identity"
identityIdentity:output:0*&
_input_shapes
: ���������:J F
'
_output_shapes
: ���������

_user_specified_namew
�
T
(__inference_encoder_layer_call_fn_498421
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_4982032
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
+
__inference_f_498549
w
identityL
IdentityIdentityw*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*
_input_shapes

: :A =

_output_shapes

: 

_user_specified_namew
�
+
__inference_f_498026
w
identityU
IdentityIdentityw*
T0*'
_output_shapes
: ���������2

Identity"
identityIdentity:output:0*&
_input_shapes
: ���������:J F
'
_output_shapes
: ���������

_user_specified_namew
�
+
__inference_g_498036
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xV
subSubsub/x:output:0w*
T0*'
_output_shapes
: ���������2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
: ���������2

Identity"
identityIdentity:output:0*&
_input_shapes
: ���������:J F
'
_output_shapes
: ���������

_user_specified_namew
�
l
__inference__traced_save_498589
file_prefix
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
�
N
$__inference_signature_wrapper_498227
input_1
input_2
identity�
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_4980552
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�
m
C__inference_encoder_layer_call_and_return_conditional_losses_498203

inputs
inputs_1
identity�
Alinear_block_code_product_encoder_with_external_g/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *v
fqRo
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_4980932C
Alinear_block_code_product_encoder_with_external_g/PartitionedCall�
4differentiable_bpsk_modulation_layer/PartitionedCallPartitionedCallJlinear_block_code_product_encoder_with_external_g/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *i
fdRb
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_49815626
4differentiable_bpsk_modulation_layer/PartitionedCall�
IdentityIdentity=differentiable_bpsk_modulation_layer/PartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
a
E__inference_differentiable_bpsk_modulation_layer_layer_call_fn_498545

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *i
fdRb
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_4981712
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
~
R__inference_linear_block_code_product_encoder_with_external_g_layer_call_fn_498499
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *v
fqRo
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_4980932
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�`
o
C__inference_encoder_layer_call_and_return_conditional_losses_498371
inputs_0
inputs_1
identity�
7linear_block_code_product_encoder_with_external_g/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @29
7linear_block_code_product_encoder_with_external_g/mul/y�
5linear_block_code_product_encoder_with_external_g/mulMulinputs_0@linear_block_code_product_encoder_with_external_g/mul/y:output:0*
T0*'
_output_shapes
:���������27
5linear_block_code_product_encoder_with_external_g/mul�
7linear_block_code_product_encoder_with_external_g/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?29
7linear_block_code_product_encoder_with_external_g/sub/x�
5linear_block_code_product_encoder_with_external_g/subSub@linear_block_code_product_encoder_with_external_g/sub/x:output:09linear_block_code_product_encoder_with_external_g/mul:z:0*
T0*'
_output_shapes
:���������27
5linear_block_code_product_encoder_with_external_g/sub�
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/ShapeShape9linear_block_code_product_encoder_with_external_g/sub:z:0*
T0*
_output_shapes
:2W
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape�
clinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2e
clinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack�
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2g
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1�
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2�
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_sliceStridedSlice^linear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape:output:0llinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack:output:0nlinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1:output:0nlinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2_
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice�
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2a
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1�
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shapePackflinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice:output:0hlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2_
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape�
Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ReshapeReshapeinputs_1flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape:output:0*
T0*'
_output_shapes
:��������� 2Y
Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape�
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2`
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim�
Zlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims
ExpandDims9linear_block_code_product_encoder_with_external_g/sub:z:0glinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2\
Zlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims�
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2W
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/stack�
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/TileTileclinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims:output:0^linear_block_code_product_encoder_with_external_g/product_with_external_weights/stack:output:0*
T0*+
_output_shapes
:���������2V
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile�
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2`
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm�
Ylinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose	Transpose`linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0glinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm:output:0*
T0*'
_output_shapes
: ���������2[
Ylinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose�
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCallPartitionedCall]linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_f_4980262a
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall�
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/MulMul]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile:output:0hlinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2U
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul�
`linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2b
`linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm�
[linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1	Transpose`linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0ilinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm:output:0*
T0*'
_output_shapes
: ���������2]
[linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1�
alinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1PartitionedCall_linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_g_4980362c
alinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1�
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/addAddV2Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul:z:0jlinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1:output:0*
T0*+
_output_shapes
:��������� 2U
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/add�
flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2h
flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices�
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ProdProdWlinear_block_code_product_encoder_with_external_g/product_with_external_weights/add:z:0olinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:��������� 2V
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod�
5linear_block_code_product_encoder_with_external_g/NegNeg]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod:output:0*
T0*'
_output_shapes
:��������� 27
5linear_block_code_product_encoder_with_external_g/Neg�
9linear_block_code_product_encoder_with_external_g/SigmoidSigmoid9linear_block_code_product_encoder_with_external_g/Neg:y:0*
T0*'
_output_shapes
:��������� 2;
9linear_block_code_product_encoder_with_external_g/Sigmoid�
*differentiable_bpsk_modulation_layer/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*differentiable_bpsk_modulation_layer/sub/y�
(differentiable_bpsk_modulation_layer/subSub=linear_block_code_product_encoder_with_external_g/Sigmoid:y:03differentiable_bpsk_modulation_layer/sub/y:output:0*
T0*'
_output_shapes
:��������� 2*
(differentiable_bpsk_modulation_layer/sub�
.differentiable_bpsk_modulation_layer/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.differentiable_bpsk_modulation_layer/Greater/y�
,differentiable_bpsk_modulation_layer/GreaterGreater,differentiable_bpsk_modulation_layer/sub:z:07differentiable_bpsk_modulation_layer/Greater/y:output:0*
T0*'
_output_shapes
:��������� 2.
,differentiable_bpsk_modulation_layer/Greater�
/differentiable_bpsk_modulation_layer/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  �?21
/differentiable_bpsk_modulation_layer/SelectV2/t�
/differentiable_bpsk_modulation_layer/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ��21
/differentiable_bpsk_modulation_layer/SelectV2/e�
-differentiable_bpsk_modulation_layer/SelectV2SelectV20differentiable_bpsk_modulation_layer/Greater:z:08differentiable_bpsk_modulation_layer/SelectV2/t:output:08differentiable_bpsk_modulation_layer/SelectV2/e:output:0*
T0*'
_output_shapes
:��������� 2/
-differentiable_bpsk_modulation_layer/SelectV2�
-differentiable_bpsk_modulation_layer/IdentityIdentity6differentiable_bpsk_modulation_layer/SelectV2:output:0*
T0*'
_output_shapes
:��������� 2/
-differentiable_bpsk_modulation_layer/Identity�
.differentiable_bpsk_modulation_layer/IdentityN	IdentityN6differentiable_bpsk_modulation_layer/SelectV2:output:0,differentiable_bpsk_modulation_layer/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-498361*:
_output_shapes(
&:��������� :��������� 20
.differentiable_bpsk_modulation_layer/IdentityN�
IdentityIdentity7differentiable_bpsk_modulation_layer/IdentityN:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
~
R__inference_linear_block_code_product_encoder_with_external_g_layer_call_fn_498505
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *v
fqRo
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_4981262
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
+
__inference_g_498559
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xM
subSubsub/x:output:0w*
T0*
_output_shapes

: 2
subR
IdentityIdentitysub:z:0*
T0*
_output_shapes

: 2

Identity"
identityIdentity:output:0*
_input_shapes

: :A =

_output_shapes

: 

_user_specified_namew
�
R
(__inference_encoder_layer_call_fn_498321
input_1
input_2
identity�
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_4982032
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�
a
E__inference_differentiable_bpsk_modulation_layer_layer_call_fn_498540

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *i
fdRb
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_4981562
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
H
"__inference__traced_restore_498599
file_prefix

identity_1��
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
22
	RestoreV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpd
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
T
(__inference_encoder_layer_call_fn_498427
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_4982162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�g
K
!__inference__wrapped_model_498055
input_1
input_2
identity�
?encoder/linear_block_code_product_encoder_with_external_g/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2A
?encoder/linear_block_code_product_encoder_with_external_g/mul/y�
=encoder/linear_block_code_product_encoder_with_external_g/mulMulinput_1Hencoder/linear_block_code_product_encoder_with_external_g/mul/y:output:0*
T0*'
_output_shapes
:���������2?
=encoder/linear_block_code_product_encoder_with_external_g/mul�
?encoder/linear_block_code_product_encoder_with_external_g/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2A
?encoder/linear_block_code_product_encoder_with_external_g/sub/x�
=encoder/linear_block_code_product_encoder_with_external_g/subSubHencoder/linear_block_code_product_encoder_with_external_g/sub/x:output:0Aencoder/linear_block_code_product_encoder_with_external_g/mul:z:0*
T0*'
_output_shapes
:���������2?
=encoder/linear_block_code_product_encoder_with_external_g/sub�
]encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/ShapeShapeAencoder/linear_block_code_product_encoder_with_external_g/sub:z:0*
T0*
_output_shapes
:2_
]encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape�
kencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2m
kencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack�
mencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2o
mencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1�
mencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2o
mencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2�
eencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_sliceStridedSlicefencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape:output:0tencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack:output:0vencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1:output:0vencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2g
eencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice�
gencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2i
gencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1�
eencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shapePacknencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice:output:0pencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2g
eencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape�
_encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/ReshapeReshapeinput_2nencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape:output:0*
T0*'
_output_shapes
:��������� 2a
_encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape�
fencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2h
fencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim�
bencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims
ExpandDimsAencoder/linear_block_code_product_encoder_with_external_g/sub:z:0oencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2d
bencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims�
]encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2_
]encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/stack�
\encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/TileTilekencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims:output:0fencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/stack:output:0*
T0*+
_output_shapes
:���������2^
\encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile�
fencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2h
fencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm�
aencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose	Transposehencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0oencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm:output:0*
T0*'
_output_shapes
: ���������2c
aencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose�
gencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCallPartitionedCalleencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_f_4980262i
gencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall�
[encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/MulMuleencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile:output:0pencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2]
[encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul�
hencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2j
hencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm�
cencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1	Transposehencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0qencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm:output:0*
T0*'
_output_shapes
: ���������2e
cencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1�
iencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1PartitionedCallgencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_g_4980362k
iencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1�
[encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/addAddV2_encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul:z:0rencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1:output:0*
T0*+
_output_shapes
:��������� 2]
[encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/add�
nencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2p
nencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices�
\encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/ProdProd_encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/add:z:0wencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:��������� 2^
\encoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod�
=encoder/linear_block_code_product_encoder_with_external_g/NegNegeencoder/linear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod:output:0*
T0*'
_output_shapes
:��������� 2?
=encoder/linear_block_code_product_encoder_with_external_g/Neg�
Aencoder/linear_block_code_product_encoder_with_external_g/SigmoidSigmoidAencoder/linear_block_code_product_encoder_with_external_g/Neg:y:0*
T0*'
_output_shapes
:��������� 2C
Aencoder/linear_block_code_product_encoder_with_external_g/Sigmoid�
2encoder/differentiable_bpsk_modulation_layer/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2encoder/differentiable_bpsk_modulation_layer/sub/y�
0encoder/differentiable_bpsk_modulation_layer/subSubEencoder/linear_block_code_product_encoder_with_external_g/Sigmoid:y:0;encoder/differentiable_bpsk_modulation_layer/sub/y:output:0*
T0*'
_output_shapes
:��������� 22
0encoder/differentiable_bpsk_modulation_layer/sub�
6encoder/differentiable_bpsk_modulation_layer/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6encoder/differentiable_bpsk_modulation_layer/Greater/y�
4encoder/differentiable_bpsk_modulation_layer/GreaterGreater4encoder/differentiable_bpsk_modulation_layer/sub:z:0?encoder/differentiable_bpsk_modulation_layer/Greater/y:output:0*
T0*'
_output_shapes
:��������� 26
4encoder/differentiable_bpsk_modulation_layer/Greater�
7encoder/differentiable_bpsk_modulation_layer/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  �?29
7encoder/differentiable_bpsk_modulation_layer/SelectV2/t�
7encoder/differentiable_bpsk_modulation_layer/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ��29
7encoder/differentiable_bpsk_modulation_layer/SelectV2/e�
5encoder/differentiable_bpsk_modulation_layer/SelectV2SelectV28encoder/differentiable_bpsk_modulation_layer/Greater:z:0@encoder/differentiable_bpsk_modulation_layer/SelectV2/t:output:0@encoder/differentiable_bpsk_modulation_layer/SelectV2/e:output:0*
T0*'
_output_shapes
:��������� 27
5encoder/differentiable_bpsk_modulation_layer/SelectV2�
5encoder/differentiable_bpsk_modulation_layer/IdentityIdentity>encoder/differentiable_bpsk_modulation_layer/SelectV2:output:0*
T0*'
_output_shapes
:��������� 27
5encoder/differentiable_bpsk_modulation_layer/Identity�
6encoder/differentiable_bpsk_modulation_layer/IdentityN	IdentityN>encoder/differentiable_bpsk_modulation_layer/SelectV2:output:04encoder/differentiable_bpsk_modulation_layer/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-498045*:
_output_shapes(
&:��������� :��������� 28
6encoder/differentiable_bpsk_modulation_layer/IdentityN�
IdentityIdentity?encoder/differentiable_bpsk_modulation_layer/IdentityN:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�
R
(__inference_encoder_layer_call_fn_498327
input_1
input_2
identity�
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_4982162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�
m
C__inference_encoder_layer_call_and_return_conditional_losses_498216

inputs
inputs_1
identity�
Alinear_block_code_product_encoder_with_external_g/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *v
fqRo
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_4981262C
Alinear_block_code_product_encoder_with_external_g/PartitionedCall�
4differentiable_bpsk_modulation_layer/PartitionedCallPartitionedCallJlinear_block_code_product_encoder_with_external_g/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *i
fdRb
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_49817126
4differentiable_bpsk_modulation_layer/PartitionedCall�
IdentityIdentity=differentiable_bpsk_modulation_layer/PartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

~
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_498171

inputs

identity_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sub/y[
subSubinputssub/y:output:0*
T0*'
_output_shapes
:��������� 2
sub[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yl
GreaterGreatersub:z:0Greater/y:output:0*
T0*'
_output_shapes
:��������� 2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ��2

SelectV2/e�
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:��������� 2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:��������� 2

Identity�
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-498161*:
_output_shapes(
&:��������� :��������� 2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

~
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_498535

inputs

identity_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sub/y[
subSubinputssub/y:output:0*
T0*'
_output_shapes
:��������� 2
sub[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yl
GreaterGreatersub:z:0Greater/y:output:0*
T0*'
_output_shapes
:��������� 2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ��2

SelectV2/e�
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:��������� 2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:��������� 2

Identity�
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-498525*:
_output_shapes(
&:��������� :��������� 2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�/
�
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_498093

inputs
inputs_1
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/y[
mulMulinputsmul/y:output:0*
T0*'
_output_shapes
:���������2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:���������2
sub�
#product_with_external_weights/ShapeShapesub:z:0*
T0*
_output_shapes
:2%
#product_with_external_weights/Shape�
1product_with_external_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������23
1product_with_external_weights/strided_slice/stack�
3product_with_external_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3product_with_external_weights/strided_slice/stack_1�
3product_with_external_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3product_with_external_weights/strided_slice/stack_2�
+product_with_external_weights/strided_sliceStridedSlice,product_with_external_weights/Shape:output:0:product_with_external_weights/strided_slice/stack:output:0<product_with_external_weights/strided_slice/stack_1:output:0<product_with_external_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+product_with_external_weights/strided_slice�
-product_with_external_weights/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2/
-product_with_external_weights/Reshape/shape/1�
+product_with_external_weights/Reshape/shapePack4product_with_external_weights/strided_slice:output:06product_with_external_weights/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+product_with_external_weights/Reshape/shape�
%product_with_external_weights/ReshapeReshapeinputs_14product_with_external_weights/Reshape/shape:output:0*
T0*'
_output_shapes
:��������� 2'
%product_with_external_weights/Reshape�
,product_with_external_weights/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,product_with_external_weights/ExpandDims/dim�
(product_with_external_weights/ExpandDims
ExpandDimssub:z:05product_with_external_weights/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2*
(product_with_external_weights/ExpandDims�
#product_with_external_weights/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2%
#product_with_external_weights/stack�
"product_with_external_weights/TileTile1product_with_external_weights/ExpandDims:output:0,product_with_external_weights/stack:output:0*
T0*+
_output_shapes
:���������2$
"product_with_external_weights/Tile�
,product_with_external_weights/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2.
,product_with_external_weights/transpose/perm�
'product_with_external_weights/transpose	Transpose.product_with_external_weights/Reshape:output:05product_with_external_weights/transpose/perm:output:0*
T0*'
_output_shapes
: ���������2)
'product_with_external_weights/transpose�
-product_with_external_weights/PartitionedCallPartitionedCall+product_with_external_weights/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_f_4980262/
-product_with_external_weights/PartitionedCall�
!product_with_external_weights/MulMul+product_with_external_weights/Tile:output:06product_with_external_weights/PartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2#
!product_with_external_weights/Mul�
.product_with_external_weights/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.product_with_external_weights/transpose_1/perm�
)product_with_external_weights/transpose_1	Transpose.product_with_external_weights/Reshape:output:07product_with_external_weights/transpose_1/perm:output:0*
T0*'
_output_shapes
: ���������2+
)product_with_external_weights/transpose_1�
/product_with_external_weights/PartitionedCall_1PartitionedCall-product_with_external_weights/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_g_49803621
/product_with_external_weights/PartitionedCall_1�
!product_with_external_weights/addAddV2%product_with_external_weights/Mul:z:08product_with_external_weights/PartitionedCall_1:output:0*
T0*+
_output_shapes
:��������� 2#
!product_with_external_weights/add�
4product_with_external_weights/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������26
4product_with_external_weights/Prod/reduction_indices�
"product_with_external_weights/ProdProd%product_with_external_weights/add:z:0=product_with_external_weights/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:��������� 2$
"product_with_external_weights/Prodp
NegNeg+product_with_external_weights/Prod:output:0*
T0*'
_output_shapes
:��������� 2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:��������� 2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�`
m
C__inference_encoder_layer_call_and_return_conditional_losses_498271
input_1
input_2
identity�
7linear_block_code_product_encoder_with_external_g/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @29
7linear_block_code_product_encoder_with_external_g/mul/y�
5linear_block_code_product_encoder_with_external_g/mulMulinput_1@linear_block_code_product_encoder_with_external_g/mul/y:output:0*
T0*'
_output_shapes
:���������27
5linear_block_code_product_encoder_with_external_g/mul�
7linear_block_code_product_encoder_with_external_g/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?29
7linear_block_code_product_encoder_with_external_g/sub/x�
5linear_block_code_product_encoder_with_external_g/subSub@linear_block_code_product_encoder_with_external_g/sub/x:output:09linear_block_code_product_encoder_with_external_g/mul:z:0*
T0*'
_output_shapes
:���������27
5linear_block_code_product_encoder_with_external_g/sub�
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/ShapeShape9linear_block_code_product_encoder_with_external_g/sub:z:0*
T0*
_output_shapes
:2W
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape�
clinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2e
clinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack�
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2g
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1�
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2�
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_sliceStridedSlice^linear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape:output:0llinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack:output:0nlinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1:output:0nlinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2_
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice�
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2a
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1�
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shapePackflinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice:output:0hlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2_
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape�
Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ReshapeReshapeinput_2flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape:output:0*
T0*'
_output_shapes
:��������� 2Y
Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape�
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2`
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim�
Zlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims
ExpandDims9linear_block_code_product_encoder_with_external_g/sub:z:0glinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2\
Zlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims�
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2W
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/stack�
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/TileTileclinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims:output:0^linear_block_code_product_encoder_with_external_g/product_with_external_weights/stack:output:0*
T0*+
_output_shapes
:���������2V
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile�
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2`
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm�
Ylinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose	Transpose`linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0glinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm:output:0*
T0*'
_output_shapes
: ���������2[
Ylinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose�
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCallPartitionedCall]linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_f_4980262a
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall�
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/MulMul]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile:output:0hlinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2U
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul�
`linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2b
`linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm�
[linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1	Transpose`linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0ilinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm:output:0*
T0*'
_output_shapes
: ���������2]
[linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1�
alinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1PartitionedCall_linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_g_4980362c
alinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1�
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/addAddV2Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul:z:0jlinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1:output:0*
T0*+
_output_shapes
:��������� 2U
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/add�
flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2h
flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices�
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ProdProdWlinear_block_code_product_encoder_with_external_g/product_with_external_weights/add:z:0olinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:��������� 2V
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod�
5linear_block_code_product_encoder_with_external_g/NegNeg]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod:output:0*
T0*'
_output_shapes
:��������� 27
5linear_block_code_product_encoder_with_external_g/Neg�
9linear_block_code_product_encoder_with_external_g/SigmoidSigmoid9linear_block_code_product_encoder_with_external_g/Neg:y:0*
T0*'
_output_shapes
:��������� 2;
9linear_block_code_product_encoder_with_external_g/Sigmoid�
*differentiable_bpsk_modulation_layer/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*differentiable_bpsk_modulation_layer/sub/y�
(differentiable_bpsk_modulation_layer/subSub=linear_block_code_product_encoder_with_external_g/Sigmoid:y:03differentiable_bpsk_modulation_layer/sub/y:output:0*
T0*'
_output_shapes
:��������� 2*
(differentiable_bpsk_modulation_layer/sub�
.differentiable_bpsk_modulation_layer/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.differentiable_bpsk_modulation_layer/Greater/y�
,differentiable_bpsk_modulation_layer/GreaterGreater,differentiable_bpsk_modulation_layer/sub:z:07differentiable_bpsk_modulation_layer/Greater/y:output:0*
T0*'
_output_shapes
:��������� 2.
,differentiable_bpsk_modulation_layer/Greater�
/differentiable_bpsk_modulation_layer/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  �?21
/differentiable_bpsk_modulation_layer/SelectV2/t�
/differentiable_bpsk_modulation_layer/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ��21
/differentiable_bpsk_modulation_layer/SelectV2/e�
-differentiable_bpsk_modulation_layer/SelectV2SelectV20differentiable_bpsk_modulation_layer/Greater:z:08differentiable_bpsk_modulation_layer/SelectV2/t:output:08differentiable_bpsk_modulation_layer/SelectV2/e:output:0*
T0*'
_output_shapes
:��������� 2/
-differentiable_bpsk_modulation_layer/SelectV2�
-differentiable_bpsk_modulation_layer/IdentityIdentity6differentiable_bpsk_modulation_layer/SelectV2:output:0*
T0*'
_output_shapes
:��������� 2/
-differentiable_bpsk_modulation_layer/Identity�
.differentiable_bpsk_modulation_layer/IdentityN	IdentityN6differentiable_bpsk_modulation_layer/SelectV2:output:0,differentiable_bpsk_modulation_layer/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-498261*:
_output_shapes(
&:��������� :��������� 20
.differentiable_bpsk_modulation_layer/IdentityN�
IdentityIdentity7differentiable_bpsk_modulation_layer/IdentityN:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:��������� 
!
_user_specified_name	input_2
�`
o
C__inference_encoder_layer_call_and_return_conditional_losses_498415
inputs_0
inputs_1
identity�
7linear_block_code_product_encoder_with_external_g/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @29
7linear_block_code_product_encoder_with_external_g/mul/y�
5linear_block_code_product_encoder_with_external_g/mulMulinputs_0@linear_block_code_product_encoder_with_external_g/mul/y:output:0*
T0*'
_output_shapes
:���������27
5linear_block_code_product_encoder_with_external_g/mul�
7linear_block_code_product_encoder_with_external_g/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?29
7linear_block_code_product_encoder_with_external_g/sub/x�
5linear_block_code_product_encoder_with_external_g/subSub@linear_block_code_product_encoder_with_external_g/sub/x:output:09linear_block_code_product_encoder_with_external_g/mul:z:0*
T0*'
_output_shapes
:���������27
5linear_block_code_product_encoder_with_external_g/sub�
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/ShapeShape9linear_block_code_product_encoder_with_external_g/sub:z:0*
T0*
_output_shapes
:2W
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape�
clinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2e
clinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack�
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2g
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1�
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2g
elinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2�
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_sliceStridedSlice^linear_block_code_product_encoder_with_external_g/product_with_external_weights/Shape:output:0llinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack:output:0nlinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_1:output:0nlinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2_
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice�
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2a
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1�
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shapePackflinear_block_code_product_encoder_with_external_g/product_with_external_weights/strided_slice:output:0hlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2_
]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape�
Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ReshapeReshapeinputs_1flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape/shape:output:0*
T0*'
_output_shapes
:��������� 2Y
Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape�
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2`
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim�
Zlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims
ExpandDims9linear_block_code_product_encoder_with_external_g/sub:z:0glinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2\
Zlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims�
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2W
Ulinear_block_code_product_encoder_with_external_g/product_with_external_weights/stack�
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/TileTileclinear_block_code_product_encoder_with_external_g/product_with_external_weights/ExpandDims:output:0^linear_block_code_product_encoder_with_external_g/product_with_external_weights/stack:output:0*
T0*+
_output_shapes
:���������2V
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile�
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2`
^linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm�
Ylinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose	Transpose`linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0glinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose/perm:output:0*
T0*'
_output_shapes
: ���������2[
Ylinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose�
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCallPartitionedCall]linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_f_4980262a
_linear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall�
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/MulMul]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Tile:output:0hlinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2U
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul�
`linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2b
`linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm�
[linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1	Transpose`linear_block_code_product_encoder_with_external_g/product_with_external_weights/Reshape:output:0ilinear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1/perm:output:0*
T0*'
_output_shapes
: ���������2]
[linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1�
alinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1PartitionedCall_linear_block_code_product_encoder_with_external_g/product_with_external_weights/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_g_4980362c
alinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1�
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/addAddV2Wlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Mul:z:0jlinear_block_code_product_encoder_with_external_g/product_with_external_weights/PartitionedCall_1:output:0*
T0*+
_output_shapes
:��������� 2U
Slinear_block_code_product_encoder_with_external_g/product_with_external_weights/add�
flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2h
flinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices�
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/ProdProdWlinear_block_code_product_encoder_with_external_g/product_with_external_weights/add:z:0olinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:��������� 2V
Tlinear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod�
5linear_block_code_product_encoder_with_external_g/NegNeg]linear_block_code_product_encoder_with_external_g/product_with_external_weights/Prod:output:0*
T0*'
_output_shapes
:��������� 27
5linear_block_code_product_encoder_with_external_g/Neg�
9linear_block_code_product_encoder_with_external_g/SigmoidSigmoid9linear_block_code_product_encoder_with_external_g/Neg:y:0*
T0*'
_output_shapes
:��������� 2;
9linear_block_code_product_encoder_with_external_g/Sigmoid�
*differentiable_bpsk_modulation_layer/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*differentiable_bpsk_modulation_layer/sub/y�
(differentiable_bpsk_modulation_layer/subSub=linear_block_code_product_encoder_with_external_g/Sigmoid:y:03differentiable_bpsk_modulation_layer/sub/y:output:0*
T0*'
_output_shapes
:��������� 2*
(differentiable_bpsk_modulation_layer/sub�
.differentiable_bpsk_modulation_layer/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.differentiable_bpsk_modulation_layer/Greater/y�
,differentiable_bpsk_modulation_layer/GreaterGreater,differentiable_bpsk_modulation_layer/sub:z:07differentiable_bpsk_modulation_layer/Greater/y:output:0*
T0*'
_output_shapes
:��������� 2.
,differentiable_bpsk_modulation_layer/Greater�
/differentiable_bpsk_modulation_layer/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  �?21
/differentiable_bpsk_modulation_layer/SelectV2/t�
/differentiable_bpsk_modulation_layer/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ��21
/differentiable_bpsk_modulation_layer/SelectV2/e�
-differentiable_bpsk_modulation_layer/SelectV2SelectV20differentiable_bpsk_modulation_layer/Greater:z:08differentiable_bpsk_modulation_layer/SelectV2/t:output:08differentiable_bpsk_modulation_layer/SelectV2/e:output:0*
T0*'
_output_shapes
:��������� 2/
-differentiable_bpsk_modulation_layer/SelectV2�
-differentiable_bpsk_modulation_layer/IdentityIdentity6differentiable_bpsk_modulation_layer/SelectV2:output:0*
T0*'
_output_shapes
:��������� 2/
-differentiable_bpsk_modulation_layer/Identity�
.differentiable_bpsk_modulation_layer/IdentityN	IdentityN6differentiable_bpsk_modulation_layer/SelectV2:output:0,differentiable_bpsk_modulation_layer/sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-498405*:
_output_shapes(
&:��������� :��������� 20
.differentiable_bpsk_modulation_layer/IdentityN�
IdentityIdentity7differentiable_bpsk_modulation_layer/IdentityN:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�/
�
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_498493
inputs_0
inputs_1
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/y]
mulMulinputs_0mul/y:output:0*
T0*'
_output_shapes
:���������2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:���������2
sub�
#product_with_external_weights/ShapeShapesub:z:0*
T0*
_output_shapes
:2%
#product_with_external_weights/Shape�
1product_with_external_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������23
1product_with_external_weights/strided_slice/stack�
3product_with_external_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3product_with_external_weights/strided_slice/stack_1�
3product_with_external_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3product_with_external_weights/strided_slice/stack_2�
+product_with_external_weights/strided_sliceStridedSlice,product_with_external_weights/Shape:output:0:product_with_external_weights/strided_slice/stack:output:0<product_with_external_weights/strided_slice/stack_1:output:0<product_with_external_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+product_with_external_weights/strided_slice�
-product_with_external_weights/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2/
-product_with_external_weights/Reshape/shape/1�
+product_with_external_weights/Reshape/shapePack4product_with_external_weights/strided_slice:output:06product_with_external_weights/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+product_with_external_weights/Reshape/shape�
%product_with_external_weights/ReshapeReshapeinputs_14product_with_external_weights/Reshape/shape:output:0*
T0*'
_output_shapes
:��������� 2'
%product_with_external_weights/Reshape�
,product_with_external_weights/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,product_with_external_weights/ExpandDims/dim�
(product_with_external_weights/ExpandDims
ExpandDimssub:z:05product_with_external_weights/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2*
(product_with_external_weights/ExpandDims�
#product_with_external_weights/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2%
#product_with_external_weights/stack�
"product_with_external_weights/TileTile1product_with_external_weights/ExpandDims:output:0,product_with_external_weights/stack:output:0*
T0*+
_output_shapes
:���������2$
"product_with_external_weights/Tile�
,product_with_external_weights/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2.
,product_with_external_weights/transpose/perm�
'product_with_external_weights/transpose	Transpose.product_with_external_weights/Reshape:output:05product_with_external_weights/transpose/perm:output:0*
T0*'
_output_shapes
: ���������2)
'product_with_external_weights/transpose�
-product_with_external_weights/PartitionedCallPartitionedCall+product_with_external_weights/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_f_4980262/
-product_with_external_weights/PartitionedCall�
!product_with_external_weights/MulMul+product_with_external_weights/Tile:output:06product_with_external_weights/PartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2#
!product_with_external_weights/Mul�
.product_with_external_weights/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.product_with_external_weights/transpose_1/perm�
)product_with_external_weights/transpose_1	Transpose.product_with_external_weights/Reshape:output:07product_with_external_weights/transpose_1/perm:output:0*
T0*'
_output_shapes
: ���������2+
)product_with_external_weights/transpose_1�
/product_with_external_weights/PartitionedCall_1PartitionedCall-product_with_external_weights/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_g_49803621
/product_with_external_weights/PartitionedCall_1�
!product_with_external_weights/addAddV2%product_with_external_weights/Mul:z:08product_with_external_weights/PartitionedCall_1:output:0*
T0*+
_output_shapes
:��������� 2#
!product_with_external_weights/add�
4product_with_external_weights/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������26
4product_with_external_weights/Prod/reduction_indices�
"product_with_external_weights/ProdProd%product_with_external_weights/add:z:0=product_with_external_weights/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:��������� 2$
"product_with_external_weights/Prodp
NegNeg+product_with_external_weights/Prod:output:0*
T0*'
_output_shapes
:��������� 2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:��������� 2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�

~
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_498520

inputs

identity_1S
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sub/y[
subSubinputssub/y:output:0*
T0*'
_output_shapes
:��������� 2
sub[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/yl
GreaterGreatersub:z:0Greater/y:output:0*
T0*'
_output_shapes
:��������� 2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  ��2

SelectV2/e�
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:��������� 2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:��������� 2

Identity�
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*,
_gradient_op_typeCustomGradient-498510*:
_output_shapes(
&:��������� :��������� 2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�/
�
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_498126

inputs
inputs_1
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/y[
mulMulinputsmul/y:output:0*
T0*'
_output_shapes
:���������2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:���������2
sub�
#product_with_external_weights/ShapeShapesub:z:0*
T0*
_output_shapes
:2%
#product_with_external_weights/Shape�
1product_with_external_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������23
1product_with_external_weights/strided_slice/stack�
3product_with_external_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3product_with_external_weights/strided_slice/stack_1�
3product_with_external_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3product_with_external_weights/strided_slice/stack_2�
+product_with_external_weights/strided_sliceStridedSlice,product_with_external_weights/Shape:output:0:product_with_external_weights/strided_slice/stack:output:0<product_with_external_weights/strided_slice/stack_1:output:0<product_with_external_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+product_with_external_weights/strided_slice�
-product_with_external_weights/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2/
-product_with_external_weights/Reshape/shape/1�
+product_with_external_weights/Reshape/shapePack4product_with_external_weights/strided_slice:output:06product_with_external_weights/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+product_with_external_weights/Reshape/shape�
%product_with_external_weights/ReshapeReshapeinputs_14product_with_external_weights/Reshape/shape:output:0*
T0*'
_output_shapes
:��������� 2'
%product_with_external_weights/Reshape�
,product_with_external_weights/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,product_with_external_weights/ExpandDims/dim�
(product_with_external_weights/ExpandDims
ExpandDimssub:z:05product_with_external_weights/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2*
(product_with_external_weights/ExpandDims�
#product_with_external_weights/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2%
#product_with_external_weights/stack�
"product_with_external_weights/TileTile1product_with_external_weights/ExpandDims:output:0,product_with_external_weights/stack:output:0*
T0*+
_output_shapes
:���������2$
"product_with_external_weights/Tile�
,product_with_external_weights/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2.
,product_with_external_weights/transpose/perm�
'product_with_external_weights/transpose	Transpose.product_with_external_weights/Reshape:output:05product_with_external_weights/transpose/perm:output:0*
T0*'
_output_shapes
: ���������2)
'product_with_external_weights/transpose�
-product_with_external_weights/PartitionedCallPartitionedCall+product_with_external_weights/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_f_4980262/
-product_with_external_weights/PartitionedCall�
!product_with_external_weights/MulMul+product_with_external_weights/Tile:output:06product_with_external_weights/PartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2#
!product_with_external_weights/Mul�
.product_with_external_weights/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.product_with_external_weights/transpose_1/perm�
)product_with_external_weights/transpose_1	Transpose.product_with_external_weights/Reshape:output:07product_with_external_weights/transpose_1/perm:output:0*
T0*'
_output_shapes
: ���������2+
)product_with_external_weights/transpose_1�
/product_with_external_weights/PartitionedCall_1PartitionedCall-product_with_external_weights/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_g_49803621
/product_with_external_weights/PartitionedCall_1�
!product_with_external_weights/addAddV2%product_with_external_weights/Mul:z:08product_with_external_weights/PartitionedCall_1:output:0*
T0*+
_output_shapes
:��������� 2#
!product_with_external_weights/add�
4product_with_external_weights/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������26
4product_with_external_weights/Prod/reduction_indices�
"product_with_external_weights/ProdProd%product_with_external_weights/add:z:0=product_with_external_weights/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:��������� 2$
"product_with_external_weights/Prodp
NegNeg+product_with_external_weights/Prod:output:0*
T0*'
_output_shapes
:��������� 2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:��������� 2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�/
�
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_498460
inputs_0
inputs_1
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
mul/y]
mulMulinputs_0mul/y:output:0*
T0*'
_output_shapes
:���������2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:���������2
sub�
#product_with_external_weights/ShapeShapesub:z:0*
T0*
_output_shapes
:2%
#product_with_external_weights/Shape�
1product_with_external_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������23
1product_with_external_weights/strided_slice/stack�
3product_with_external_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3product_with_external_weights/strided_slice/stack_1�
3product_with_external_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3product_with_external_weights/strided_slice/stack_2�
+product_with_external_weights/strided_sliceStridedSlice,product_with_external_weights/Shape:output:0:product_with_external_weights/strided_slice/stack:output:0<product_with_external_weights/strided_slice/stack_1:output:0<product_with_external_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+product_with_external_weights/strided_slice�
-product_with_external_weights/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2/
-product_with_external_weights/Reshape/shape/1�
+product_with_external_weights/Reshape/shapePack4product_with_external_weights/strided_slice:output:06product_with_external_weights/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+product_with_external_weights/Reshape/shape�
%product_with_external_weights/ReshapeReshapeinputs_14product_with_external_weights/Reshape/shape:output:0*
T0*'
_output_shapes
:��������� 2'
%product_with_external_weights/Reshape�
,product_with_external_weights/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,product_with_external_weights/ExpandDims/dim�
(product_with_external_weights/ExpandDims
ExpandDimssub:z:05product_with_external_weights/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2*
(product_with_external_weights/ExpandDims�
#product_with_external_weights/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2%
#product_with_external_weights/stack�
"product_with_external_weights/TileTile1product_with_external_weights/ExpandDims:output:0,product_with_external_weights/stack:output:0*
T0*+
_output_shapes
:���������2$
"product_with_external_weights/Tile�
,product_with_external_weights/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2.
,product_with_external_weights/transpose/perm�
'product_with_external_weights/transpose	Transpose.product_with_external_weights/Reshape:output:05product_with_external_weights/transpose/perm:output:0*
T0*'
_output_shapes
: ���������2)
'product_with_external_weights/transpose�
-product_with_external_weights/PartitionedCallPartitionedCall+product_with_external_weights/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_f_4980262/
-product_with_external_weights/PartitionedCall�
!product_with_external_weights/MulMul+product_with_external_weights/Tile:output:06product_with_external_weights/PartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2#
!product_with_external_weights/Mul�
.product_with_external_weights/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.product_with_external_weights/transpose_1/perm�
)product_with_external_weights/transpose_1	Transpose.product_with_external_weights/Reshape:output:07product_with_external_weights/transpose_1/perm:output:0*
T0*'
_output_shapes
: ���������2+
)product_with_external_weights/transpose_1�
/product_with_external_weights/PartitionedCall_1PartitionedCall-product_with_external_weights/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_g_49803621
/product_with_external_weights/PartitionedCall_1�
!product_with_external_weights/addAddV2%product_with_external_weights/Mul:z:08product_with_external_weights/PartitionedCall_1:output:0*
T0*+
_output_shapes
:��������� 2#
!product_with_external_weights/add�
4product_with_external_weights/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������26
4product_with_external_weights/Prod/reduction_indices�
"product_with_external_weights/ProdProd%product_with_external_weights/add:z:0=product_with_external_weights/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:��������� 2$
"product_with_external_weights/Prodp
NegNeg+product_with_external_weights/Prod:output:0*
T0*'
_output_shapes
:��������� 2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:��������� 2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:���������:��������� :Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
+
__inference_g_498565
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xV
subSubsub/x:output:0w*
T0*'
_output_shapes
: ���������2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
: ���������2

Identity"
identityIdentity:output:0*&
_input_shapes
: ���������:J F
'
_output_shapes
: ���������

_user_specified_namew"�J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������
;
input_20
serving_default_input_2:0��������� 4
output_1(
PartitionedCall:0��������� tensorflow/serving/predict:�]
�
	lbcpe

modulation
trainable_variables
	variables
regularization_losses
	keras_api

signatures
)__call__
*_default_save_signature
*+&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "Encoder", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Encoder"}}
�
product
	trainable_variables

	variables
regularization_losses
	keras_api
,__call__
*-&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LinearBlockCodeProductEncoderWithExternalG", "name": "linear_block_code_product_encoder_with_external_g", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
trainable_variables
	variables
regularization_losses
	keras_api
.__call__
*/&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "DifferentiableBPSKModulationLayer", "name": "differentiable_bpsk_modulation_layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
layer_regularization_losses
trainable_variables
metrics
	variables
regularization_losses
layer_metrics
)__call__
*_default_save_signature
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
,
0serving_default"
signature_map
�
trainable_variables
	variables
regularization_losses
	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3f
4g"�
_tf_keras_layer�{"class_name": "ProductWithExternalWeights", "name": "product_with_external_weights", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": [{"class_name": "TensorShape", "items": [64, 16]}, {"class_name": "TensorShape", "items": [16, 32]}]}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
layer_regularization_losses
	trainable_variables
metrics

	variables
regularization_losses
layer_metrics
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

 layers
!layer_regularization_losses
trainable_variables
"metrics
	variables
regularization_losses
#layer_metrics
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
$non_trainable_variables

%layers
&layer_regularization_losses
trainable_variables
'metrics
	variables
regularization_losses
(layer_metrics
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�2�
(__inference_encoder_layer_call_fn_498327
(__inference_encoder_layer_call_fn_498421
(__inference_encoder_layer_call_fn_498427
(__inference_encoder_layer_call_fn_498321�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_498055�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *N�K
I�F
!�
input_1���������
!�
input_2��������� 
�2�
C__inference_encoder_layer_call_and_return_conditional_losses_498271
C__inference_encoder_layer_call_and_return_conditional_losses_498415
C__inference_encoder_layer_call_and_return_conditional_losses_498315
C__inference_encoder_layer_call_and_return_conditional_losses_498371�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_linear_block_code_product_encoder_with_external_g_layer_call_fn_498499
R__inference_linear_block_code_product_encoder_with_external_g_layer_call_fn_498505�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_498460
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_498493�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_differentiable_bpsk_modulation_layer_layer_call_fn_498540
E__inference_differentiable_bpsk_modulation_layer_layer_call_fn_498545�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_498520
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_498535�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
$__inference_signature_wrapper_498227input_1input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_f_498549
__inference_f_498553�
���
FullArgSpec
args�
jself
jw
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_g_498559
__inference_g_498565�
���
FullArgSpec
args�
jself
jw
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_498055�X�U
N�K
I�F
!�
input_1���������
!�
input_2��������� 
� "3�0
.
output_1"�
output_1��������� �
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_498520\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� �
`__inference_differentiable_bpsk_modulation_layer_layer_call_and_return_conditional_losses_498535\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
E__inference_differentiable_bpsk_modulation_layer_layer_call_fn_498540O3�0
)�&
 �
inputs��������� 
p
� "���������� �
E__inference_differentiable_bpsk_modulation_layer_layer_call_fn_498545O3�0
)�&
 �
inputs��������� 
p 
� "���������� �
C__inference_encoder_layer_call_and_return_conditional_losses_498271�\�Y
R�O
I�F
!�
input_1���������
!�
input_2��������� 
p
� "%�"
�
0��������� 
� �
C__inference_encoder_layer_call_and_return_conditional_losses_498315�\�Y
R�O
I�F
!�
input_1���������
!�
input_2��������� 
p 
� "%�"
�
0��������� 
� �
C__inference_encoder_layer_call_and_return_conditional_losses_498371�^�[
T�Q
K�H
"�
inputs/0���������
"�
inputs/1��������� 
p
� "%�"
�
0��������� 
� �
C__inference_encoder_layer_call_and_return_conditional_losses_498415�^�[
T�Q
K�H
"�
inputs/0���������
"�
inputs/1��������� 
p 
� "%�"
�
0��������� 
� �
(__inference_encoder_layer_call_fn_498321x\�Y
R�O
I�F
!�
input_1���������
!�
input_2��������� 
p
� "���������� �
(__inference_encoder_layer_call_fn_498327x\�Y
R�O
I�F
!�
input_1���������
!�
input_2��������� 
p 
� "���������� �
(__inference_encoder_layer_call_fn_498421z^�[
T�Q
K�H
"�
inputs/0���������
"�
inputs/1��������� 
p
� "���������� �
(__inference_encoder_layer_call_fn_498427z^�[
T�Q
K�H
"�
inputs/0���������
"�
inputs/1��������� 
p 
� "���������� L
__inference_f_4985494!�
�
�
w 
� "� ^
__inference_f_498553F*�'
 �
�
w ���������
� "� ���������L
__inference_g_4985594!�
�
�
w 
� "� ^
__inference_g_498565F*�'
 �
�
w ���������
� "� ����������
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_498460�^�[
T�Q
K�H
"�
inputs/0���������
"�
inputs/1��������� 
p
� "%�"
�
0��������� 
� �
m__inference_linear_block_code_product_encoder_with_external_g_layer_call_and_return_conditional_losses_498493�^�[
T�Q
K�H
"�
inputs/0���������
"�
inputs/1��������� 
p 
� "%�"
�
0��������� 
� �
R__inference_linear_block_code_product_encoder_with_external_g_layer_call_fn_498499z^�[
T�Q
K�H
"�
inputs/0���������
"�
inputs/1��������� 
p
� "���������� �
R__inference_linear_block_code_product_encoder_with_external_g_layer_call_fn_498505z^�[
T�Q
K�H
"�
inputs/0���������
"�
inputs/1��������� 
p 
� "���������� �
$__inference_signature_wrapper_498227�i�f
� 
_�\
,
input_1!�
input_1���������
,
input_2!�
input_2��������� "3�0
.
output_1"�
output_1��������� 