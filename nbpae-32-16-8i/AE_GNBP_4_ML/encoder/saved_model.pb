õ
„
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
³
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

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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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
2	"serve*2.4.32unknown8¦’

NoOpNoOp
ų	
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*³	
value©	B¦	 B	
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
­
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
­
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
­
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
­
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
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
z
serving_default_input_2Placeholder*'
_output_shapes
:’’’’’’’’’ *
dtype0*
shape:’’’’’’’’’ 
¾
PartitionedCallPartitionedCallserving_default_input_1serving_default_input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_7060898
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_7061260

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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_7061270Ėė

S
)__inference_encoder_layer_call_fn_7060992
input_1
input_2
identityŠ
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_70608742
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’ 
!
_user_specified_name	input_2
ūk
L
"__inference__wrapped_model_7060726
input_1
input_2
identityĶ
Bencoder/linear_block_code_product_encoder_with_external_g_16/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2D
Bencoder/linear_block_code_product_encoder_with_external_g_16/mul/y
@encoder/linear_block_code_product_encoder_with_external_g_16/mulMulinput_1Kencoder/linear_block_code_product_encoder_with_external_g_16/mul/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2B
@encoder/linear_block_code_product_encoder_with_external_g_16/mulĶ
Bencoder/linear_block_code_product_encoder_with_external_g_16/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
Bencoder/linear_block_code_product_encoder_with_external_g_16/sub/xŠ
@encoder/linear_block_code_product_encoder_with_external_g_16/subSubKencoder/linear_block_code_product_encoder_with_external_g_16/sub/x:output:0Dencoder/linear_block_code_product_encoder_with_external_g_16/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2B
@encoder/linear_block_code_product_encoder_with_external_g_16/sub¾
cencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ShapeShapeDencoder/linear_block_code_product_encoder_with_external_g_16/sub:z:0*
T0*
_output_shapes
:2e
cencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape¹
qencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2s
qencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack“
sencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2u
sencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1“
sencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2u
sencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2
kencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_sliceStridedSlicelencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape:output:0zencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack:output:0|encoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1:output:0|encoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2m
kencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice 
mencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2o
mencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1ž
kencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shapePacktencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice:output:0vencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2m
kencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape
eencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ReshapeReshapeinput_2tencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2g
eencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape
lencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2n
lencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimÕ
hencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
ExpandDimsDencoder/linear_block_code_product_encoder_with_external_g_16/sub:z:0uencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2j
hencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
cencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2e
cencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackē
bencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/TileTileqencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims:output:0lencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stack:output:0*
T0*+
_output_shapes
:’’’’’’’’’2d
bencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile­
lencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2n
lencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permų
gencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose	Transposenencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0uencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2i
gencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose·
mencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallPartitionedCallkencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_f_70606972o
mencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallč
aencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/MulMulkencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile:output:0vencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2c
aencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul±
nencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2p
nencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permž
iencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1	Transposenencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0wencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2k
iencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1½
oencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1PartitionedCallmencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_g_70607072q
oencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1ę
aencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/addAddV2eencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul:z:0xencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2c
aencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add·
tencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2v
tencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesč
bencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ProdProdeencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add:z:0}encoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2d
bencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ProdŖ
@encoder/linear_block_code_product_encoder_with_external_g_16/NegNegkencoder/linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2B
@encoder/linear_block_code_product_encoder_with_external_g_16/Neg
Dencoder/linear_block_code_product_encoder_with_external_g_16/SigmoidSigmoidDencoder/linear_block_code_product_encoder_with_external_g_16/Neg:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2F
Dencoder/linear_block_code_product_encoder_with_external_g_16/Sigmoid³
5encoder/differentiable_bpsk_modulation_layer_16/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?27
5encoder/differentiable_bpsk_modulation_layer_16/sub/y­
3encoder/differentiable_bpsk_modulation_layer_16/subSubHencoder/linear_block_code_product_encoder_with_external_g_16/Sigmoid:y:0>encoder/differentiable_bpsk_modulation_layer_16/sub/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 25
3encoder/differentiable_bpsk_modulation_layer_16/sub»
9encoder/differentiable_bpsk_modulation_layer_16/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9encoder/differentiable_bpsk_modulation_layer_16/Greater/y¬
7encoder/differentiable_bpsk_modulation_layer_16/GreaterGreater7encoder/differentiable_bpsk_modulation_layer_16/sub:z:0Bencoder/differentiable_bpsk_modulation_layer_16/Greater/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 29
7encoder/differentiable_bpsk_modulation_layer_16/Greater½
:encoder/differentiable_bpsk_modulation_layer_16/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:encoder/differentiable_bpsk_modulation_layer_16/SelectV2/t½
:encoder/differentiable_bpsk_modulation_layer_16/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  æ2<
:encoder/differentiable_bpsk_modulation_layer_16/SelectV2/eł
8encoder/differentiable_bpsk_modulation_layer_16/SelectV2SelectV2;encoder/differentiable_bpsk_modulation_layer_16/Greater:z:0Cencoder/differentiable_bpsk_modulation_layer_16/SelectV2/t:output:0Cencoder/differentiable_bpsk_modulation_layer_16/SelectV2/e:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2:
8encoder/differentiable_bpsk_modulation_layer_16/SelectV2õ
8encoder/differentiable_bpsk_modulation_layer_16/IdentityIdentityAencoder/differentiable_bpsk_modulation_layer_16/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2:
8encoder/differentiable_bpsk_modulation_layer_16/Identity÷
9encoder/differentiable_bpsk_modulation_layer_16/IdentityN	IdentityNAencoder/differentiable_bpsk_modulation_layer_16/SelectV2:output:07encoder/differentiable_bpsk_modulation_layer_16/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-7060716*:
_output_shapes(
&:’’’’’’’’’ :’’’’’’’’’ 2;
9encoder/differentiable_bpsk_modulation_layer_16/IdentityN
IdentityIdentityBencoder/differentiable_bpsk_modulation_layer_16/IdentityN:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’ 
!
_user_specified_name	input_2

U
)__inference_encoder_layer_call_fn_7061092
inputs_0
inputs_1
identityŅ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_70608742
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’ 
"
_user_specified_name
inputs/1
e
n
D__inference_encoder_layer_call_and_return_conditional_losses_7060986
input_1
input_2
identity½
:linear_block_code_product_encoder_with_external_g_16/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:linear_block_code_product_encoder_with_external_g_16/mul/yū
8linear_block_code_product_encoder_with_external_g_16/mulMulinput_1Clinear_block_code_product_encoder_with_external_g_16/mul/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2:
8linear_block_code_product_encoder_with_external_g_16/mul½
:linear_block_code_product_encoder_with_external_g_16/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:linear_block_code_product_encoder_with_external_g_16/sub/x°
8linear_block_code_product_encoder_with_external_g_16/subSubClinear_block_code_product_encoder_with_external_g_16/sub/x:output:0<linear_block_code_product_encoder_with_external_g_16/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2:
8linear_block_code_product_encoder_with_external_g_16/sub¦
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ShapeShape<linear_block_code_product_encoder_with_external_g_16/sub:z:0*
T0*
_output_shapes
:2]
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape©
ilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2k
ilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack¤
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2m
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1¤
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2m
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2ę
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_sliceStridedSlicedlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape:output:0rlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack:output:0tlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1:output:0tlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2e
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2g
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1Ž
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shapePackllinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice:output:0nlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2e
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shapeņ
]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ReshapeReshapeinput_2llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2_
]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2f
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimµ
`linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
ExpandDims<linear_block_code_product_encoder_with_external_g_16/sub:z:0mlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2b
`linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2]
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackĒ
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/TileTileilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims:output:0dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stack:output:0*
T0*+
_output_shapes
:’’’’’’’’’2\
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permŲ
_linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose	Transposeflinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2a
_linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallPartitionedCallclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_f_70606972g
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallČ
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/MulMulclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile:output:0nlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2[
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul”
flinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2h
flinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permŽ
alinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1	Transposeflinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0olinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2c
alinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1„
glinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1PartitionedCallelinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_g_70607072i
glinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1Ę
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/addAddV2]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul:z:0plinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2[
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add§
llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2n
llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesČ
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ProdProd]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add:z:0ulinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2\
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod
8linear_block_code_product_encoder_with_external_g_16/NegNegclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2:
8linear_block_code_product_encoder_with_external_g_16/Neg÷
<linear_block_code_product_encoder_with_external_g_16/SigmoidSigmoid<linear_block_code_product_encoder_with_external_g_16/Neg:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2>
<linear_block_code_product_encoder_with_external_g_16/Sigmoid£
-differentiable_bpsk_modulation_layer_16/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-differentiable_bpsk_modulation_layer_16/sub/y
+differentiable_bpsk_modulation_layer_16/subSub@linear_block_code_product_encoder_with_external_g_16/Sigmoid:y:06differentiable_bpsk_modulation_layer_16/sub/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2-
+differentiable_bpsk_modulation_layer_16/sub«
1differentiable_bpsk_modulation_layer_16/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1differentiable_bpsk_modulation_layer_16/Greater/y
/differentiable_bpsk_modulation_layer_16/GreaterGreater/differentiable_bpsk_modulation_layer_16/sub:z:0:differentiable_bpsk_modulation_layer_16/Greater/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 21
/differentiable_bpsk_modulation_layer_16/Greater­
2differentiable_bpsk_modulation_layer_16/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2differentiable_bpsk_modulation_layer_16/SelectV2/t­
2differentiable_bpsk_modulation_layer_16/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  æ24
2differentiable_bpsk_modulation_layer_16/SelectV2/eŃ
0differentiable_bpsk_modulation_layer_16/SelectV2SelectV23differentiable_bpsk_modulation_layer_16/Greater:z:0;differentiable_bpsk_modulation_layer_16/SelectV2/t:output:0;differentiable_bpsk_modulation_layer_16/SelectV2/e:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 22
0differentiable_bpsk_modulation_layer_16/SelectV2Ż
0differentiable_bpsk_modulation_layer_16/IdentityIdentity9differentiable_bpsk_modulation_layer_16/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 22
0differentiable_bpsk_modulation_layer_16/Identity×
1differentiable_bpsk_modulation_layer_16/IdentityN	IdentityN9differentiable_bpsk_modulation_layer_16/SelectV2:output:0/differentiable_bpsk_modulation_layer_16/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-7060976*:
_output_shapes(
&:’’’’’’’’’ :’’’’’’’’’ 23
1differentiable_bpsk_modulation_layer_16/IdentityN
IdentityIdentity:differentiable_bpsk_modulation_layer_16/IdentityN:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’ 
!
_user_specified_name	input_2
ģ
n
D__inference_encoder_layer_call_and_return_conditional_losses_7060887

inputs
inputs_1
identityē
Dlinear_block_code_product_encoder_with_external_g_16/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *z
fuRs
q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_70607972F
Dlinear_block_code_product_encoder_with_external_g_16/PartitionedCallü
7differentiable_bpsk_modulation_layer_16/PartitionedCallPartitionedCallMlinear_block_code_product_encoder_with_external_g_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *m
fhRf
d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_706084229
7differentiable_bpsk_modulation_layer_16/PartitionedCall
IdentityIdentity@differentiable_bpsk_modulation_layer_16/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

,
__inference_f_7060697
w
identityU
IdentityIdentityw*
T0*'
_output_shapes
: ’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
: ’’’’’’’’’:J F
'
_output_shapes
: ’’’’’’’’’

_user_specified_namew
ż
,
__inference_f_7061220
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
Ö
m
 __inference__traced_save_7061260
file_prefix
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slicesŗ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
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
1

q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_7060764

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
:’’’’’’’’’2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
sub
&product_with_external_weights_16/ShapeShapesub:z:0*
T0*
_output_shapes
:2(
&product_with_external_weights_16/Shapeæ
4product_with_external_weights_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’26
4product_with_external_weights_16/strided_slice/stackŗ
6product_with_external_weights_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6product_with_external_weights_16/strided_slice/stack_1ŗ
6product_with_external_weights_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6product_with_external_weights_16/strided_slice/stack_2Ø
.product_with_external_weights_16/strided_sliceStridedSlice/product_with_external_weights_16/Shape:output:0=product_with_external_weights_16/strided_slice/stack:output:0?product_with_external_weights_16/strided_slice/stack_1:output:0?product_with_external_weights_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.product_with_external_weights_16/strided_slice¦
0product_with_external_weights_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 22
0product_with_external_weights_16/Reshape/shape/1
.product_with_external_weights_16/Reshape/shapePack7product_with_external_weights_16/strided_slice:output:09product_with_external_weights_16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.product_with_external_weights_16/Reshape/shapeŌ
(product_with_external_weights_16/ReshapeReshapeinputs_17product_with_external_weights_16/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2*
(product_with_external_weights_16/Reshape¤
/product_with_external_weights_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/product_with_external_weights_16/ExpandDims/dimį
+product_with_external_weights_16/ExpandDims
ExpandDimssub:z:08product_with_external_weights_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2-
+product_with_external_weights_16/ExpandDims„
&product_with_external_weights_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2(
&product_with_external_weights_16/stackó
%product_with_external_weights_16/TileTile4product_with_external_weights_16/ExpandDims:output:0/product_with_external_weights_16/stack:output:0*
T0*+
_output_shapes
:’’’’’’’’’2'
%product_with_external_weights_16/Tile³
/product_with_external_weights_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/product_with_external_weights_16/transpose/perm
*product_with_external_weights_16/transpose	Transpose1product_with_external_weights_16/Reshape:output:08product_with_external_weights_16/transpose/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2,
*product_with_external_weights_16/transpose
0product_with_external_weights_16/PartitionedCallPartitionedCall.product_with_external_weights_16/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_f_706069722
0product_with_external_weights_16/PartitionedCallō
$product_with_external_weights_16/MulMul.product_with_external_weights_16/Tile:output:09product_with_external_weights_16/PartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$product_with_external_weights_16/Mul·
1product_with_external_weights_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       23
1product_with_external_weights_16/transpose_1/perm
,product_with_external_weights_16/transpose_1	Transpose1product_with_external_weights_16/Reshape:output:0:product_with_external_weights_16/transpose_1/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2.
,product_with_external_weights_16/transpose_1
2product_with_external_weights_16/PartitionedCall_1PartitionedCall0product_with_external_weights_16/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_g_706070724
2product_with_external_weights_16/PartitionedCall_1ņ
$product_with_external_weights_16/addAddV2(product_with_external_weights_16/Mul:z:0;product_with_external_weights_16/PartitionedCall_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$product_with_external_weights_16/add½
7product_with_external_weights_16/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’29
7product_with_external_weights_16/Prod/reduction_indicesō
%product_with_external_weights_16/ProdProd(product_with_external_weights_16/add:z:0@product_with_external_weights_16/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%product_with_external_weights_16/Prods
NegNeg.product_with_external_weights_16/Prod:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Õ
e
I__inference_differentiable_bpsk_modulation_layer_16_layer_call_fn_7061211

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *m
fhRf
d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_70608272
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
õ


d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_7061191

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
:’’’’’’’’’ 2
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
:’’’’’’’’’ 2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  æ2

SelectV2/e
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity·
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-7061181*:
_output_shapes(
&:’’’’’’’’’ :’’’’’’’’’ 2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

,
__inference_f_7061224
w
identityU
IdentityIdentityw*
T0*'
_output_shapes
: ’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
: ’’’’’’’’’:J F
'
_output_shapes
: ’’’’’’’’’

_user_specified_namew

U
)__inference_encoder_layer_call_fn_7061098
inputs_0
inputs_1
identityŅ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_70608872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’ 
"
_user_specified_name
inputs/1
ī
O
%__inference_signature_wrapper_7060898
input_1
input_2
identity®
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_70607262
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’ 
!
_user_specified_name	input_2
õ

V__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_fn_7061176
inputs_0
inputs_1
identity’
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *z
fuRs
q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_70607972
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’ 
"
_user_specified_name
inputs/1
Õ
e
I__inference_differentiable_bpsk_modulation_layer_16_layer_call_fn_7061216

inputs
identityå
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *m
fhRf
d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_70608422
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
Ė
,
__inference_g_7060707
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xV
subSubsub/x:output:0w*
T0*'
_output_shapes
: ’’’’’’’’’2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
: ’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
: ’’’’’’’’’:J F
'
_output_shapes
: ’’’’’’’’’

_user_specified_namew
e
n
D__inference_encoder_layer_call_and_return_conditional_losses_7060942
input_1
input_2
identity½
:linear_block_code_product_encoder_with_external_g_16/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:linear_block_code_product_encoder_with_external_g_16/mul/yū
8linear_block_code_product_encoder_with_external_g_16/mulMulinput_1Clinear_block_code_product_encoder_with_external_g_16/mul/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2:
8linear_block_code_product_encoder_with_external_g_16/mul½
:linear_block_code_product_encoder_with_external_g_16/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:linear_block_code_product_encoder_with_external_g_16/sub/x°
8linear_block_code_product_encoder_with_external_g_16/subSubClinear_block_code_product_encoder_with_external_g_16/sub/x:output:0<linear_block_code_product_encoder_with_external_g_16/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2:
8linear_block_code_product_encoder_with_external_g_16/sub¦
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ShapeShape<linear_block_code_product_encoder_with_external_g_16/sub:z:0*
T0*
_output_shapes
:2]
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape©
ilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2k
ilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack¤
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2m
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1¤
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2m
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2ę
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_sliceStridedSlicedlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape:output:0rlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack:output:0tlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1:output:0tlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2e
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2g
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1Ž
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shapePackllinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice:output:0nlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2e
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shapeņ
]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ReshapeReshapeinput_2llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2_
]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2f
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimµ
`linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
ExpandDims<linear_block_code_product_encoder_with_external_g_16/sub:z:0mlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2b
`linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2]
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackĒ
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/TileTileilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims:output:0dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stack:output:0*
T0*+
_output_shapes
:’’’’’’’’’2\
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permŲ
_linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose	Transposeflinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2a
_linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallPartitionedCallclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_f_70606972g
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallČ
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/MulMulclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile:output:0nlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2[
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul”
flinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2h
flinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permŽ
alinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1	Transposeflinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0olinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2c
alinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1„
glinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1PartitionedCallelinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_g_70607072i
glinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1Ę
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/addAddV2]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul:z:0plinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2[
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add§
llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2n
llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesČ
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ProdProd]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add:z:0ulinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2\
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod
8linear_block_code_product_encoder_with_external_g_16/NegNegclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2:
8linear_block_code_product_encoder_with_external_g_16/Neg÷
<linear_block_code_product_encoder_with_external_g_16/SigmoidSigmoid<linear_block_code_product_encoder_with_external_g_16/Neg:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2>
<linear_block_code_product_encoder_with_external_g_16/Sigmoid£
-differentiable_bpsk_modulation_layer_16/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-differentiable_bpsk_modulation_layer_16/sub/y
+differentiable_bpsk_modulation_layer_16/subSub@linear_block_code_product_encoder_with_external_g_16/Sigmoid:y:06differentiable_bpsk_modulation_layer_16/sub/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2-
+differentiable_bpsk_modulation_layer_16/sub«
1differentiable_bpsk_modulation_layer_16/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1differentiable_bpsk_modulation_layer_16/Greater/y
/differentiable_bpsk_modulation_layer_16/GreaterGreater/differentiable_bpsk_modulation_layer_16/sub:z:0:differentiable_bpsk_modulation_layer_16/Greater/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 21
/differentiable_bpsk_modulation_layer_16/Greater­
2differentiable_bpsk_modulation_layer_16/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2differentiable_bpsk_modulation_layer_16/SelectV2/t­
2differentiable_bpsk_modulation_layer_16/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  æ24
2differentiable_bpsk_modulation_layer_16/SelectV2/eŃ
0differentiable_bpsk_modulation_layer_16/SelectV2SelectV23differentiable_bpsk_modulation_layer_16/Greater:z:0;differentiable_bpsk_modulation_layer_16/SelectV2/t:output:0;differentiable_bpsk_modulation_layer_16/SelectV2/e:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 22
0differentiable_bpsk_modulation_layer_16/SelectV2Ż
0differentiable_bpsk_modulation_layer_16/IdentityIdentity9differentiable_bpsk_modulation_layer_16/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 22
0differentiable_bpsk_modulation_layer_16/Identity×
1differentiable_bpsk_modulation_layer_16/IdentityN	IdentityN9differentiable_bpsk_modulation_layer_16/SelectV2:output:0/differentiable_bpsk_modulation_layer_16/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-7060932*:
_output_shapes(
&:’’’’’’’’’ :’’’’’’’’’ 23
1differentiable_bpsk_modulation_layer_16/IdentityN
IdentityIdentity:differentiable_bpsk_modulation_layer_16/IdentityN:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’ 
!
_user_specified_name	input_2
e
p
D__inference_encoder_layer_call_and_return_conditional_losses_7061086
inputs_0
inputs_1
identity½
:linear_block_code_product_encoder_with_external_g_16/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:linear_block_code_product_encoder_with_external_g_16/mul/yü
8linear_block_code_product_encoder_with_external_g_16/mulMulinputs_0Clinear_block_code_product_encoder_with_external_g_16/mul/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2:
8linear_block_code_product_encoder_with_external_g_16/mul½
:linear_block_code_product_encoder_with_external_g_16/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:linear_block_code_product_encoder_with_external_g_16/sub/x°
8linear_block_code_product_encoder_with_external_g_16/subSubClinear_block_code_product_encoder_with_external_g_16/sub/x:output:0<linear_block_code_product_encoder_with_external_g_16/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2:
8linear_block_code_product_encoder_with_external_g_16/sub¦
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ShapeShape<linear_block_code_product_encoder_with_external_g_16/sub:z:0*
T0*
_output_shapes
:2]
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape©
ilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2k
ilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack¤
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2m
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1¤
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2m
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2ę
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_sliceStridedSlicedlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape:output:0rlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack:output:0tlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1:output:0tlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2e
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2g
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1Ž
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shapePackllinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice:output:0nlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2e
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shapeó
]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ReshapeReshapeinputs_1llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2_
]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2f
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimµ
`linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
ExpandDims<linear_block_code_product_encoder_with_external_g_16/sub:z:0mlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2b
`linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2]
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackĒ
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/TileTileilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims:output:0dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stack:output:0*
T0*+
_output_shapes
:’’’’’’’’’2\
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permŲ
_linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose	Transposeflinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2a
_linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallPartitionedCallclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_f_70606972g
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallČ
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/MulMulclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile:output:0nlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2[
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul”
flinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2h
flinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permŽ
alinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1	Transposeflinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0olinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2c
alinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1„
glinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1PartitionedCallelinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_g_70607072i
glinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1Ę
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/addAddV2]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul:z:0plinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2[
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add§
llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2n
llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesČ
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ProdProd]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add:z:0ulinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2\
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod
8linear_block_code_product_encoder_with_external_g_16/NegNegclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2:
8linear_block_code_product_encoder_with_external_g_16/Neg÷
<linear_block_code_product_encoder_with_external_g_16/SigmoidSigmoid<linear_block_code_product_encoder_with_external_g_16/Neg:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2>
<linear_block_code_product_encoder_with_external_g_16/Sigmoid£
-differentiable_bpsk_modulation_layer_16/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-differentiable_bpsk_modulation_layer_16/sub/y
+differentiable_bpsk_modulation_layer_16/subSub@linear_block_code_product_encoder_with_external_g_16/Sigmoid:y:06differentiable_bpsk_modulation_layer_16/sub/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2-
+differentiable_bpsk_modulation_layer_16/sub«
1differentiable_bpsk_modulation_layer_16/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1differentiable_bpsk_modulation_layer_16/Greater/y
/differentiable_bpsk_modulation_layer_16/GreaterGreater/differentiable_bpsk_modulation_layer_16/sub:z:0:differentiable_bpsk_modulation_layer_16/Greater/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 21
/differentiable_bpsk_modulation_layer_16/Greater­
2differentiable_bpsk_modulation_layer_16/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2differentiable_bpsk_modulation_layer_16/SelectV2/t­
2differentiable_bpsk_modulation_layer_16/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  æ24
2differentiable_bpsk_modulation_layer_16/SelectV2/eŃ
0differentiable_bpsk_modulation_layer_16/SelectV2SelectV23differentiable_bpsk_modulation_layer_16/Greater:z:0;differentiable_bpsk_modulation_layer_16/SelectV2/t:output:0;differentiable_bpsk_modulation_layer_16/SelectV2/e:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 22
0differentiable_bpsk_modulation_layer_16/SelectV2Ż
0differentiable_bpsk_modulation_layer_16/IdentityIdentity9differentiable_bpsk_modulation_layer_16/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 22
0differentiable_bpsk_modulation_layer_16/Identity×
1differentiable_bpsk_modulation_layer_16/IdentityN	IdentityN9differentiable_bpsk_modulation_layer_16/SelectV2:output:0/differentiable_bpsk_modulation_layer_16/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-7061076*:
_output_shapes(
&:’’’’’’’’’ :’’’’’’’’’ 23
1differentiable_bpsk_modulation_layer_16/IdentityN
IdentityIdentity:differentiable_bpsk_modulation_layer_16/IdentityN:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’ 
"
_user_specified_name
inputs/1
õ


d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_7061206

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
:’’’’’’’’’ 2
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
:’’’’’’’’’ 2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  æ2

SelectV2/e
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity·
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-7061196*:
_output_shapes(
&:’’’’’’’’’ :’’’’’’’’’ 2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
õ


d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_7060842

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
:’’’’’’’’’ 2
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
:’’’’’’’’’ 2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  æ2

SelectV2/e
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity·
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-7060832*:
_output_shapes(
&:’’’’’’’’’ :’’’’’’’’’ 2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
õ


d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_7060827

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
:’’’’’’’’’ 2
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
:’’’’’’’’’ 2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  æ2

SelectV2/e
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity·
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-7060817*:
_output_shapes(
&:’’’’’’’’’ :’’’’’’’’’ 2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
õ

V__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_fn_7061170
inputs_0
inputs_1
identity’
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *z
fuRs
q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_70607642
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’ 
"
_user_specified_name
inputs/1
1

q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_7061131
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
:’’’’’’’’’2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
sub
&product_with_external_weights_16/ShapeShapesub:z:0*
T0*
_output_shapes
:2(
&product_with_external_weights_16/Shapeæ
4product_with_external_weights_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’26
4product_with_external_weights_16/strided_slice/stackŗ
6product_with_external_weights_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6product_with_external_weights_16/strided_slice/stack_1ŗ
6product_with_external_weights_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6product_with_external_weights_16/strided_slice/stack_2Ø
.product_with_external_weights_16/strided_sliceStridedSlice/product_with_external_weights_16/Shape:output:0=product_with_external_weights_16/strided_slice/stack:output:0?product_with_external_weights_16/strided_slice/stack_1:output:0?product_with_external_weights_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.product_with_external_weights_16/strided_slice¦
0product_with_external_weights_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 22
0product_with_external_weights_16/Reshape/shape/1
.product_with_external_weights_16/Reshape/shapePack7product_with_external_weights_16/strided_slice:output:09product_with_external_weights_16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.product_with_external_weights_16/Reshape/shapeŌ
(product_with_external_weights_16/ReshapeReshapeinputs_17product_with_external_weights_16/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2*
(product_with_external_weights_16/Reshape¤
/product_with_external_weights_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/product_with_external_weights_16/ExpandDims/dimį
+product_with_external_weights_16/ExpandDims
ExpandDimssub:z:08product_with_external_weights_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2-
+product_with_external_weights_16/ExpandDims„
&product_with_external_weights_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2(
&product_with_external_weights_16/stackó
%product_with_external_weights_16/TileTile4product_with_external_weights_16/ExpandDims:output:0/product_with_external_weights_16/stack:output:0*
T0*+
_output_shapes
:’’’’’’’’’2'
%product_with_external_weights_16/Tile³
/product_with_external_weights_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/product_with_external_weights_16/transpose/perm
*product_with_external_weights_16/transpose	Transpose1product_with_external_weights_16/Reshape:output:08product_with_external_weights_16/transpose/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2,
*product_with_external_weights_16/transpose
0product_with_external_weights_16/PartitionedCallPartitionedCall.product_with_external_weights_16/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_f_706069722
0product_with_external_weights_16/PartitionedCallō
$product_with_external_weights_16/MulMul.product_with_external_weights_16/Tile:output:09product_with_external_weights_16/PartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$product_with_external_weights_16/Mul·
1product_with_external_weights_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       23
1product_with_external_weights_16/transpose_1/perm
,product_with_external_weights_16/transpose_1	Transpose1product_with_external_weights_16/Reshape:output:0:product_with_external_weights_16/transpose_1/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2.
,product_with_external_weights_16/transpose_1
2product_with_external_weights_16/PartitionedCall_1PartitionedCall0product_with_external_weights_16/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_g_706070724
2product_with_external_weights_16/PartitionedCall_1ņ
$product_with_external_weights_16/addAddV2(product_with_external_weights_16/Mul:z:0;product_with_external_weights_16/PartitionedCall_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$product_with_external_weights_16/add½
7product_with_external_weights_16/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’29
7product_with_external_weights_16/Prod/reduction_indicesō
%product_with_external_weights_16/ProdProd(product_with_external_weights_16/add:z:0@product_with_external_weights_16/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%product_with_external_weights_16/Prods
NegNeg.product_with_external_weights_16/Prod:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’ 
"
_user_specified_name
inputs/1
²
I
#__inference__traced_restore_7061270
file_prefix

identity_1¤
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices°
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
ģ
n
D__inference_encoder_layer_call_and_return_conditional_losses_7060874

inputs
inputs_1
identityē
Dlinear_block_code_product_encoder_with_external_g_16/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *z
fuRs
q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_70607642F
Dlinear_block_code_product_encoder_with_external_g_16/PartitionedCallü
7differentiable_bpsk_modulation_layer_16/PartitionedCallPartitionedCallMlinear_block_code_product_encoder_with_external_g_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *m
fhRf
d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_706082729
7differentiable_bpsk_modulation_layer_16/PartitionedCall
IdentityIdentity@differentiable_bpsk_modulation_layer_16/PartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs
§
,
__inference_g_7061230
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
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
e
p
D__inference_encoder_layer_call_and_return_conditional_losses_7061042
inputs_0
inputs_1
identity½
:linear_block_code_product_encoder_with_external_g_16/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:linear_block_code_product_encoder_with_external_g_16/mul/yü
8linear_block_code_product_encoder_with_external_g_16/mulMulinputs_0Clinear_block_code_product_encoder_with_external_g_16/mul/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’2:
8linear_block_code_product_encoder_with_external_g_16/mul½
:linear_block_code_product_encoder_with_external_g_16/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:linear_block_code_product_encoder_with_external_g_16/sub/x°
8linear_block_code_product_encoder_with_external_g_16/subSubClinear_block_code_product_encoder_with_external_g_16/sub/x:output:0<linear_block_code_product_encoder_with_external_g_16/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2:
8linear_block_code_product_encoder_with_external_g_16/sub¦
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ShapeShape<linear_block_code_product_encoder_with_external_g_16/sub:z:0*
T0*
_output_shapes
:2]
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape©
ilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2k
ilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack¤
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2m
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1¤
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2m
klinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2ę
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_sliceStridedSlicedlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Shape:output:0rlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack:output:0tlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_1:output:0tlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2e
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 2g
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1Ž
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shapePackllinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/strided_slice:output:0nlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2e
clinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shapeó
]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ReshapeReshapeinputs_1llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2_
]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2f
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dimµ
`linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
ExpandDims<linear_block_code_product_encoder_with_external_g_16/sub:z:0mlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2b
`linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2]
[linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stackĒ
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/TileTileilinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ExpandDims:output:0dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/stack:output:0*
T0*+
_output_shapes
:’’’’’’’’’2\
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/permŲ
_linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose	Transposeflinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2a
_linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallPartitionedCallclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_f_70606972g
elinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCallČ
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/MulMulclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Tile:output:0nlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2[
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul”
flinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2h
flinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/permŽ
alinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1	Transposeflinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Reshape:output:0olinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2c
alinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1„
glinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1PartitionedCallelinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_g_70607072i
glinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1Ę
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/addAddV2]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Mul:z:0plinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/PartitionedCall_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2[
Ylinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add§
llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2n
llinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indicesČ
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/ProdProd]linear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/add:z:0ulinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2\
Zlinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod
8linear_block_code_product_encoder_with_external_g_16/NegNegclinear_block_code_product_encoder_with_external_g_16/product_with_external_weights_16/Prod:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2:
8linear_block_code_product_encoder_with_external_g_16/Neg÷
<linear_block_code_product_encoder_with_external_g_16/SigmoidSigmoid<linear_block_code_product_encoder_with_external_g_16/Neg:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2>
<linear_block_code_product_encoder_with_external_g_16/Sigmoid£
-differentiable_bpsk_modulation_layer_16/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-differentiable_bpsk_modulation_layer_16/sub/y
+differentiable_bpsk_modulation_layer_16/subSub@linear_block_code_product_encoder_with_external_g_16/Sigmoid:y:06differentiable_bpsk_modulation_layer_16/sub/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2-
+differentiable_bpsk_modulation_layer_16/sub«
1differentiable_bpsk_modulation_layer_16/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1differentiable_bpsk_modulation_layer_16/Greater/y
/differentiable_bpsk_modulation_layer_16/GreaterGreater/differentiable_bpsk_modulation_layer_16/sub:z:0:differentiable_bpsk_modulation_layer_16/Greater/y:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 21
/differentiable_bpsk_modulation_layer_16/Greater­
2differentiable_bpsk_modulation_layer_16/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2differentiable_bpsk_modulation_layer_16/SelectV2/t­
2differentiable_bpsk_modulation_layer_16/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  æ24
2differentiable_bpsk_modulation_layer_16/SelectV2/eŃ
0differentiable_bpsk_modulation_layer_16/SelectV2SelectV23differentiable_bpsk_modulation_layer_16/Greater:z:0;differentiable_bpsk_modulation_layer_16/SelectV2/t:output:0;differentiable_bpsk_modulation_layer_16/SelectV2/e:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 22
0differentiable_bpsk_modulation_layer_16/SelectV2Ż
0differentiable_bpsk_modulation_layer_16/IdentityIdentity9differentiable_bpsk_modulation_layer_16/SelectV2:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 22
0differentiable_bpsk_modulation_layer_16/Identity×
1differentiable_bpsk_modulation_layer_16/IdentityN	IdentityN9differentiable_bpsk_modulation_layer_16/SelectV2:output:0/differentiable_bpsk_modulation_layer_16/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-7061032*:
_output_shapes(
&:’’’’’’’’’ :’’’’’’’’’ 23
1differentiable_bpsk_modulation_layer_16/IdentityN
IdentityIdentity:differentiable_bpsk_modulation_layer_16/IdentityN:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’ 
"
_user_specified_name
inputs/1
Ė
,
__inference_g_7061236
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xV
subSubsub/x:output:0w*
T0*'
_output_shapes
: ’’’’’’’’’2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
: ’’’’’’’’’2

Identity"
identityIdentity:output:0*&
_input_shapes
: ’’’’’’’’’:J F
'
_output_shapes
: ’’’’’’’’’

_user_specified_namew
1

q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_7060797

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
:’’’’’’’’’2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
sub
&product_with_external_weights_16/ShapeShapesub:z:0*
T0*
_output_shapes
:2(
&product_with_external_weights_16/Shapeæ
4product_with_external_weights_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’26
4product_with_external_weights_16/strided_slice/stackŗ
6product_with_external_weights_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6product_with_external_weights_16/strided_slice/stack_1ŗ
6product_with_external_weights_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6product_with_external_weights_16/strided_slice/stack_2Ø
.product_with_external_weights_16/strided_sliceStridedSlice/product_with_external_weights_16/Shape:output:0=product_with_external_weights_16/strided_slice/stack:output:0?product_with_external_weights_16/strided_slice/stack_1:output:0?product_with_external_weights_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.product_with_external_weights_16/strided_slice¦
0product_with_external_weights_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 22
0product_with_external_weights_16/Reshape/shape/1
.product_with_external_weights_16/Reshape/shapePack7product_with_external_weights_16/strided_slice:output:09product_with_external_weights_16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.product_with_external_weights_16/Reshape/shapeŌ
(product_with_external_weights_16/ReshapeReshapeinputs_17product_with_external_weights_16/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2*
(product_with_external_weights_16/Reshape¤
/product_with_external_weights_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/product_with_external_weights_16/ExpandDims/dimį
+product_with_external_weights_16/ExpandDims
ExpandDimssub:z:08product_with_external_weights_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2-
+product_with_external_weights_16/ExpandDims„
&product_with_external_weights_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2(
&product_with_external_weights_16/stackó
%product_with_external_weights_16/TileTile4product_with_external_weights_16/ExpandDims:output:0/product_with_external_weights_16/stack:output:0*
T0*+
_output_shapes
:’’’’’’’’’2'
%product_with_external_weights_16/Tile³
/product_with_external_weights_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/product_with_external_weights_16/transpose/perm
*product_with_external_weights_16/transpose	Transpose1product_with_external_weights_16/Reshape:output:08product_with_external_weights_16/transpose/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2,
*product_with_external_weights_16/transpose
0product_with_external_weights_16/PartitionedCallPartitionedCall.product_with_external_weights_16/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_f_706069722
0product_with_external_weights_16/PartitionedCallō
$product_with_external_weights_16/MulMul.product_with_external_weights_16/Tile:output:09product_with_external_weights_16/PartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$product_with_external_weights_16/Mul·
1product_with_external_weights_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       23
1product_with_external_weights_16/transpose_1/perm
,product_with_external_weights_16/transpose_1	Transpose1product_with_external_weights_16/Reshape:output:0:product_with_external_weights_16/transpose_1/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2.
,product_with_external_weights_16/transpose_1
2product_with_external_weights_16/PartitionedCall_1PartitionedCall0product_with_external_weights_16/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_g_706070724
2product_with_external_weights_16/PartitionedCall_1ņ
$product_with_external_weights_16/addAddV2(product_with_external_weights_16/Mul:z:0;product_with_external_weights_16/PartitionedCall_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$product_with_external_weights_16/add½
7product_with_external_weights_16/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’29
7product_with_external_weights_16/Prod/reduction_indicesō
%product_with_external_weights_16/ProdProd(product_with_external_weights_16/add:z:0@product_with_external_weights_16/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%product_with_external_weights_16/Prods
NegNeg.product_with_external_weights_16/Prod:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’ 
 
_user_specified_nameinputs

S
)__inference_encoder_layer_call_fn_7060998
input_1
input_2
identityŠ
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_70608872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :P L
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’ 
!
_user_specified_name	input_2
1

q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_7061164
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
:’’’’’’’’’2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
sub
&product_with_external_weights_16/ShapeShapesub:z:0*
T0*
_output_shapes
:2(
&product_with_external_weights_16/Shapeæ
4product_with_external_weights_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’26
4product_with_external_weights_16/strided_slice/stackŗ
6product_with_external_weights_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6product_with_external_weights_16/strided_slice/stack_1ŗ
6product_with_external_weights_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6product_with_external_weights_16/strided_slice/stack_2Ø
.product_with_external_weights_16/strided_sliceStridedSlice/product_with_external_weights_16/Shape:output:0=product_with_external_weights_16/strided_slice/stack:output:0?product_with_external_weights_16/strided_slice/stack_1:output:0?product_with_external_weights_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.product_with_external_weights_16/strided_slice¦
0product_with_external_weights_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 22
0product_with_external_weights_16/Reshape/shape/1
.product_with_external_weights_16/Reshape/shapePack7product_with_external_weights_16/strided_slice:output:09product_with_external_weights_16/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.product_with_external_weights_16/Reshape/shapeŌ
(product_with_external_weights_16/ReshapeReshapeinputs_17product_with_external_weights_16/Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2*
(product_with_external_weights_16/Reshape¤
/product_with_external_weights_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/product_with_external_weights_16/ExpandDims/dimį
+product_with_external_weights_16/ExpandDims
ExpandDimssub:z:08product_with_external_weights_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’2-
+product_with_external_weights_16/ExpandDims„
&product_with_external_weights_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2(
&product_with_external_weights_16/stackó
%product_with_external_weights_16/TileTile4product_with_external_weights_16/ExpandDims:output:0/product_with_external_weights_16/stack:output:0*
T0*+
_output_shapes
:’’’’’’’’’2'
%product_with_external_weights_16/Tile³
/product_with_external_weights_16/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/product_with_external_weights_16/transpose/perm
*product_with_external_weights_16/transpose	Transpose1product_with_external_weights_16/Reshape:output:08product_with_external_weights_16/transpose/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2,
*product_with_external_weights_16/transpose
0product_with_external_weights_16/PartitionedCallPartitionedCall.product_with_external_weights_16/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_f_706069722
0product_with_external_weights_16/PartitionedCallō
$product_with_external_weights_16/MulMul.product_with_external_weights_16/Tile:output:09product_with_external_weights_16/PartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$product_with_external_weights_16/Mul·
1product_with_external_weights_16/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       23
1product_with_external_weights_16/transpose_1/perm
,product_with_external_weights_16/transpose_1	Transpose1product_with_external_weights_16/Reshape:output:0:product_with_external_weights_16/transpose_1/perm:output:0*
T0*'
_output_shapes
: ’’’’’’’’’2.
,product_with_external_weights_16/transpose_1
2product_with_external_weights_16/PartitionedCall_1PartitionedCall0product_with_external_weights_16/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
: ’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_g_706070724
2product_with_external_weights_16/PartitionedCall_1ņ
$product_with_external_weights_16/addAddV2(product_with_external_weights_16/Mul:z:0;product_with_external_weights_16/PartitionedCall_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’ 2&
$product_with_external_weights_16/add½
7product_with_external_weights_16/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’29
7product_with_external_weights_16/Prod/reduction_indicesō
%product_with_external_weights_16/ProdProd(product_with_external_weights_16/add:z:0@product_with_external_weights_16/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2'
%product_with_external_weights_16/Prods
NegNeg.product_with_external_weights_16/Prod:output:0*
T0*'
_output_shapes
:’’’’’’’’’ 2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:’’’’’’’’’ 2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’ :Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’ 
"
_user_specified_name
inputs/1"±J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ą
serving_defaultĢ
;
input_10
serving_default_input_1:0’’’’’’’’’
;
input_20
serving_default_input_2:0’’’’’’’’’ 4
output_1(
PartitionedCall:0’’’’’’’’’ tensorflow/serving/predict:ø^
Ķ
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
*+&call_and_return_all_conditional_losses"ö
_tf_keras_modelÜ{"class_name": "Encoder", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Encoder"}}
ü
product
	trainable_variables

	variables
regularization_losses
	keras_api
,__call__
*-&call_and_return_all_conditional_losses"ą
_tf_keras_layerĘ{"class_name": "LinearBlockCodeProductEncoderWithExternalG", "name": "linear_block_code_product_encoder_with_external_g_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ł
trainable_variables
	variables
regularization_losses
	keras_api
.__call__
*/&call_and_return_all_conditional_losses"Ź
_tf_keras_layer°{"class_name": "DifferentiableBPSKModulationLayer", "name": "differentiable_bpsk_modulation_layer_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
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
Õ
trainable_variables
	variables
regularization_losses
	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3f
4g"ø
_tf_keras_layer{"class_name": "ProductWithExternalWeights", "name": "product_with_external_weights_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": [{"class_name": "TensorShape", "items": [64, 16]}, {"class_name": "TensorShape", "items": [16, 32]}]}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
­
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
­
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
ę2ć
)__inference_encoder_layer_call_fn_7060998
)__inference_encoder_layer_call_fn_7061092
)__inference_encoder_layer_call_fn_7060992
)__inference_encoder_layer_call_fn_7061098“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
"__inference__wrapped_model_7060726Ž
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *N¢K
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’ 
Ņ2Ļ
D__inference_encoder_layer_call_and_return_conditional_losses_7060986
D__inference_encoder_layer_call_and_return_conditional_losses_7061042
D__inference_encoder_layer_call_and_return_conditional_losses_7060942
D__inference_encoder_layer_call_and_return_conditional_losses_7061086“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ź2ē
V__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_fn_7061176
V__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_fn_7061170“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
 2
q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_7061131
q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_7061164“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Š2Ķ
I__inference_differentiable_bpsk_modulation_layer_16_layer_call_fn_7061216
I__inference_differentiable_bpsk_modulation_layer_16_layer_call_fn_7061211“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_7061191
d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_7061206“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ÓBŠ
%__inference_signature_wrapper_7060898input_1input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ń2Ī
__inference_f_7061220
__inference_f_7061224
²
FullArgSpec
args
jself
jw
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ń2Ī
__inference_g_7061230
__inference_g_7061236
²
FullArgSpec
args
jself
jw
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ¶
"__inference__wrapped_model_7060726X¢U
N¢K
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’ 
Ŗ "3Ŗ0
.
output_1"
output_1’’’’’’’’’ Ä
d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_7061191\3¢0
)¢&
 
inputs’’’’’’’’’ 
p
Ŗ "%¢"

0’’’’’’’’’ 
 Ä
d__inference_differentiable_bpsk_modulation_layer_16_layer_call_and_return_conditional_losses_7061206\3¢0
)¢&
 
inputs’’’’’’’’’ 
p 
Ŗ "%¢"

0’’’’’’’’’ 
 
I__inference_differentiable_bpsk_modulation_layer_16_layer_call_fn_7061211O3¢0
)¢&
 
inputs’’’’’’’’’ 
p
Ŗ "’’’’’’’’’ 
I__inference_differentiable_bpsk_modulation_layer_16_layer_call_fn_7061216O3¢0
)¢&
 
inputs’’’’’’’’’ 
p 
Ŗ "’’’’’’’’’ Ī
D__inference_encoder_layer_call_and_return_conditional_losses_7060942\¢Y
R¢O
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’ 
p
Ŗ "%¢"

0’’’’’’’’’ 
 Ī
D__inference_encoder_layer_call_and_return_conditional_losses_7060986\¢Y
R¢O
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’ 
p 
Ŗ "%¢"

0’’’’’’’’’ 
 Š
D__inference_encoder_layer_call_and_return_conditional_losses_7061042^¢[
T¢Q
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’ 
p
Ŗ "%¢"

0’’’’’’’’’ 
 Š
D__inference_encoder_layer_call_and_return_conditional_losses_7061086^¢[
T¢Q
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’ 
p 
Ŗ "%¢"

0’’’’’’’’’ 
 „
)__inference_encoder_layer_call_fn_7060992x\¢Y
R¢O
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’ 
p
Ŗ "’’’’’’’’’ „
)__inference_encoder_layer_call_fn_7060998x\¢Y
R¢O
IF
!
input_1’’’’’’’’’
!
input_2’’’’’’’’’ 
p 
Ŗ "’’’’’’’’’ §
)__inference_encoder_layer_call_fn_7061092z^¢[
T¢Q
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’ 
p
Ŗ "’’’’’’’’’ §
)__inference_encoder_layer_call_fn_7061098z^¢[
T¢Q
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’ 
p 
Ŗ "’’’’’’’’’ M
__inference_f_70612204!¢
¢

w 
Ŗ " _
__inference_f_7061224F*¢'
 ¢

w ’’’’’’’’’
Ŗ " ’’’’’’’’’M
__inference_g_70612304!¢
¢

w 
Ŗ " _
__inference_g_7061236F*¢'
 ¢

w ’’’’’’’’’
Ŗ " ’’’’’’’’’ż
q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_7061131^¢[
T¢Q
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’ 
p
Ŗ "%¢"

0’’’’’’’’’ 
 ż
q__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_and_return_conditional_losses_7061164^¢[
T¢Q
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’ 
p 
Ŗ "%¢"

0’’’’’’’’’ 
 Ō
V__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_fn_7061170z^¢[
T¢Q
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’ 
p
Ŗ "’’’’’’’’’ Ō
V__inference_linear_block_code_product_encoder_with_external_g_16_layer_call_fn_7061176z^¢[
T¢Q
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’ 
p 
Ŗ "’’’’’’’’’ Ź
%__inference_signature_wrapper_7060898 i¢f
¢ 
_Ŗ\
,
input_1!
input_1’’’’’’’’’
,
input_2!
input_2’’’’’’’’’ "3Ŗ0
.
output_1"
output_1’’’’’’’’’ 