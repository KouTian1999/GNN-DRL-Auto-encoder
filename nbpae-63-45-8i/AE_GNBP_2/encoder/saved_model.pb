’л
•Й
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
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
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
≥
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
Н
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Њ
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
2	"serve*2.4.32unknown8сх

NoOpNoOp
ш	
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*≥	
value©	B¶	 BЯ	
}
	lbcpe

modulation
regularization_losses
	variables
trainable_variables
	keras_api

signatures
_
product
	regularization_losses

	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
 
 
 
≠
metrics

layers
non_trainable_variables
regularization_losses
layer_regularization_losses
	variables
layer_metrics
trainable_variables
 
R
regularization_losses
	variables
trainable_variables
	keras_api
 
 
 
≠

layers
metrics
non_trainable_variables
	regularization_losses
layer_regularization_losses

	variables
layer_metrics
trainable_variables
 
 
 
≠

layers
 metrics
!non_trainable_variables
regularization_losses
"layer_regularization_losses
	variables
#layer_metrics
trainable_variables
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
≠

$layers
%metrics
&non_trainable_variables
regularization_losses
'layer_regularization_losses
	variables
(layer_metrics
trainable_variables
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
 
z
serving_default_input_1Placeholder*'
_output_shapes
:€€€€€€€€€-*
dtype0*
shape:€€€€€€€€€-
z
serving_default_input_2Placeholder*'
_output_shapes
:€€€€€€€€€?*
dtype0*
shape:€€€€€€€€€?
Њ
PartitionedCallPartitionedCallserving_default_input_1serving_default_input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_1516968
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Э
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_1517330
Ъ
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_1517340Цв
ф

Б
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_1517261

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
:€€€€€€€€€?2
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
:€€€€€€€€€?2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањ2

SelectV2/eЙ
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

IdentityЈ
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-1517251*:
_output_shapes(
&:€€€€€€€€€?:€€€€€€€€€?2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
д
n
D__inference_encoder_layer_call_and_return_conditional_losses_1516944

inputs
inputs_1
identityд
Clinear_block_code_product_encoder_with_external_g_4/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *y
ftRr
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_15168342E
Clinear_block_code_product_encoder_with_external_g_4/PartitionedCallш
6differentiable_bpsk_modulation_layer_4/PartitionedCallPartitionedCallLlinear_block_code_product_encoder_with_external_g_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *l
fgRe
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_151689728
6differentiable_bpsk_modulation_layer_4/PartitionedCallУ
IdentityIdentity?differentiable_bpsk_modulation_layer_4/PartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€-
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
Ћ
,
__inference_g_1517306
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xV
subSubsub/x:output:0w*
T0*'
_output_shapes
:?€€€€€€€€€2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:?€€€€€€€€€:J F
'
_output_shapes
:?€€€€€€€€€

_user_specified_namew
Ќ0
Ь
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_1517234
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
:€€€€€€€€€-2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€-2
subЕ
%product_with_external_weights_4/ShapeShapesub:z:0*
T0*
_output_shapes
:2'
%product_with_external_weights_4/Shapeљ
3product_with_external_weights_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€25
3product_with_external_weights_4/strided_slice/stackЄ
5product_with_external_weights_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5product_with_external_weights_4/strided_slice/stack_1Є
5product_with_external_weights_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5product_with_external_weights_4/strided_slice/stack_2Ґ
-product_with_external_weights_4/strided_sliceStridedSlice.product_with_external_weights_4/Shape:output:0<product_with_external_weights_4/strided_slice/stack:output:0>product_with_external_weights_4/strided_slice/stack_1:output:0>product_with_external_weights_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-product_with_external_weights_4/strided_slice§
/product_with_external_weights_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?21
/product_with_external_weights_4/Reshape/shape/1Ж
-product_with_external_weights_4/Reshape/shapePack6product_with_external_weights_4/strided_slice:output:08product_with_external_weights_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-product_with_external_weights_4/Reshape/shape—
'product_with_external_weights_4/ReshapeReshapeinputs_16product_with_external_weights_4/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2)
'product_with_external_weights_4/ReshapeҐ
.product_with_external_weights_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :20
.product_with_external_weights_4/ExpandDims/dimё
*product_with_external_weights_4/ExpandDims
ExpandDimssub:z:07product_with_external_weights_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*product_with_external_weights_4/ExpandDims£
%product_with_external_weights_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2'
%product_with_external_weights_4/stackп
$product_with_external_weights_4/TileTile3product_with_external_weights_4/ExpandDims:output:0.product_with_external_weights_4/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2&
$product_with_external_weights_4/Tile±
.product_with_external_weights_4/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.product_with_external_weights_4/transpose/permА
)product_with_external_weights_4/transpose	Transpose0product_with_external_weights_4/Reshape:output:07product_with_external_weights_4/transpose/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2+
)product_with_external_weights_4/transposeэ
/product_with_external_weights_4/PartitionedCallPartitionedCall-product_with_external_weights_4/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_f_151676721
/product_with_external_weights_4/PartitionedCallр
#product_with_external_weights_4/MulMul-product_with_external_weights_4/Tile:output:08product_with_external_weights_4/PartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2%
#product_with_external_weights_4/Mulµ
0product_with_external_weights_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       22
0product_with_external_weights_4/transpose_1/permЖ
+product_with_external_weights_4/transpose_1	Transpose0product_with_external_weights_4/Reshape:output:09product_with_external_weights_4/transpose_1/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2-
+product_with_external_weights_4/transpose_1Г
1product_with_external_weights_4/PartitionedCall_1PartitionedCall/product_with_external_weights_4/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_g_151677723
1product_with_external_weights_4/PartitionedCall_1о
#product_with_external_weights_4/addAddV2'product_with_external_weights_4/Mul:z:0:product_with_external_weights_4/PartitionedCall_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2%
#product_with_external_weights_4/addї
6product_with_external_weights_4/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€28
6product_with_external_weights_4/Prod/reduction_indicesр
$product_with_external_weights_4/ProdProd'product_with_external_weights_4/add:z:0?product_with_external_weights_4/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2&
$product_with_external_weights_4/Prodr
NegNeg-product_with_external_weights_4/Prod:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:Q M
'
_output_shapes
:€€€€€€€€€-
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€?
"
_user_specified_name
inputs/1
Ф
S
)__inference_encoder_layer_call_fn_1517068
input_1
input_2
identity–
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15169572
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:P L
'
_output_shapes
:€€€€€€€€€-
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€?
!
_user_specified_name	input_2
Ш
,
__inference_f_1517294
w
identityU
IdentityIdentityw*
T0*'
_output_shapes
:?€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:?€€€€€€€€€:J F
'
_output_shapes
:?€€€€€€€€€

_user_specified_namew
Ћ
,
__inference_g_1516777
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xV
subSubsub/x:output:0w*
T0*'
_output_shapes
:?€€€€€€€€€2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:?€€€€€€€€€:J F
'
_output_shapes
:?€€€€€€€€€

_user_specified_namew
ф

Б
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_1516897

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
:€€€€€€€€€?2
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
:€€€€€€€€€?2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањ2

SelectV2/eЙ
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

IdentityЈ
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-1516887*:
_output_shapes(
&:€€€€€€€€€?:€€€€€€€€€?2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
у
Б
U__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_fn_1517240
inputs_0
inputs_1
identityю
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *y
ftRr
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_15168342
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:Q M
'
_output_shapes
:€€€€€€€€€-
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€?
"
_user_specified_name
inputs/1
”
d
H__inference_differentiable_bpsk_modulation_layer_4_layer_call_fn_1517286

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *l
fgRe
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_15169122
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
э
,
__inference_f_1517290
w
identityL
IdentityIdentityw*
T0*
_output_shapes

:?-2

Identity"
identityIdentity:output:0*
_input_shapes

:?-:A =

_output_shapes

:?-

_user_specified_namew
∆j
L
"__inference__wrapped_model_1516796
input_1
input_2
identityЋ
Aencoder/linear_block_code_product_encoder_with_external_g_4/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2C
Aencoder/linear_block_code_product_encoder_with_external_g_4/mul/yР
?encoder/linear_block_code_product_encoder_with_external_g_4/mulMulinput_1Jencoder/linear_block_code_product_encoder_with_external_g_4/mul/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€-2A
?encoder/linear_block_code_product_encoder_with_external_g_4/mulЋ
Aencoder/linear_block_code_product_encoder_with_external_g_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2C
Aencoder/linear_block_code_product_encoder_with_external_g_4/sub/xћ
?encoder/linear_block_code_product_encoder_with_external_g_4/subSubJencoder/linear_block_code_product_encoder_with_external_g_4/sub/x:output:0Cencoder/linear_block_code_product_encoder_with_external_g_4/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€-2A
?encoder/linear_block_code_product_encoder_with_external_g_4/subє
aencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ShapeShapeCencoder/linear_block_code_product_encoder_with_external_g_4/sub:z:0*
T0*
_output_shapes
:2c
aencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shapeµ
oencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2q
oencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack∞
qencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2s
qencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1∞
qencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2s
qencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2К
iencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceStridedSlicejencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shape:output:0xencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack:output:0zencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1:output:0zencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2k
iencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceЬ
kencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?2m
kencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1ц
iencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapePackrencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice:output:0tencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2k
iencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapeД
cencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeReshapeinput_2rencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2e
cencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeЪ
jencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2l
jencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimќ
fencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims
ExpandDimsCencoder/linear_block_code_product_encoder_with_external_g_4/sub:z:0sencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2h
fencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDimsЫ
aencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2c
aencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackя
`encoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/TileTileoencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims:output:0jencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2b
`encoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Tile©
jencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2l
jencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/permр
eencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose	Transposelencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0sencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2g
eencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose±
kencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallPartitionedCalliencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_f_15167672m
kencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallа
_encoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/MulMuliencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Tile:output:0tencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2a
_encoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Mul≠
lencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2n
lencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/permц
gencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1	Transposelencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0uencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2i
gencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1Ј
mencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1PartitionedCallkencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_g_15167772o
mencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1ё
_encoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/addAddV2cencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Mul:z:0vencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2a
_encoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add≥
rencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2t
rencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesа
`encoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ProdProdcencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add:z:0{encoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2b
`encoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod¶
?encoder/linear_block_code_product_encoder_with_external_g_4/NegNegiencoder/linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2A
?encoder/linear_block_code_product_encoder_with_external_g_4/NegМ
Cencoder/linear_block_code_product_encoder_with_external_g_4/SigmoidSigmoidCencoder/linear_block_code_product_encoder_with_external_g_4/Neg:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2E
Cencoder/linear_block_code_product_encoder_with_external_g_4/Sigmoid±
4encoder/differentiable_bpsk_modulation_layer_4/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4encoder/differentiable_bpsk_modulation_layer_4/sub/y©
2encoder/differentiable_bpsk_modulation_layer_4/subSubGencoder/linear_block_code_product_encoder_with_external_g_4/Sigmoid:y:0=encoder/differentiable_bpsk_modulation_layer_4/sub/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?24
2encoder/differentiable_bpsk_modulation_layer_4/subє
8encoder/differentiable_bpsk_modulation_layer_4/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8encoder/differentiable_bpsk_modulation_layer_4/Greater/y®
6encoder/differentiable_bpsk_modulation_layer_4/GreaterGreater6encoder/differentiable_bpsk_modulation_layer_4/sub:z:0Aencoder/differentiable_bpsk_modulation_layer_4/Greater/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?28
6encoder/differentiable_bpsk_modulation_layer_4/Greaterї
9encoder/differentiable_bpsk_modulation_layer_4/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2;
9encoder/differentiable_bpsk_modulation_layer_4/SelectV2/tї
9encoder/differentiable_bpsk_modulation_layer_4/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањ2;
9encoder/differentiable_bpsk_modulation_layer_4/SelectV2/eф
7encoder/differentiable_bpsk_modulation_layer_4/SelectV2SelectV2:encoder/differentiable_bpsk_modulation_layer_4/Greater:z:0Bencoder/differentiable_bpsk_modulation_layer_4/SelectV2/t:output:0Bencoder/differentiable_bpsk_modulation_layer_4/SelectV2/e:output:0*
T0*'
_output_shapes
:€€€€€€€€€?29
7encoder/differentiable_bpsk_modulation_layer_4/SelectV2т
7encoder/differentiable_bpsk_modulation_layer_4/IdentityIdentity@encoder/differentiable_bpsk_modulation_layer_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€?29
7encoder/differentiable_bpsk_modulation_layer_4/Identityу
8encoder/differentiable_bpsk_modulation_layer_4/IdentityN	IdentityN@encoder/differentiable_bpsk_modulation_layer_4/SelectV2:output:06encoder/differentiable_bpsk_modulation_layer_4/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-1516786*:
_output_shapes(
&:€€€€€€€€€?:€€€€€€€€€?2:
8encoder/differentiable_bpsk_modulation_layer_4/IdentityNХ
IdentityIdentityAencoder/differentiable_bpsk_modulation_layer_4/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:P L
'
_output_shapes
:€€€€€€€€€-
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€?
!
_user_specified_name	input_2
Ф
S
)__inference_encoder_layer_call_fn_1517062
input_1
input_2
identity–
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15169442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:P L
'
_output_shapes
:€€€€€€€€€-
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€?
!
_user_specified_name	input_2
÷
m
 __inference__traced_save_1517330
file_prefix
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЮ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesК
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slicesЇ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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
Ќ0
Ь
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_1517201
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
:€€€€€€€€€-2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€-2
subЕ
%product_with_external_weights_4/ShapeShapesub:z:0*
T0*
_output_shapes
:2'
%product_with_external_weights_4/Shapeљ
3product_with_external_weights_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€25
3product_with_external_weights_4/strided_slice/stackЄ
5product_with_external_weights_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5product_with_external_weights_4/strided_slice/stack_1Є
5product_with_external_weights_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5product_with_external_weights_4/strided_slice/stack_2Ґ
-product_with_external_weights_4/strided_sliceStridedSlice.product_with_external_weights_4/Shape:output:0<product_with_external_weights_4/strided_slice/stack:output:0>product_with_external_weights_4/strided_slice/stack_1:output:0>product_with_external_weights_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-product_with_external_weights_4/strided_slice§
/product_with_external_weights_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?21
/product_with_external_weights_4/Reshape/shape/1Ж
-product_with_external_weights_4/Reshape/shapePack6product_with_external_weights_4/strided_slice:output:08product_with_external_weights_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-product_with_external_weights_4/Reshape/shape—
'product_with_external_weights_4/ReshapeReshapeinputs_16product_with_external_weights_4/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2)
'product_with_external_weights_4/ReshapeҐ
.product_with_external_weights_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :20
.product_with_external_weights_4/ExpandDims/dimё
*product_with_external_weights_4/ExpandDims
ExpandDimssub:z:07product_with_external_weights_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*product_with_external_weights_4/ExpandDims£
%product_with_external_weights_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2'
%product_with_external_weights_4/stackп
$product_with_external_weights_4/TileTile3product_with_external_weights_4/ExpandDims:output:0.product_with_external_weights_4/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2&
$product_with_external_weights_4/Tile±
.product_with_external_weights_4/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.product_with_external_weights_4/transpose/permА
)product_with_external_weights_4/transpose	Transpose0product_with_external_weights_4/Reshape:output:07product_with_external_weights_4/transpose/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2+
)product_with_external_weights_4/transposeэ
/product_with_external_weights_4/PartitionedCallPartitionedCall-product_with_external_weights_4/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_f_151676721
/product_with_external_weights_4/PartitionedCallр
#product_with_external_weights_4/MulMul-product_with_external_weights_4/Tile:output:08product_with_external_weights_4/PartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2%
#product_with_external_weights_4/Mulµ
0product_with_external_weights_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       22
0product_with_external_weights_4/transpose_1/permЖ
+product_with_external_weights_4/transpose_1	Transpose0product_with_external_weights_4/Reshape:output:09product_with_external_weights_4/transpose_1/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2-
+product_with_external_weights_4/transpose_1Г
1product_with_external_weights_4/PartitionedCall_1PartitionedCall/product_with_external_weights_4/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_g_151677723
1product_with_external_weights_4/PartitionedCall_1о
#product_with_external_weights_4/addAddV2'product_with_external_weights_4/Mul:z:0:product_with_external_weights_4/PartitionedCall_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2%
#product_with_external_weights_4/addї
6product_with_external_weights_4/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€28
6product_with_external_weights_4/Prod/reduction_indicesр
$product_with_external_weights_4/ProdProd'product_with_external_weights_4/add:z:0?product_with_external_weights_4/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2&
$product_with_external_weights_4/Prodr
NegNeg-product_with_external_weights_4/Prod:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:Q M
'
_output_shapes
:€€€€€€€€€-
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€?
"
_user_specified_name
inputs/1
о
O
%__inference_signature_wrapper_1516968
input_1
input_2
identityЃ
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_15167962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:P L
'
_output_shapes
:€€€€€€€€€-
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€?
!
_user_specified_name	input_2
Ъ
U
)__inference_encoder_layer_call_fn_1517168
inputs_0
inputs_1
identity“
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15169572
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:Q M
'
_output_shapes
:€€€€€€€€€-
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€?
"
_user_specified_name
inputs/1
ф

Б
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_1516912

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
:€€€€€€€€€?2
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
:€€€€€€€€€?2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањ2

SelectV2/eЙ
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

IdentityЈ
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-1516902*:
_output_shapes(
&:€€€€€€€€€?:€€€€€€€€€?2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
Ш
,
__inference_f_1516767
w
identityU
IdentityIdentityw*
T0*'
_output_shapes
:?€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:?€€€€€€€€€:J F
'
_output_shapes
:?€€€€€€€€€

_user_specified_namew
≤
I
#__inference__traced_restore_1517340
file_prefix

identity_1И§
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesР
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices∞
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
Ўc
n
D__inference_encoder_layer_call_and_return_conditional_losses_1517056
input_1
input_2
identityї
9linear_block_code_product_encoder_with_external_g_4/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2;
9linear_block_code_product_encoder_with_external_g_4/mul/yш
7linear_block_code_product_encoder_with_external_g_4/mulMulinput_1Blinear_block_code_product_encoder_with_external_g_4/mul/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€-29
7linear_block_code_product_encoder_with_external_g_4/mulї
9linear_block_code_product_encoder_with_external_g_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2;
9linear_block_code_product_encoder_with_external_g_4/sub/xђ
7linear_block_code_product_encoder_with_external_g_4/subSubBlinear_block_code_product_encoder_with_external_g_4/sub/x:output:0;linear_block_code_product_encoder_with_external_g_4/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€-29
7linear_block_code_product_encoder_with_external_g_4/sub°
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ShapeShape;linear_block_code_product_encoder_with_external_g_4/sub:z:0*
T0*
_output_shapes
:2[
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shape•
glinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2i
glinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack†
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2k
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1†
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2k
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2Џ
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceStridedSliceblinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shape:output:0plinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack:output:0rlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1:output:0rlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2c
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceМ
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?2e
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1÷
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapePackjlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice:output:0llinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2c
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapeм
[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeReshapeinput_2jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2]
[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeК
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2d
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimЃ
^linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims
ExpandDims;linear_block_code_product_encoder_with_external_g_4/sub:z:0klinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2`
^linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDimsЛ
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2[
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackњ
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/TileTileglinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims:output:0blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2Z
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/TileЩ
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2d
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/perm–
]linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose	Transposedlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0klinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2_
]linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transposeЩ
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallPartitionedCallalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_f_15167672e
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallј
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/MulMulalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Tile:output:0llinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2Y
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/MulЭ
dlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/perm÷
_linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1	Transposedlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2a
_linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1Я
elinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1PartitionedCallclinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_g_15167772g
elinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1Њ
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/addAddV2[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Mul:z:0nlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2Y
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add£
jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2l
jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesј
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ProdProd[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add:z:0slinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2Z
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ProdО
7linear_block_code_product_encoder_with_external_g_4/NegNegalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod:output:0*
T0*'
_output_shapes
:€€€€€€€€€?29
7linear_block_code_product_encoder_with_external_g_4/Negф
;linear_block_code_product_encoder_with_external_g_4/SigmoidSigmoid;linear_block_code_product_encoder_with_external_g_4/Neg:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2=
;linear_block_code_product_encoder_with_external_g_4/Sigmoid°
,differentiable_bpsk_modulation_layer_4/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,differentiable_bpsk_modulation_layer_4/sub/yЙ
*differentiable_bpsk_modulation_layer_4/subSub?linear_block_code_product_encoder_with_external_g_4/Sigmoid:y:05differentiable_bpsk_modulation_layer_4/sub/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2,
*differentiable_bpsk_modulation_layer_4/sub©
0differentiable_bpsk_modulation_layer_4/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    22
0differentiable_bpsk_modulation_layer_4/Greater/yИ
.differentiable_bpsk_modulation_layer_4/GreaterGreater.differentiable_bpsk_modulation_layer_4/sub:z:09differentiable_bpsk_modulation_layer_4/Greater/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?20
.differentiable_bpsk_modulation_layer_4/GreaterЂ
1differentiable_bpsk_modulation_layer_4/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  А?23
1differentiable_bpsk_modulation_layer_4/SelectV2/tЂ
1differentiable_bpsk_modulation_layer_4/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањ23
1differentiable_bpsk_modulation_layer_4/SelectV2/eћ
/differentiable_bpsk_modulation_layer_4/SelectV2SelectV22differentiable_bpsk_modulation_layer_4/Greater:z:0:differentiable_bpsk_modulation_layer_4/SelectV2/t:output:0:differentiable_bpsk_modulation_layer_4/SelectV2/e:output:0*
T0*'
_output_shapes
:€€€€€€€€€?21
/differentiable_bpsk_modulation_layer_4/SelectV2Џ
/differentiable_bpsk_modulation_layer_4/IdentityIdentity8differentiable_bpsk_modulation_layer_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€?21
/differentiable_bpsk_modulation_layer_4/Identity”
0differentiable_bpsk_modulation_layer_4/IdentityN	IdentityN8differentiable_bpsk_modulation_layer_4/SelectV2:output:0.differentiable_bpsk_modulation_layer_4/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-1517046*:
_output_shapes(
&:€€€€€€€€€?:€€€€€€€€€?22
0differentiable_bpsk_modulation_layer_4/IdentityNН
IdentityIdentity9differentiable_bpsk_modulation_layer_4/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:P L
'
_output_shapes
:€€€€€€€€€-
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€?
!
_user_specified_name	input_2
ф

Б
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_1517276

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
:€€€€€€€€€?2
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
:€€€€€€€€€?2	
Greater]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањ2

SelectV2/eЙ
SelectV2SelectV2Greater:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

IdentityЈ
	IdentityN	IdentityNSelectV2:output:0sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-1517266*:
_output_shapes(
&:€€€€€€€€€?:€€€€€€€€€?2
	IdentityNj

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
у
Б
U__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_fn_1517246
inputs_0
inputs_1
identityю
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *y
ftRr
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_15168672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:Q M
'
_output_shapes
:€€€€€€€€€-
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€?
"
_user_specified_name
inputs/1
ёc
p
D__inference_encoder_layer_call_and_return_conditional_losses_1517156
inputs_0
inputs_1
identityї
9linear_block_code_product_encoder_with_external_g_4/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2;
9linear_block_code_product_encoder_with_external_g_4/mul/yщ
7linear_block_code_product_encoder_with_external_g_4/mulMulinputs_0Blinear_block_code_product_encoder_with_external_g_4/mul/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€-29
7linear_block_code_product_encoder_with_external_g_4/mulї
9linear_block_code_product_encoder_with_external_g_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2;
9linear_block_code_product_encoder_with_external_g_4/sub/xђ
7linear_block_code_product_encoder_with_external_g_4/subSubBlinear_block_code_product_encoder_with_external_g_4/sub/x:output:0;linear_block_code_product_encoder_with_external_g_4/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€-29
7linear_block_code_product_encoder_with_external_g_4/sub°
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ShapeShape;linear_block_code_product_encoder_with_external_g_4/sub:z:0*
T0*
_output_shapes
:2[
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shape•
glinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2i
glinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack†
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2k
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1†
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2k
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2Џ
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceStridedSliceblinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shape:output:0plinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack:output:0rlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1:output:0rlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2c
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceМ
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?2e
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1÷
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapePackjlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice:output:0llinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2c
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapeн
[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeReshapeinputs_1jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2]
[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeК
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2d
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimЃ
^linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims
ExpandDims;linear_block_code_product_encoder_with_external_g_4/sub:z:0klinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2`
^linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDimsЛ
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2[
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackњ
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/TileTileglinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims:output:0blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2Z
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/TileЩ
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2d
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/perm–
]linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose	Transposedlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0klinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2_
]linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transposeЩ
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallPartitionedCallalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_f_15167672e
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallј
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/MulMulalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Tile:output:0llinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2Y
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/MulЭ
dlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/perm÷
_linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1	Transposedlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2a
_linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1Я
elinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1PartitionedCallclinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_g_15167772g
elinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1Њ
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/addAddV2[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Mul:z:0nlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2Y
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add£
jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2l
jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesј
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ProdProd[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add:z:0slinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2Z
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ProdО
7linear_block_code_product_encoder_with_external_g_4/NegNegalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod:output:0*
T0*'
_output_shapes
:€€€€€€€€€?29
7linear_block_code_product_encoder_with_external_g_4/Negф
;linear_block_code_product_encoder_with_external_g_4/SigmoidSigmoid;linear_block_code_product_encoder_with_external_g_4/Neg:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2=
;linear_block_code_product_encoder_with_external_g_4/Sigmoid°
,differentiable_bpsk_modulation_layer_4/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,differentiable_bpsk_modulation_layer_4/sub/yЙ
*differentiable_bpsk_modulation_layer_4/subSub?linear_block_code_product_encoder_with_external_g_4/Sigmoid:y:05differentiable_bpsk_modulation_layer_4/sub/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2,
*differentiable_bpsk_modulation_layer_4/sub©
0differentiable_bpsk_modulation_layer_4/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    22
0differentiable_bpsk_modulation_layer_4/Greater/yИ
.differentiable_bpsk_modulation_layer_4/GreaterGreater.differentiable_bpsk_modulation_layer_4/sub:z:09differentiable_bpsk_modulation_layer_4/Greater/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?20
.differentiable_bpsk_modulation_layer_4/GreaterЂ
1differentiable_bpsk_modulation_layer_4/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  А?23
1differentiable_bpsk_modulation_layer_4/SelectV2/tЂ
1differentiable_bpsk_modulation_layer_4/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањ23
1differentiable_bpsk_modulation_layer_4/SelectV2/eћ
/differentiable_bpsk_modulation_layer_4/SelectV2SelectV22differentiable_bpsk_modulation_layer_4/Greater:z:0:differentiable_bpsk_modulation_layer_4/SelectV2/t:output:0:differentiable_bpsk_modulation_layer_4/SelectV2/e:output:0*
T0*'
_output_shapes
:€€€€€€€€€?21
/differentiable_bpsk_modulation_layer_4/SelectV2Џ
/differentiable_bpsk_modulation_layer_4/IdentityIdentity8differentiable_bpsk_modulation_layer_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€?21
/differentiable_bpsk_modulation_layer_4/Identity”
0differentiable_bpsk_modulation_layer_4/IdentityN	IdentityN8differentiable_bpsk_modulation_layer_4/SelectV2:output:0.differentiable_bpsk_modulation_layer_4/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-1517146*:
_output_shapes(
&:€€€€€€€€€?:€€€€€€€€€?22
0differentiable_bpsk_modulation_layer_4/IdentityNН
IdentityIdentity9differentiable_bpsk_modulation_layer_4/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:Q M
'
_output_shapes
:€€€€€€€€€-
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€?
"
_user_specified_name
inputs/1
Ъ
U
)__inference_encoder_layer_call_fn_1517162
inputs_0
inputs_1
identity“
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15169442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:Q M
'
_output_shapes
:€€€€€€€€€-
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€?
"
_user_specified_name
inputs/1
І
,
__inference_g_1517300
w
identityS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xM
subSubsub/x:output:0w*
T0*
_output_shapes

:?-2
subR
IdentityIdentitysub:z:0*
T0*
_output_shapes

:?-2

Identity"
identityIdentity:output:0*
_input_shapes

:?-:A =

_output_shapes

:?-

_user_specified_namew
д
n
D__inference_encoder_layer_call_and_return_conditional_losses_1516957

inputs
inputs_1
identityд
Clinear_block_code_product_encoder_with_external_g_4/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *y
ftRr
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_15168672E
Clinear_block_code_product_encoder_with_external_g_4/PartitionedCallш
6differentiable_bpsk_modulation_layer_4/PartitionedCallPartitionedCallLlinear_block_code_product_encoder_with_external_g_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *l
fgRe
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_151691228
6differentiable_bpsk_modulation_layer_4/PartitionedCallУ
IdentityIdentity?differentiable_bpsk_modulation_layer_4/PartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€-
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
”
d
H__inference_differentiable_bpsk_modulation_layer_4_layer_call_fn_1517281

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *l
fgRe
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_15168972
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
Ўc
n
D__inference_encoder_layer_call_and_return_conditional_losses_1517012
input_1
input_2
identityї
9linear_block_code_product_encoder_with_external_g_4/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2;
9linear_block_code_product_encoder_with_external_g_4/mul/yш
7linear_block_code_product_encoder_with_external_g_4/mulMulinput_1Blinear_block_code_product_encoder_with_external_g_4/mul/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€-29
7linear_block_code_product_encoder_with_external_g_4/mulї
9linear_block_code_product_encoder_with_external_g_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2;
9linear_block_code_product_encoder_with_external_g_4/sub/xђ
7linear_block_code_product_encoder_with_external_g_4/subSubBlinear_block_code_product_encoder_with_external_g_4/sub/x:output:0;linear_block_code_product_encoder_with_external_g_4/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€-29
7linear_block_code_product_encoder_with_external_g_4/sub°
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ShapeShape;linear_block_code_product_encoder_with_external_g_4/sub:z:0*
T0*
_output_shapes
:2[
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shape•
glinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2i
glinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack†
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2k
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1†
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2k
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2Џ
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceStridedSliceblinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shape:output:0plinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack:output:0rlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1:output:0rlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2c
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceМ
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?2e
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1÷
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapePackjlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice:output:0llinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2c
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapeм
[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeReshapeinput_2jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2]
[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeК
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2d
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimЃ
^linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims
ExpandDims;linear_block_code_product_encoder_with_external_g_4/sub:z:0klinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2`
^linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDimsЛ
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2[
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackњ
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/TileTileglinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims:output:0blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2Z
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/TileЩ
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2d
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/perm–
]linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose	Transposedlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0klinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2_
]linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transposeЩ
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallPartitionedCallalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_f_15167672e
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallј
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/MulMulalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Tile:output:0llinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2Y
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/MulЭ
dlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/perm÷
_linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1	Transposedlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2a
_linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1Я
elinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1PartitionedCallclinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_g_15167772g
elinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1Њ
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/addAddV2[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Mul:z:0nlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2Y
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add£
jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2l
jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesј
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ProdProd[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add:z:0slinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2Z
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ProdО
7linear_block_code_product_encoder_with_external_g_4/NegNegalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod:output:0*
T0*'
_output_shapes
:€€€€€€€€€?29
7linear_block_code_product_encoder_with_external_g_4/Negф
;linear_block_code_product_encoder_with_external_g_4/SigmoidSigmoid;linear_block_code_product_encoder_with_external_g_4/Neg:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2=
;linear_block_code_product_encoder_with_external_g_4/Sigmoid°
,differentiable_bpsk_modulation_layer_4/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,differentiable_bpsk_modulation_layer_4/sub/yЙ
*differentiable_bpsk_modulation_layer_4/subSub?linear_block_code_product_encoder_with_external_g_4/Sigmoid:y:05differentiable_bpsk_modulation_layer_4/sub/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2,
*differentiable_bpsk_modulation_layer_4/sub©
0differentiable_bpsk_modulation_layer_4/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    22
0differentiable_bpsk_modulation_layer_4/Greater/yИ
.differentiable_bpsk_modulation_layer_4/GreaterGreater.differentiable_bpsk_modulation_layer_4/sub:z:09differentiable_bpsk_modulation_layer_4/Greater/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?20
.differentiable_bpsk_modulation_layer_4/GreaterЂ
1differentiable_bpsk_modulation_layer_4/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  А?23
1differentiable_bpsk_modulation_layer_4/SelectV2/tЂ
1differentiable_bpsk_modulation_layer_4/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањ23
1differentiable_bpsk_modulation_layer_4/SelectV2/eћ
/differentiable_bpsk_modulation_layer_4/SelectV2SelectV22differentiable_bpsk_modulation_layer_4/Greater:z:0:differentiable_bpsk_modulation_layer_4/SelectV2/t:output:0:differentiable_bpsk_modulation_layer_4/SelectV2/e:output:0*
T0*'
_output_shapes
:€€€€€€€€€?21
/differentiable_bpsk_modulation_layer_4/SelectV2Џ
/differentiable_bpsk_modulation_layer_4/IdentityIdentity8differentiable_bpsk_modulation_layer_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€?21
/differentiable_bpsk_modulation_layer_4/Identity”
0differentiable_bpsk_modulation_layer_4/IdentityN	IdentityN8differentiable_bpsk_modulation_layer_4/SelectV2:output:0.differentiable_bpsk_modulation_layer_4/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-1517002*:
_output_shapes(
&:€€€€€€€€€?:€€€€€€€€€?22
0differentiable_bpsk_modulation_layer_4/IdentityNН
IdentityIdentity9differentiable_bpsk_modulation_layer_4/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:P L
'
_output_shapes
:€€€€€€€€€-
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€?
!
_user_specified_name	input_2
≈0
Ъ
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_1516867

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
:€€€€€€€€€-2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€-2
subЕ
%product_with_external_weights_4/ShapeShapesub:z:0*
T0*
_output_shapes
:2'
%product_with_external_weights_4/Shapeљ
3product_with_external_weights_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€25
3product_with_external_weights_4/strided_slice/stackЄ
5product_with_external_weights_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5product_with_external_weights_4/strided_slice/stack_1Є
5product_with_external_weights_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5product_with_external_weights_4/strided_slice/stack_2Ґ
-product_with_external_weights_4/strided_sliceStridedSlice.product_with_external_weights_4/Shape:output:0<product_with_external_weights_4/strided_slice/stack:output:0>product_with_external_weights_4/strided_slice/stack_1:output:0>product_with_external_weights_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-product_with_external_weights_4/strided_slice§
/product_with_external_weights_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?21
/product_with_external_weights_4/Reshape/shape/1Ж
-product_with_external_weights_4/Reshape/shapePack6product_with_external_weights_4/strided_slice:output:08product_with_external_weights_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-product_with_external_weights_4/Reshape/shape—
'product_with_external_weights_4/ReshapeReshapeinputs_16product_with_external_weights_4/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2)
'product_with_external_weights_4/ReshapeҐ
.product_with_external_weights_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :20
.product_with_external_weights_4/ExpandDims/dimё
*product_with_external_weights_4/ExpandDims
ExpandDimssub:z:07product_with_external_weights_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*product_with_external_weights_4/ExpandDims£
%product_with_external_weights_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2'
%product_with_external_weights_4/stackп
$product_with_external_weights_4/TileTile3product_with_external_weights_4/ExpandDims:output:0.product_with_external_weights_4/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2&
$product_with_external_weights_4/Tile±
.product_with_external_weights_4/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.product_with_external_weights_4/transpose/permА
)product_with_external_weights_4/transpose	Transpose0product_with_external_weights_4/Reshape:output:07product_with_external_weights_4/transpose/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2+
)product_with_external_weights_4/transposeэ
/product_with_external_weights_4/PartitionedCallPartitionedCall-product_with_external_weights_4/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_f_151676721
/product_with_external_weights_4/PartitionedCallр
#product_with_external_weights_4/MulMul-product_with_external_weights_4/Tile:output:08product_with_external_weights_4/PartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2%
#product_with_external_weights_4/Mulµ
0product_with_external_weights_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       22
0product_with_external_weights_4/transpose_1/permЖ
+product_with_external_weights_4/transpose_1	Transpose0product_with_external_weights_4/Reshape:output:09product_with_external_weights_4/transpose_1/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2-
+product_with_external_weights_4/transpose_1Г
1product_with_external_weights_4/PartitionedCall_1PartitionedCall/product_with_external_weights_4/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_g_151677723
1product_with_external_weights_4/PartitionedCall_1о
#product_with_external_weights_4/addAddV2'product_with_external_weights_4/Mul:z:0:product_with_external_weights_4/PartitionedCall_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2%
#product_with_external_weights_4/addї
6product_with_external_weights_4/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€28
6product_with_external_weights_4/Prod/reduction_indicesр
$product_with_external_weights_4/ProdProd'product_with_external_weights_4/add:z:0?product_with_external_weights_4/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2&
$product_with_external_weights_4/Prodr
NegNeg-product_with_external_weights_4/Prod:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€-
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
≈0
Ъ
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_1516834

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
:€€€€€€€€€-2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/x\
subSubsub/x:output:0mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€-2
subЕ
%product_with_external_weights_4/ShapeShapesub:z:0*
T0*
_output_shapes
:2'
%product_with_external_weights_4/Shapeљ
3product_with_external_weights_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€25
3product_with_external_weights_4/strided_slice/stackЄ
5product_with_external_weights_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5product_with_external_weights_4/strided_slice/stack_1Є
5product_with_external_weights_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5product_with_external_weights_4/strided_slice/stack_2Ґ
-product_with_external_weights_4/strided_sliceStridedSlice.product_with_external_weights_4/Shape:output:0<product_with_external_weights_4/strided_slice/stack:output:0>product_with_external_weights_4/strided_slice/stack_1:output:0>product_with_external_weights_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-product_with_external_weights_4/strided_slice§
/product_with_external_weights_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?21
/product_with_external_weights_4/Reshape/shape/1Ж
-product_with_external_weights_4/Reshape/shapePack6product_with_external_weights_4/strided_slice:output:08product_with_external_weights_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-product_with_external_weights_4/Reshape/shape—
'product_with_external_weights_4/ReshapeReshapeinputs_16product_with_external_weights_4/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2)
'product_with_external_weights_4/ReshapeҐ
.product_with_external_weights_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :20
.product_with_external_weights_4/ExpandDims/dimё
*product_with_external_weights_4/ExpandDims
ExpandDimssub:z:07product_with_external_weights_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*product_with_external_weights_4/ExpandDims£
%product_with_external_weights_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2'
%product_with_external_weights_4/stackп
$product_with_external_weights_4/TileTile3product_with_external_weights_4/ExpandDims:output:0.product_with_external_weights_4/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2&
$product_with_external_weights_4/Tile±
.product_with_external_weights_4/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       20
.product_with_external_weights_4/transpose/permА
)product_with_external_weights_4/transpose	Transpose0product_with_external_weights_4/Reshape:output:07product_with_external_weights_4/transpose/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2+
)product_with_external_weights_4/transposeэ
/product_with_external_weights_4/PartitionedCallPartitionedCall-product_with_external_weights_4/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_f_151676721
/product_with_external_weights_4/PartitionedCallр
#product_with_external_weights_4/MulMul-product_with_external_weights_4/Tile:output:08product_with_external_weights_4/PartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2%
#product_with_external_weights_4/Mulµ
0product_with_external_weights_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       22
0product_with_external_weights_4/transpose_1/permЖ
+product_with_external_weights_4/transpose_1	Transpose0product_with_external_weights_4/Reshape:output:09product_with_external_weights_4/transpose_1/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2-
+product_with_external_weights_4/transpose_1Г
1product_with_external_weights_4/PartitionedCall_1PartitionedCall/product_with_external_weights_4/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_g_151677723
1product_with_external_weights_4/PartitionedCall_1о
#product_with_external_weights_4/addAddV2'product_with_external_weights_4/Mul:z:0:product_with_external_weights_4/PartitionedCall_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2%
#product_with_external_weights_4/addї
6product_with_external_weights_4/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€28
6product_with_external_weights_4/Prod/reduction_indicesр
$product_with_external_weights_4/ProdProd'product_with_external_weights_4/add:z:0?product_with_external_weights_4/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2&
$product_with_external_weights_4/Prodr
NegNeg-product_with_external_weights_4/Prod:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2
NegX
SigmoidSigmoidNeg:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:O K
'
_output_shapes
:€€€€€€€€€-
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€?
 
_user_specified_nameinputs
ёc
p
D__inference_encoder_layer_call_and_return_conditional_losses_1517112
inputs_0
inputs_1
identityї
9linear_block_code_product_encoder_with_external_g_4/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2;
9linear_block_code_product_encoder_with_external_g_4/mul/yщ
7linear_block_code_product_encoder_with_external_g_4/mulMulinputs_0Blinear_block_code_product_encoder_with_external_g_4/mul/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€-29
7linear_block_code_product_encoder_with_external_g_4/mulї
9linear_block_code_product_encoder_with_external_g_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2;
9linear_block_code_product_encoder_with_external_g_4/sub/xђ
7linear_block_code_product_encoder_with_external_g_4/subSubBlinear_block_code_product_encoder_with_external_g_4/sub/x:output:0;linear_block_code_product_encoder_with_external_g_4/mul:z:0*
T0*'
_output_shapes
:€€€€€€€€€-29
7linear_block_code_product_encoder_with_external_g_4/sub°
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ShapeShape;linear_block_code_product_encoder_with_external_g_4/sub:z:0*
T0*
_output_shapes
:2[
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shape•
glinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2i
glinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack†
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2k
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1†
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2k
ilinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2Џ
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceStridedSliceblinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Shape:output:0plinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack:output:0rlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_1:output:0rlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2c
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_sliceМ
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?2e
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1÷
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapePackjlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/strided_slice:output:0llinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2c
alinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shapeн
[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeReshapeinputs_1jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2]
[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ReshapeК
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2d
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dimЃ
^linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims
ExpandDims;linear_block_code_product_encoder_with_external_g_4/sub:z:0klinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2`
^linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDimsЛ
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"         2[
Ylinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stackњ
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/TileTileglinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ExpandDims:output:0blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/stack:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2Z
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/TileЩ
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2d
blinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/perm–
]linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose	Transposedlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0klinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2_
]linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transposeЩ
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallPartitionedCallalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_f_15167672e
clinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCallј
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/MulMulalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Tile:output:0llinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2Y
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/MulЭ
dlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2f
dlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/perm÷
_linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1	Transposedlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Reshape:output:0mlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1/perm:output:0*
T0*'
_output_shapes
:?€€€€€€€€€2a
_linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1Я
elinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1PartitionedCallclinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/transpose_1:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *
fR
__inference_g_15167772g
elinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1Њ
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/addAddV2[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Mul:z:0nlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/PartitionedCall_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€?-2Y
Wlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add£
jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2l
jlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indicesј
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ProdProd[linear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/add:z:0slinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2Z
Xlinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/ProdО
7linear_block_code_product_encoder_with_external_g_4/NegNegalinear_block_code_product_encoder_with_external_g_4/product_with_external_weights_4/Prod:output:0*
T0*'
_output_shapes
:€€€€€€€€€?29
7linear_block_code_product_encoder_with_external_g_4/Negф
;linear_block_code_product_encoder_with_external_g_4/SigmoidSigmoid;linear_block_code_product_encoder_with_external_g_4/Neg:y:0*
T0*'
_output_shapes
:€€€€€€€€€?2=
;linear_block_code_product_encoder_with_external_g_4/Sigmoid°
,differentiable_bpsk_modulation_layer_4/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,differentiable_bpsk_modulation_layer_4/sub/yЙ
*differentiable_bpsk_modulation_layer_4/subSub?linear_block_code_product_encoder_with_external_g_4/Sigmoid:y:05differentiable_bpsk_modulation_layer_4/sub/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2,
*differentiable_bpsk_modulation_layer_4/sub©
0differentiable_bpsk_modulation_layer_4/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    22
0differentiable_bpsk_modulation_layer_4/Greater/yИ
.differentiable_bpsk_modulation_layer_4/GreaterGreater.differentiable_bpsk_modulation_layer_4/sub:z:09differentiable_bpsk_modulation_layer_4/Greater/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€?20
.differentiable_bpsk_modulation_layer_4/GreaterЂ
1differentiable_bpsk_modulation_layer_4/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *  А?23
1differentiable_bpsk_modulation_layer_4/SelectV2/tЂ
1differentiable_bpsk_modulation_layer_4/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањ23
1differentiable_bpsk_modulation_layer_4/SelectV2/eћ
/differentiable_bpsk_modulation_layer_4/SelectV2SelectV22differentiable_bpsk_modulation_layer_4/Greater:z:0:differentiable_bpsk_modulation_layer_4/SelectV2/t:output:0:differentiable_bpsk_modulation_layer_4/SelectV2/e:output:0*
T0*'
_output_shapes
:€€€€€€€€€?21
/differentiable_bpsk_modulation_layer_4/SelectV2Џ
/differentiable_bpsk_modulation_layer_4/IdentityIdentity8differentiable_bpsk_modulation_layer_4/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€?21
/differentiable_bpsk_modulation_layer_4/Identity”
0differentiable_bpsk_modulation_layer_4/IdentityN	IdentityN8differentiable_bpsk_modulation_layer_4/SelectV2:output:0.differentiable_bpsk_modulation_layer_4/sub:z:0*
T
2*-
_gradient_op_typeCustomGradient-1517102*:
_output_shapes(
&:€€€€€€€€€?:€€€€€€€€€?22
0differentiable_bpsk_modulation_layer_4/IdentityNН
IdentityIdentity9differentiable_bpsk_modulation_layer_4/IdentityN:output:0*
T0*'
_output_shapes
:€€€€€€€€€?2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€-:€€€€€€€€€?:Q M
'
_output_shapes
:€€€€€€€€€-
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€?
"
_user_specified_name
inputs/1"±J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*а
serving_defaultћ
;
input_10
serving_default_input_1:0€€€€€€€€€-
;
input_20
serving_default_input_2:0€€€€€€€€€?4
output_1(
PartitionedCall:0€€€€€€€€€?tensorflow/serving/predict:•^
Ќ
	lbcpe

modulation
regularization_losses
	variables
trainable_variables
	keras_api

signatures
)__call__
**&call_and_return_all_conditional_losses
+_default_save_signature"ц
_tf_keras_model№{"class_name": "Encoder", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Encoder"}}
ы
product
	regularization_losses

	variables
trainable_variables
	keras_api
,__call__
*-&call_and_return_all_conditional_losses"я
_tf_keras_layer≈{"class_name": "LinearBlockCodeProductEncoderWithExternalG", "name": "linear_block_code_product_encoder_with_external_g_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Ў
regularization_losses
	variables
trainable_variables
	keras_api
.__call__
*/&call_and_return_all_conditional_losses"…
_tf_keras_layerѓ{"class_name": "DifferentiableBPSKModulationLayer", "name": "differentiable_bpsk_modulation_layer_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 
metrics

layers
non_trainable_variables
regularization_losses
layer_regularization_losses
	variables
layer_metrics
trainable_variables
)__call__
+_default_save_signature
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
,
0serving_default"
signature_map
‘
regularization_losses
	variables
trainable_variables
	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3f
4g"Ј
_tf_keras_layerЭ{"class_name": "ProductWithExternalWeights", "name": "product_with_external_weights_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": [{"class_name": "TensorShape", "items": [64, 45]}, {"class_name": "TensorShape", "items": [45, 63]}]}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠

layers
metrics
non_trainable_variables
	regularization_losses
layer_regularization_losses

	variables
layer_metrics
trainable_variables
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
≠

layers
 metrics
!non_trainable_variables
regularization_losses
"layer_regularization_losses
	variables
#layer_metrics
trainable_variables
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
≠

$layers
%metrics
&non_trainable_variables
regularization_losses
'layer_regularization_losses
	variables
(layer_metrics
trainable_variables
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
ж2г
)__inference_encoder_layer_call_fn_1517068
)__inference_encoder_layer_call_fn_1517162
)__inference_encoder_layer_call_fn_1517168
)__inference_encoder_layer_call_fn_1517062і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
D__inference_encoder_layer_call_and_return_conditional_losses_1517012
D__inference_encoder_layer_call_and_return_conditional_losses_1517056
D__inference_encoder_layer_call_and_return_conditional_losses_1517156
D__inference_encoder_layer_call_and_return_conditional_losses_1517112і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
И2Е
"__inference__wrapped_model_1516796ё
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *NҐK
IЪF
!К
input_1€€€€€€€€€-
!К
input_2€€€€€€€€€?
и2е
U__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_fn_1517240
U__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_fn_1517246і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ю2Ы
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_1517234
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_1517201і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ќ2Ћ
H__inference_differentiable_bpsk_modulation_layer_4_layer_call_fn_1517286
H__inference_differentiable_bpsk_modulation_layer_4_layer_call_fn_1517281і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Д2Б
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_1517276
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_1517261і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
”B–
%__inference_signature_wrapper_1516968input_1input_2"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
__inference_f_1517290
__inference_f_1517294Э
Ф≤Р
FullArgSpec
argsЪ
jself
jw
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
__inference_g_1517306
__inference_g_1517300Э
Ф≤Р
FullArgSpec
argsЪ
jself
jw
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ґ
"__inference__wrapped_model_1516796ПXҐU
NҐK
IЪF
!К
input_1€€€€€€€€€-
!К
input_2€€€€€€€€€?
™ "3™0
.
output_1"К
output_1€€€€€€€€€?√
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_1517261\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€?
p
™ "%Ґ"
К
0€€€€€€€€€?
Ъ √
c__inference_differentiable_bpsk_modulation_layer_4_layer_call_and_return_conditional_losses_1517276\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€?
p 
™ "%Ґ"
К
0€€€€€€€€€?
Ъ Ы
H__inference_differentiable_bpsk_modulation_layer_4_layer_call_fn_1517281O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€?
p
™ "К€€€€€€€€€?Ы
H__inference_differentiable_bpsk_modulation_layer_4_layer_call_fn_1517286O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€?
p 
™ "К€€€€€€€€€?ќ
D__inference_encoder_layer_call_and_return_conditional_losses_1517012Е\ҐY
RҐO
IЪF
!К
input_1€€€€€€€€€-
!К
input_2€€€€€€€€€?
p
™ "%Ґ"
К
0€€€€€€€€€?
Ъ ќ
D__inference_encoder_layer_call_and_return_conditional_losses_1517056Е\ҐY
RҐO
IЪF
!К
input_1€€€€€€€€€-
!К
input_2€€€€€€€€€?
p 
™ "%Ґ"
К
0€€€€€€€€€?
Ъ –
D__inference_encoder_layer_call_and_return_conditional_losses_1517112З^Ґ[
TҐQ
KЪH
"К
inputs/0€€€€€€€€€-
"К
inputs/1€€€€€€€€€?
p
™ "%Ґ"
К
0€€€€€€€€€?
Ъ –
D__inference_encoder_layer_call_and_return_conditional_losses_1517156З^Ґ[
TҐQ
KЪH
"К
inputs/0€€€€€€€€€-
"К
inputs/1€€€€€€€€€?
p 
™ "%Ґ"
К
0€€€€€€€€€?
Ъ •
)__inference_encoder_layer_call_fn_1517062x\ҐY
RҐO
IЪF
!К
input_1€€€€€€€€€-
!К
input_2€€€€€€€€€?
p
™ "К€€€€€€€€€?•
)__inference_encoder_layer_call_fn_1517068x\ҐY
RҐO
IЪF
!К
input_1€€€€€€€€€-
!К
input_2€€€€€€€€€?
p 
™ "К€€€€€€€€€?І
)__inference_encoder_layer_call_fn_1517162z^Ґ[
TҐQ
KЪH
"К
inputs/0€€€€€€€€€-
"К
inputs/1€€€€€€€€€?
p
™ "К€€€€€€€€€?І
)__inference_encoder_layer_call_fn_1517168z^Ґ[
TҐQ
KЪH
"К
inputs/0€€€€€€€€€-
"К
inputs/1€€€€€€€€€?
p 
™ "К€€€€€€€€€?M
__inference_f_15172904!Ґ
Ґ
К
w?-
™ "К?-_
__inference_f_1517294F*Ґ'
 Ґ
К
w?€€€€€€€€€
™ "К?€€€€€€€€€M
__inference_g_15173004!Ґ
Ґ
К
w?-
™ "К?-_
__inference_g_1517306F*Ґ'
 Ґ
К
w?€€€€€€€€€
™ "К?€€€€€€€€€ь
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_1517201З^Ґ[
TҐQ
KЪH
"К
inputs/0€€€€€€€€€-
"К
inputs/1€€€€€€€€€?
p
™ "%Ґ"
К
0€€€€€€€€€?
Ъ ь
p__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_and_return_conditional_losses_1517234З^Ґ[
TҐQ
KЪH
"К
inputs/0€€€€€€€€€-
"К
inputs/1€€€€€€€€€?
p 
™ "%Ґ"
К
0€€€€€€€€€?
Ъ ”
U__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_fn_1517240z^Ґ[
TҐQ
KЪH
"К
inputs/0€€€€€€€€€-
"К
inputs/1€€€€€€€€€?
p
™ "К€€€€€€€€€?”
U__inference_linear_block_code_product_encoder_with_external_g_4_layer_call_fn_1517246z^Ґ[
TҐQ
KЪH
"К
inputs/0€€€€€€€€€-
"К
inputs/1€€€€€€€€€?
p 
™ "К€€€€€€€€€? 
%__inference_signature_wrapper_1516968†iҐf
Ґ 
_™\
,
input_1!К
input_1€€€€€€€€€-
,
input_2!К
input_2€€€€€€€€€?"3™0
.
output_1"К
output_1€€€€€€€€€?