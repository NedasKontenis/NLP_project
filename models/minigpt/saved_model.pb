��2
�!�!
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
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
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8��.
o
Adam/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/beta/v
h
Adam/beta/v/Read/ReadVariableOpReadVariableOpAdam/beta/v*
_output_shapes	
:�*
dtype0
q
Adam/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/gamma/v
j
 Adam/gamma/v/Read/ReadVariableOpReadVariableOpAdam/gamma/v*
_output_shapes	
:�*
dtype0
o
Adam/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/bias/v
h
Adam/bias/v/Read/ReadVariableOpReadVariableOpAdam/bias/v*
_output_shapes	
:�*
dtype0
x
Adam/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_nameAdam/kernel/v
q
!Adam/kernel/v/Read/ReadVariableOpReadVariableOpAdam/kernel/v* 
_output_shapes
:
��*
dtype0
s
Adam/bias/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/bias/v_1
l
!Adam/bias/v_1/Read/ReadVariableOpReadVariableOpAdam/bias/v_1*
_output_shapes	
:�*
dtype0
|
Adam/kernel/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_nameAdam/kernel/v_1
u
#Adam/kernel/v_1/Read/ReadVariableOpReadVariableOpAdam/kernel/v_1* 
_output_shapes
:
��*
dtype0
s
Adam/beta/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/beta/v_1
l
!Adam/beta/v_1/Read/ReadVariableOpReadVariableOpAdam/beta/v_1*
_output_shapes	
:�*
dtype0
u
Adam/gamma/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/gamma/v_1
n
"Adam/gamma/v_1/Read/ReadVariableOpReadVariableOpAdam/gamma/v_1*
_output_shapes	
:�*
dtype0
�
AAdam/transformer_decoder_1/self_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*R
shared_nameCAAdam/transformer_decoder_1/self_attention/attention_output/bias/v
�
UAdam/transformer_decoder_1/self_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_decoder_1/self_attention/attention_output/bias/v*
_output_shapes	
:�*
dtype0
�
CAdam/transformer_decoder_1/self_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:U�*T
shared_nameECAdam/transformer_decoder_1/self_attention/attention_output/kernel/v
�
WAdam/transformer_decoder_1/self_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpCAdam/transformer_decoder_1/self_attention/attention_output/kernel/v*#
_output_shapes
:U�*
dtype0
�
6Adam/transformer_decoder_1/self_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*G
shared_name86Adam/transformer_decoder_1/self_attention/value/bias/v
�
JAdam/transformer_decoder_1/self_attention/value/bias/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_1/self_attention/value/bias/v*
_output_shapes

:U*
dtype0
�
8Adam/transformer_decoder_1/self_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*I
shared_name:8Adam/transformer_decoder_1/self_attention/value/kernel/v
�
LAdam/transformer_decoder_1/self_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_1/self_attention/value/kernel/v*#
_output_shapes
:�U*
dtype0
�
4Adam/transformer_decoder_1/self_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*E
shared_name64Adam/transformer_decoder_1/self_attention/key/bias/v
�
HAdam/transformer_decoder_1/self_attention/key/bias/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_decoder_1/self_attention/key/bias/v*
_output_shapes

:U*
dtype0
�
6Adam/transformer_decoder_1/self_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*G
shared_name86Adam/transformer_decoder_1/self_attention/key/kernel/v
�
JAdam/transformer_decoder_1/self_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_1/self_attention/key/kernel/v*#
_output_shapes
:�U*
dtype0
�
6Adam/transformer_decoder_1/self_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*G
shared_name86Adam/transformer_decoder_1/self_attention/query/bias/v
�
JAdam/transformer_decoder_1/self_attention/query/bias/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_1/self_attention/query/bias/v*
_output_shapes

:U*
dtype0
�
8Adam/transformer_decoder_1/self_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*I
shared_name:8Adam/transformer_decoder_1/self_attention/query/kernel/v
�
LAdam/transformer_decoder_1/self_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_1/self_attention/query/kernel/v*#
_output_shapes
:�U*
dtype0
s
Adam/beta/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/beta/v_2
l
!Adam/beta/v_2/Read/ReadVariableOpReadVariableOpAdam/beta/v_2*
_output_shapes	
:�*
dtype0
u
Adam/gamma/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/gamma/v_2
n
"Adam/gamma/v_2/Read/ReadVariableOpReadVariableOpAdam/gamma/v_2*
_output_shapes	
:�*
dtype0
s
Adam/bias/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/bias/v_2
l
!Adam/bias/v_2/Read/ReadVariableOpReadVariableOpAdam/bias/v_2*
_output_shapes	
:�*
dtype0
|
Adam/kernel/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_nameAdam/kernel/v_2
u
#Adam/kernel/v_2/Read/ReadVariableOpReadVariableOpAdam/kernel/v_2* 
_output_shapes
:
��*
dtype0
s
Adam/bias/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/bias/v_3
l
!Adam/bias/v_3/Read/ReadVariableOpReadVariableOpAdam/bias/v_3*
_output_shapes	
:�*
dtype0
|
Adam/kernel/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_nameAdam/kernel/v_3
u
#Adam/kernel/v_3/Read/ReadVariableOpReadVariableOpAdam/kernel/v_3* 
_output_shapes
:
��*
dtype0
s
Adam/beta/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/beta/v_3
l
!Adam/beta/v_3/Read/ReadVariableOpReadVariableOpAdam/beta/v_3*
_output_shapes	
:�*
dtype0
u
Adam/gamma/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/gamma/v_3
n
"Adam/gamma/v_3/Read/ReadVariableOpReadVariableOpAdam/gamma/v_3*
_output_shapes	
:�*
dtype0
�
?Adam/transformer_decoder/self_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*P
shared_nameA?Adam/transformer_decoder/self_attention/attention_output/bias/v
�
SAdam/transformer_decoder/self_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder/self_attention/attention_output/bias/v*
_output_shapes	
:�*
dtype0
�
AAdam/transformer_decoder/self_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:U�*R
shared_nameCAAdam/transformer_decoder/self_attention/attention_output/kernel/v
�
UAdam/transformer_decoder/self_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_decoder/self_attention/attention_output/kernel/v*#
_output_shapes
:U�*
dtype0
�
4Adam/transformer_decoder/self_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*E
shared_name64Adam/transformer_decoder/self_attention/value/bias/v
�
HAdam/transformer_decoder/self_attention/value/bias/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_decoder/self_attention/value/bias/v*
_output_shapes

:U*
dtype0
�
6Adam/transformer_decoder/self_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*G
shared_name86Adam/transformer_decoder/self_attention/value/kernel/v
�
JAdam/transformer_decoder/self_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder/self_attention/value/kernel/v*#
_output_shapes
:�U*
dtype0
�
2Adam/transformer_decoder/self_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*C
shared_name42Adam/transformer_decoder/self_attention/key/bias/v
�
FAdam/transformer_decoder/self_attention/key/bias/v/Read/ReadVariableOpReadVariableOp2Adam/transformer_decoder/self_attention/key/bias/v*
_output_shapes

:U*
dtype0
�
4Adam/transformer_decoder/self_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*E
shared_name64Adam/transformer_decoder/self_attention/key/kernel/v
�
HAdam/transformer_decoder/self_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_decoder/self_attention/key/kernel/v*#
_output_shapes
:�U*
dtype0
�
4Adam/transformer_decoder/self_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*E
shared_name64Adam/transformer_decoder/self_attention/query/bias/v
�
HAdam/transformer_decoder/self_attention/query/bias/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_decoder/self_attention/query/bias/v*
_output_shapes

:U*
dtype0
�
6Adam/transformer_decoder/self_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*G
shared_name86Adam/transformer_decoder/self_attention/query/kernel/v
�
JAdam/transformer_decoder/self_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder/self_attention/query/kernel/v*#
_output_shapes
:�U*
dtype0
�
.Adam/token_and_position_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*?
shared_name0.Adam/token_and_position_embedding/embeddings/v
�
BAdam/token_and_position_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp.Adam/token_and_position_embedding/embeddings/v* 
_output_shapes
:
��*
dtype0
�
0Adam/token_and_position_embedding/embeddings/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
�'�*A
shared_name20Adam/token_and_position_embedding/embeddings/v_1
�
DAdam/token_and_position_embedding/embeddings/v_1/Read/ReadVariableOpReadVariableOp0Adam/token_and_position_embedding/embeddings/v_1* 
_output_shapes
:
�'�*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�'*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:�'*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��'*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
��'*
dtype0
o
Adam/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/beta/m
h
Adam/beta/m/Read/ReadVariableOpReadVariableOpAdam/beta/m*
_output_shapes	
:�*
dtype0
q
Adam/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/gamma/m
j
 Adam/gamma/m/Read/ReadVariableOpReadVariableOpAdam/gamma/m*
_output_shapes	
:�*
dtype0
o
Adam/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/bias/m
h
Adam/bias/m/Read/ReadVariableOpReadVariableOpAdam/bias/m*
_output_shapes	
:�*
dtype0
x
Adam/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_nameAdam/kernel/m
q
!Adam/kernel/m/Read/ReadVariableOpReadVariableOpAdam/kernel/m* 
_output_shapes
:
��*
dtype0
s
Adam/bias/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/bias/m_1
l
!Adam/bias/m_1/Read/ReadVariableOpReadVariableOpAdam/bias/m_1*
_output_shapes	
:�*
dtype0
|
Adam/kernel/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_nameAdam/kernel/m_1
u
#Adam/kernel/m_1/Read/ReadVariableOpReadVariableOpAdam/kernel/m_1* 
_output_shapes
:
��*
dtype0
s
Adam/beta/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/beta/m_1
l
!Adam/beta/m_1/Read/ReadVariableOpReadVariableOpAdam/beta/m_1*
_output_shapes	
:�*
dtype0
u
Adam/gamma/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/gamma/m_1
n
"Adam/gamma/m_1/Read/ReadVariableOpReadVariableOpAdam/gamma/m_1*
_output_shapes	
:�*
dtype0
�
AAdam/transformer_decoder_1/self_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*R
shared_nameCAAdam/transformer_decoder_1/self_attention/attention_output/bias/m
�
UAdam/transformer_decoder_1/self_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_decoder_1/self_attention/attention_output/bias/m*
_output_shapes	
:�*
dtype0
�
CAdam/transformer_decoder_1/self_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:U�*T
shared_nameECAdam/transformer_decoder_1/self_attention/attention_output/kernel/m
�
WAdam/transformer_decoder_1/self_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpCAdam/transformer_decoder_1/self_attention/attention_output/kernel/m*#
_output_shapes
:U�*
dtype0
�
6Adam/transformer_decoder_1/self_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*G
shared_name86Adam/transformer_decoder_1/self_attention/value/bias/m
�
JAdam/transformer_decoder_1/self_attention/value/bias/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_1/self_attention/value/bias/m*
_output_shapes

:U*
dtype0
�
8Adam/transformer_decoder_1/self_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*I
shared_name:8Adam/transformer_decoder_1/self_attention/value/kernel/m
�
LAdam/transformer_decoder_1/self_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_1/self_attention/value/kernel/m*#
_output_shapes
:�U*
dtype0
�
4Adam/transformer_decoder_1/self_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*E
shared_name64Adam/transformer_decoder_1/self_attention/key/bias/m
�
HAdam/transformer_decoder_1/self_attention/key/bias/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_decoder_1/self_attention/key/bias/m*
_output_shapes

:U*
dtype0
�
6Adam/transformer_decoder_1/self_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*G
shared_name86Adam/transformer_decoder_1/self_attention/key/kernel/m
�
JAdam/transformer_decoder_1/self_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_1/self_attention/key/kernel/m*#
_output_shapes
:�U*
dtype0
�
6Adam/transformer_decoder_1/self_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*G
shared_name86Adam/transformer_decoder_1/self_attention/query/bias/m
�
JAdam/transformer_decoder_1/self_attention/query/bias/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_1/self_attention/query/bias/m*
_output_shapes

:U*
dtype0
�
8Adam/transformer_decoder_1/self_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*I
shared_name:8Adam/transformer_decoder_1/self_attention/query/kernel/m
�
LAdam/transformer_decoder_1/self_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_1/self_attention/query/kernel/m*#
_output_shapes
:�U*
dtype0
s
Adam/beta/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/beta/m_2
l
!Adam/beta/m_2/Read/ReadVariableOpReadVariableOpAdam/beta/m_2*
_output_shapes	
:�*
dtype0
u
Adam/gamma/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/gamma/m_2
n
"Adam/gamma/m_2/Read/ReadVariableOpReadVariableOpAdam/gamma/m_2*
_output_shapes	
:�*
dtype0
s
Adam/bias/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/bias/m_2
l
!Adam/bias/m_2/Read/ReadVariableOpReadVariableOpAdam/bias/m_2*
_output_shapes	
:�*
dtype0
|
Adam/kernel/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_nameAdam/kernel/m_2
u
#Adam/kernel/m_2/Read/ReadVariableOpReadVariableOpAdam/kernel/m_2* 
_output_shapes
:
��*
dtype0
s
Adam/bias/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/bias/m_3
l
!Adam/bias/m_3/Read/ReadVariableOpReadVariableOpAdam/bias/m_3*
_output_shapes	
:�*
dtype0
|
Adam/kernel/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_nameAdam/kernel/m_3
u
#Adam/kernel/m_3/Read/ReadVariableOpReadVariableOpAdam/kernel/m_3* 
_output_shapes
:
��*
dtype0
s
Adam/beta/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/beta/m_3
l
!Adam/beta/m_3/Read/ReadVariableOpReadVariableOpAdam/beta/m_3*
_output_shapes	
:�*
dtype0
u
Adam/gamma/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameAdam/gamma/m_3
n
"Adam/gamma/m_3/Read/ReadVariableOpReadVariableOpAdam/gamma/m_3*
_output_shapes	
:�*
dtype0
�
?Adam/transformer_decoder/self_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*P
shared_nameA?Adam/transformer_decoder/self_attention/attention_output/bias/m
�
SAdam/transformer_decoder/self_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder/self_attention/attention_output/bias/m*
_output_shapes	
:�*
dtype0
�
AAdam/transformer_decoder/self_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:U�*R
shared_nameCAAdam/transformer_decoder/self_attention/attention_output/kernel/m
�
UAdam/transformer_decoder/self_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_decoder/self_attention/attention_output/kernel/m*#
_output_shapes
:U�*
dtype0
�
4Adam/transformer_decoder/self_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*E
shared_name64Adam/transformer_decoder/self_attention/value/bias/m
�
HAdam/transformer_decoder/self_attention/value/bias/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_decoder/self_attention/value/bias/m*
_output_shapes

:U*
dtype0
�
6Adam/transformer_decoder/self_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*G
shared_name86Adam/transformer_decoder/self_attention/value/kernel/m
�
JAdam/transformer_decoder/self_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder/self_attention/value/kernel/m*#
_output_shapes
:�U*
dtype0
�
2Adam/transformer_decoder/self_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*C
shared_name42Adam/transformer_decoder/self_attention/key/bias/m
�
FAdam/transformer_decoder/self_attention/key/bias/m/Read/ReadVariableOpReadVariableOp2Adam/transformer_decoder/self_attention/key/bias/m*
_output_shapes

:U*
dtype0
�
4Adam/transformer_decoder/self_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*E
shared_name64Adam/transformer_decoder/self_attention/key/kernel/m
�
HAdam/transformer_decoder/self_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_decoder/self_attention/key/kernel/m*#
_output_shapes
:�U*
dtype0
�
4Adam/transformer_decoder/self_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*E
shared_name64Adam/transformer_decoder/self_attention/query/bias/m
�
HAdam/transformer_decoder/self_attention/query/bias/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_decoder/self_attention/query/bias/m*
_output_shapes

:U*
dtype0
�
6Adam/transformer_decoder/self_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*G
shared_name86Adam/transformer_decoder/self_attention/query/kernel/m
�
JAdam/transformer_decoder/self_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder/self_attention/query/kernel/m*#
_output_shapes
:�U*
dtype0
�
.Adam/token_and_position_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*?
shared_name0.Adam/token_and_position_embedding/embeddings/m
�
BAdam/token_and_position_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp.Adam/token_and_position_embedding/embeddings/m* 
_output_shapes
:
��*
dtype0
�
0Adam/token_and_position_embedding/embeddings/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
�'�*A
shared_name20Adam/token_and_position_embedding/embeddings/m_1
�
DAdam/token_and_position_embedding/embeddings/m_1/Read/ReadVariableOpReadVariableOp0Adam/token_and_position_embedding/embeddings/m_1* 
_output_shapes
:
�'�*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�'*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:�'*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��'*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
��'*
dtype0
v
number_of_samplesVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namenumber_of_samples
o
%number_of_samples/Read/ReadVariableOpReadVariableOpnumber_of_samples*
_output_shapes
: *
dtype0
�
aggregate_crossentropyVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameaggregate_crossentropy
y
*aggregate_crossentropy/Read/ReadVariableOpReadVariableOpaggregate_crossentropy*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
a
betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namebeta
Z
beta/Read/ReadVariableOpReadVariableOpbeta*
_output_shapes	
:�*
dtype0
c
gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namegamma
\
gamma/Read/ReadVariableOpReadVariableOpgamma*
_output_shapes	
:�*
dtype0
a
biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namebias
Z
bias/Read/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype0
j
kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namekernel
c
kernel/Read/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
��*
dtype0
e
bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namebias_1
^
bias_1/Read/ReadVariableOpReadVariableOpbias_1*
_output_shapes	
:�*
dtype0
n
kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name
kernel_1
g
kernel_1/Read/ReadVariableOpReadVariableOpkernel_1* 
_output_shapes
:
��*
dtype0
e
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namebeta_1
^
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes	
:�*
dtype0
g
gamma_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	gamma_1
`
gamma_1/Read/ReadVariableOpReadVariableOpgamma_1*
_output_shapes	
:�*
dtype0
�
:transformer_decoder_1/self_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:transformer_decoder_1/self_attention/attention_output/bias
�
Ntransformer_decoder_1/self_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp:transformer_decoder_1/self_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
<transformer_decoder_1/self_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:U�*M
shared_name><transformer_decoder_1/self_attention/attention_output/kernel
�
Ptransformer_decoder_1/self_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp<transformer_decoder_1/self_attention/attention_output/kernel*#
_output_shapes
:U�*
dtype0
�
/transformer_decoder_1/self_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*@
shared_name1/transformer_decoder_1/self_attention/value/bias
�
Ctransformer_decoder_1/self_attention/value/bias/Read/ReadVariableOpReadVariableOp/transformer_decoder_1/self_attention/value/bias*
_output_shapes

:U*
dtype0
�
1transformer_decoder_1/self_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*B
shared_name31transformer_decoder_1/self_attention/value/kernel
�
Etransformer_decoder_1/self_attention/value/kernel/Read/ReadVariableOpReadVariableOp1transformer_decoder_1/self_attention/value/kernel*#
_output_shapes
:�U*
dtype0
�
-transformer_decoder_1/self_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*>
shared_name/-transformer_decoder_1/self_attention/key/bias
�
Atransformer_decoder_1/self_attention/key/bias/Read/ReadVariableOpReadVariableOp-transformer_decoder_1/self_attention/key/bias*
_output_shapes

:U*
dtype0
�
/transformer_decoder_1/self_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*@
shared_name1/transformer_decoder_1/self_attention/key/kernel
�
Ctransformer_decoder_1/self_attention/key/kernel/Read/ReadVariableOpReadVariableOp/transformer_decoder_1/self_attention/key/kernel*#
_output_shapes
:�U*
dtype0
�
/transformer_decoder_1/self_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*@
shared_name1/transformer_decoder_1/self_attention/query/bias
�
Ctransformer_decoder_1/self_attention/query/bias/Read/ReadVariableOpReadVariableOp/transformer_decoder_1/self_attention/query/bias*
_output_shapes

:U*
dtype0
�
1transformer_decoder_1/self_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*B
shared_name31transformer_decoder_1/self_attention/query/kernel
�
Etransformer_decoder_1/self_attention/query/kernel/Read/ReadVariableOpReadVariableOp1transformer_decoder_1/self_attention/query/kernel*#
_output_shapes
:�U*
dtype0
e
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namebeta_2
^
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes	
:�*
dtype0
g
gamma_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	gamma_2
`
gamma_2/Read/ReadVariableOpReadVariableOpgamma_2*
_output_shapes	
:�*
dtype0
e
bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namebias_2
^
bias_2/Read/ReadVariableOpReadVariableOpbias_2*
_output_shapes	
:�*
dtype0
n
kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name
kernel_2
g
kernel_2/Read/ReadVariableOpReadVariableOpkernel_2* 
_output_shapes
:
��*
dtype0
e
bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namebias_3
^
bias_3/Read/ReadVariableOpReadVariableOpbias_3*
_output_shapes	
:�*
dtype0
n
kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name
kernel_3
g
kernel_3/Read/ReadVariableOpReadVariableOpkernel_3* 
_output_shapes
:
��*
dtype0
e
beta_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namebeta_3
^
beta_3/Read/ReadVariableOpReadVariableOpbeta_3*
_output_shapes	
:�*
dtype0
g
gamma_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	gamma_3
`
gamma_3/Read/ReadVariableOpReadVariableOpgamma_3*
_output_shapes	
:�*
dtype0
�
8transformer_decoder/self_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*I
shared_name:8transformer_decoder/self_attention/attention_output/bias
�
Ltransformer_decoder/self_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp8transformer_decoder/self_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
:transformer_decoder/self_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:U�*K
shared_name<:transformer_decoder/self_attention/attention_output/kernel
�
Ntransformer_decoder/self_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp:transformer_decoder/self_attention/attention_output/kernel*#
_output_shapes
:U�*
dtype0
�
-transformer_decoder/self_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*>
shared_name/-transformer_decoder/self_attention/value/bias
�
Atransformer_decoder/self_attention/value/bias/Read/ReadVariableOpReadVariableOp-transformer_decoder/self_attention/value/bias*
_output_shapes

:U*
dtype0
�
/transformer_decoder/self_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*@
shared_name1/transformer_decoder/self_attention/value/kernel
�
Ctransformer_decoder/self_attention/value/kernel/Read/ReadVariableOpReadVariableOp/transformer_decoder/self_attention/value/kernel*#
_output_shapes
:�U*
dtype0
�
+transformer_decoder/self_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*<
shared_name-+transformer_decoder/self_attention/key/bias
�
?transformer_decoder/self_attention/key/bias/Read/ReadVariableOpReadVariableOp+transformer_decoder/self_attention/key/bias*
_output_shapes

:U*
dtype0
�
-transformer_decoder/self_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*>
shared_name/-transformer_decoder/self_attention/key/kernel
�
Atransformer_decoder/self_attention/key/kernel/Read/ReadVariableOpReadVariableOp-transformer_decoder/self_attention/key/kernel*#
_output_shapes
:�U*
dtype0
�
-transformer_decoder/self_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:U*>
shared_name/-transformer_decoder/self_attention/query/bias
�
Atransformer_decoder/self_attention/query/bias/Read/ReadVariableOpReadVariableOp-transformer_decoder/self_attention/query/bias*
_output_shapes

:U*
dtype0
�
/transformer_decoder/self_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�U*@
shared_name1/transformer_decoder/self_attention/query/kernel
�
Ctransformer_decoder/self_attention/query/kernel/Read/ReadVariableOpReadVariableOp/transformer_decoder/self_attention/query/kernel*#
_output_shapes
:�U*
dtype0
�
'token_and_position_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*8
shared_name)'token_and_position_embedding/embeddings
�
;token_and_position_embedding/embeddings/Read/ReadVariableOpReadVariableOp'token_and_position_embedding/embeddings* 
_output_shapes
:
��*
dtype0
�
)token_and_position_embedding/embeddings_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
�'�*:
shared_name+)token_and_position_embedding/embeddings_1
�
=token_and_position_embedding/embeddings_1/Read/ReadVariableOpReadVariableOp)token_and_position_embedding/embeddings_1* 
_output_shapes
:
�'�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�'*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�'*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��'*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��'*
dtype0
�
serving_default_input_1Placeholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)token_and_position_embedding/embeddings_1'token_and_position_embedding/embeddings/transformer_decoder/self_attention/query/kernel-transformer_decoder/self_attention/query/bias-transformer_decoder/self_attention/key/kernel+transformer_decoder/self_attention/key/bias/transformer_decoder/self_attention/value/kernel-transformer_decoder/self_attention/value/bias:transformer_decoder/self_attention/attention_output/kernel8transformer_decoder/self_attention/attention_output/biasgamma_3beta_3kernel_3bias_3kernel_2bias_2gamma_2beta_21transformer_decoder_1/self_attention/query/kernel/transformer_decoder_1/self_attention/query/bias/transformer_decoder_1/self_attention/key/kernel-transformer_decoder_1/self_attention/key/bias1transformer_decoder_1/self_attention/value/kernel/transformer_decoder_1/self_attention/value/bias<transformer_decoder_1/self_attention/attention_output/kernel:transformer_decoder_1/self_attention/attention_output/biasgamma_1beta_1kernel_1bias_1kernelbiasgammabetadense/kernel
dense/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_44953

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�Bߒ Bג
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
position_embedding*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_self_attention_layer
_self_attention_layer_norm
_self_attention_dropout
# _feedforward_intermediate_dense
!_feedforward_output_dense
"_feedforward_layer_norm
#_feedforward_dropout*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*_self_attention_layer
+_self_attention_layer_norm
,_self_attention_dropout
#-_feedforward_intermediate_dense
._feedforward_output_dense
/_feedforward_layer_norm
0_feedforward_dropout*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
�
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
Y32
Z33
734
835*
�
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
Y32
Z33
734
835*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
`trace_0
atrace_1
btrace_2
ctrace_3* 
6
dtrace_0
etrace_1
ftrace_2
gtrace_3* 
* 
�
hiter

ibeta_1

jbeta_2
	kdecay
llearning_rate7m�8m�9m�:m�;m�<m�=m�>m�?m�@m�Am�Bm�Cm�Dm�Em�Fm�Gm�Hm�Im�Jm�Km�Lm�Mm�Nm�Om�Pm�Qm�Rm�Sm�Tm�Um�Vm�Wm�Xm�Ym�Zm�7v�8v�9v�:v�;v�<v�=v�>v�?v�@v�Av�Bv�Cv�Dv�Ev�Fv�Gv�Hv�Iv�Jv�Kv�Lv�Mv�Nv�Ov�Pv�Qv�Rv�Sv�Tv�Uv�Vv�Wv�Xv�Yv�Zv�*

mserving_default* 

90
:1*

90
:1*
* 
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

strace_0* 

ttrace_0* 
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
9
embeddings*
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
:
embeddings
:position_embeddings*
z
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15*
z
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Cgamma
Dbeta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ekernel
Fbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Gkernel
Hbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Igamma
Jbeta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
z
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12
X13
Y14
Z15*
z
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12
X13
Y14
Z15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Sgamma
Tbeta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ukernel
Vbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Wkernel
Xbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Ygamma
Zbeta*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 

70
81*

70
81*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)token_and_position_embedding/embeddings_1&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'token_and_position_embedding/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/transformer_decoder/self_attention/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-transformer_decoder/self_attention/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-transformer_decoder/self_attention/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE+transformer_decoder/self_attention/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/transformer_decoder/self_attention/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-transformer_decoder/self_attention/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:transformer_decoder/self_attention/attention_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8transformer_decoder/self_attention/attention_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEgamma_3'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEbeta_3'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEkernel_3'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEbias_3'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEkernel_2'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEbias_2'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEgamma_2'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEbeta_2'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1transformer_decoder_1/self_attention/query/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_decoder_1/self_attention/query/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_decoder_1/self_attention/key/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-transformer_decoder_1/self_attention/key/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1transformer_decoder_1/self_attention/value/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_decoder_1/self_attention/value/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<transformer_decoder_1/self_attention/attention_output/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:transformer_decoder_1/self_attention/attention_output/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEgamma_1'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEbeta_1'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEkernel_1'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEbias_1'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUEkernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
E?
VARIABLE_VALUEbias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
F@
VARIABLE_VALUEgamma'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
E?
VARIABLE_VALUEbeta'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

0
1*
* 
* 
* 
* 
* 

90*

90*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 

:0*

:0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
5
0
1
2
 3
!4
"5
#6*
* 
* 
* 
* 
* 
* 
* 
<
;0
<1
=2
>3
?4
@5
A6
B7*
<
;0
<1
=2
>3
?4
@5
A6
B7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

;kernel
<bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

=kernel
>bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

?kernel
@bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Akernel
Bbias*

C0
D1*

C0
D1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

E0
F1*

E0
F1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

G0
H1*

G0
H1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

I0
J1*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
5
*0
+1
,2
-3
.4
/5
06*
* 
* 
* 
* 
* 
* 
* 
<
K0
L1
M2
N3
O4
P5
Q6
R7*
<
K0
L1
M2
N3
O4
P5
Q6
R7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Kkernel
Lbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Mkernel
Nbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Okernel
Pbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Qkernel
Rbias*

S0
T1*

S0
T1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

U0
V1*

U0
V1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

W0
X1*

W0
X1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
�
�	variables
�	keras_api
�aggregate_crossentropy
�_aggregate_crossentropy
�number_of_samples
�_number_of_samples*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

;0
<1*

;0
<1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

=0
>1*

=0
>1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

?0
@1*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

A0
B1*

A0
B1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
�0
�1
�2
�3
�4
�5*
* 
* 
* 

K0
L1*

K0
L1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

M0
N1*

M0
N1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

O0
P1*

O0
P1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

Q0
R1*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
uo
VARIABLE_VALUEaggregate_crossentropyEkeras_api/metrics/1/aggregate_crossentropy/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEnumber_of_samples@keras_api/metrics/1/number_of_samples/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE0Adam/token_and_position_embedding/embeddings/m_1Bvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE.Adam/token_and_position_embedding/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder/self_attention/query/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/transformer_decoder/self_attention/query/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/transformer_decoder/self_attention/key/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE2Adam/transformer_decoder/self_attention/key/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder/self_attention/value/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/transformer_decoder/self_attention/value/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAAdam/transformer_decoder/self_attention/attention_output/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE?Adam/transformer_decoder/self_attention/attention_output/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/gamma/m_3Cvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/beta/m_3Cvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEAdam/kernel/m_3Cvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/bias/m_3Cvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEAdam/kernel/m_2Cvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/bias/m_2Cvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/gamma/m_2Cvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/beta/m_2Cvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE8Adam/transformer_decoder_1/self_attention/query/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder_1/self_attention/query/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder_1/self_attention/key/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/transformer_decoder_1/self_attention/key/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE8Adam/transformer_decoder_1/self_attention/value/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder_1/self_attention/value/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUECAdam/transformer_decoder_1/self_attention/attention_output/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAAdam/transformer_decoder_1/self_attention/attention_output/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/gamma/m_1Cvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/beta/m_1Cvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEAdam/kernel/m_1Cvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/bias/m_1Cvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/gamma/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/beta/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE0Adam/token_and_position_embedding/embeddings/v_1Bvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE.Adam/token_and_position_embedding/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder/self_attention/query/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/transformer_decoder/self_attention/query/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/transformer_decoder/self_attention/key/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE2Adam/transformer_decoder/self_attention/key/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder/self_attention/value/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/transformer_decoder/self_attention/value/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAAdam/transformer_decoder/self_attention/attention_output/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE?Adam/transformer_decoder/self_attention/attention_output/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/gamma/v_3Cvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/beta/v_3Cvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEAdam/kernel/v_3Cvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/bias/v_3Cvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEAdam/kernel/v_2Cvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/bias/v_2Cvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/gamma/v_2Cvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/beta/v_2Cvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE8Adam/transformer_decoder_1/self_attention/query/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder_1/self_attention/query/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder_1/self_attention/key/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE4Adam/transformer_decoder_1/self_attention/key/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE8Adam/transformer_decoder_1/self_attention/value/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/transformer_decoder_1/self_attention/value/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUECAdam/transformer_decoder_1/self_attention/attention_output/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAAdam/transformer_decoder_1/self_attention/attention_output/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/gamma/v_1Cvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/beta/v_1Cvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEAdam/kernel/v_1Cvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/bias/v_1Cvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/gamma/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/beta/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�3
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp=token_and_position_embedding/embeddings_1/Read/ReadVariableOp;token_and_position_embedding/embeddings/Read/ReadVariableOpCtransformer_decoder/self_attention/query/kernel/Read/ReadVariableOpAtransformer_decoder/self_attention/query/bias/Read/ReadVariableOpAtransformer_decoder/self_attention/key/kernel/Read/ReadVariableOp?transformer_decoder/self_attention/key/bias/Read/ReadVariableOpCtransformer_decoder/self_attention/value/kernel/Read/ReadVariableOpAtransformer_decoder/self_attention/value/bias/Read/ReadVariableOpNtransformer_decoder/self_attention/attention_output/kernel/Read/ReadVariableOpLtransformer_decoder/self_attention/attention_output/bias/Read/ReadVariableOpgamma_3/Read/ReadVariableOpbeta_3/Read/ReadVariableOpkernel_3/Read/ReadVariableOpbias_3/Read/ReadVariableOpkernel_2/Read/ReadVariableOpbias_2/Read/ReadVariableOpgamma_2/Read/ReadVariableOpbeta_2/Read/ReadVariableOpEtransformer_decoder_1/self_attention/query/kernel/Read/ReadVariableOpCtransformer_decoder_1/self_attention/query/bias/Read/ReadVariableOpCtransformer_decoder_1/self_attention/key/kernel/Read/ReadVariableOpAtransformer_decoder_1/self_attention/key/bias/Read/ReadVariableOpEtransformer_decoder_1/self_attention/value/kernel/Read/ReadVariableOpCtransformer_decoder_1/self_attention/value/bias/Read/ReadVariableOpPtransformer_decoder_1/self_attention/attention_output/kernel/Read/ReadVariableOpNtransformer_decoder_1/self_attention/attention_output/bias/Read/ReadVariableOpgamma_1/Read/ReadVariableOpbeta_1/Read/ReadVariableOpkernel_1/Read/ReadVariableOpbias_1/Read/ReadVariableOpkernel/Read/ReadVariableOpbias/Read/ReadVariableOpgamma/Read/ReadVariableOpbeta/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*aggregate_crossentropy/Read/ReadVariableOp%number_of_samples/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOpDAdam/token_and_position_embedding/embeddings/m_1/Read/ReadVariableOpBAdam/token_and_position_embedding/embeddings/m/Read/ReadVariableOpJAdam/transformer_decoder/self_attention/query/kernel/m/Read/ReadVariableOpHAdam/transformer_decoder/self_attention/query/bias/m/Read/ReadVariableOpHAdam/transformer_decoder/self_attention/key/kernel/m/Read/ReadVariableOpFAdam/transformer_decoder/self_attention/key/bias/m/Read/ReadVariableOpJAdam/transformer_decoder/self_attention/value/kernel/m/Read/ReadVariableOpHAdam/transformer_decoder/self_attention/value/bias/m/Read/ReadVariableOpUAdam/transformer_decoder/self_attention/attention_output/kernel/m/Read/ReadVariableOpSAdam/transformer_decoder/self_attention/attention_output/bias/m/Read/ReadVariableOp"Adam/gamma/m_3/Read/ReadVariableOp!Adam/beta/m_3/Read/ReadVariableOp#Adam/kernel/m_3/Read/ReadVariableOp!Adam/bias/m_3/Read/ReadVariableOp#Adam/kernel/m_2/Read/ReadVariableOp!Adam/bias/m_2/Read/ReadVariableOp"Adam/gamma/m_2/Read/ReadVariableOp!Adam/beta/m_2/Read/ReadVariableOpLAdam/transformer_decoder_1/self_attention/query/kernel/m/Read/ReadVariableOpJAdam/transformer_decoder_1/self_attention/query/bias/m/Read/ReadVariableOpJAdam/transformer_decoder_1/self_attention/key/kernel/m/Read/ReadVariableOpHAdam/transformer_decoder_1/self_attention/key/bias/m/Read/ReadVariableOpLAdam/transformer_decoder_1/self_attention/value/kernel/m/Read/ReadVariableOpJAdam/transformer_decoder_1/self_attention/value/bias/m/Read/ReadVariableOpWAdam/transformer_decoder_1/self_attention/attention_output/kernel/m/Read/ReadVariableOpUAdam/transformer_decoder_1/self_attention/attention_output/bias/m/Read/ReadVariableOp"Adam/gamma/m_1/Read/ReadVariableOp!Adam/beta/m_1/Read/ReadVariableOp#Adam/kernel/m_1/Read/ReadVariableOp!Adam/bias/m_1/Read/ReadVariableOp!Adam/kernel/m/Read/ReadVariableOpAdam/bias/m/Read/ReadVariableOp Adam/gamma/m/Read/ReadVariableOpAdam/beta/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpDAdam/token_and_position_embedding/embeddings/v_1/Read/ReadVariableOpBAdam/token_and_position_embedding/embeddings/v/Read/ReadVariableOpJAdam/transformer_decoder/self_attention/query/kernel/v/Read/ReadVariableOpHAdam/transformer_decoder/self_attention/query/bias/v/Read/ReadVariableOpHAdam/transformer_decoder/self_attention/key/kernel/v/Read/ReadVariableOpFAdam/transformer_decoder/self_attention/key/bias/v/Read/ReadVariableOpJAdam/transformer_decoder/self_attention/value/kernel/v/Read/ReadVariableOpHAdam/transformer_decoder/self_attention/value/bias/v/Read/ReadVariableOpUAdam/transformer_decoder/self_attention/attention_output/kernel/v/Read/ReadVariableOpSAdam/transformer_decoder/self_attention/attention_output/bias/v/Read/ReadVariableOp"Adam/gamma/v_3/Read/ReadVariableOp!Adam/beta/v_3/Read/ReadVariableOp#Adam/kernel/v_3/Read/ReadVariableOp!Adam/bias/v_3/Read/ReadVariableOp#Adam/kernel/v_2/Read/ReadVariableOp!Adam/bias/v_2/Read/ReadVariableOp"Adam/gamma/v_2/Read/ReadVariableOp!Adam/beta/v_2/Read/ReadVariableOpLAdam/transformer_decoder_1/self_attention/query/kernel/v/Read/ReadVariableOpJAdam/transformer_decoder_1/self_attention/query/bias/v/Read/ReadVariableOpJAdam/transformer_decoder_1/self_attention/key/kernel/v/Read/ReadVariableOpHAdam/transformer_decoder_1/self_attention/key/bias/v/Read/ReadVariableOpLAdam/transformer_decoder_1/self_attention/value/kernel/v/Read/ReadVariableOpJAdam/transformer_decoder_1/self_attention/value/bias/v/Read/ReadVariableOpWAdam/transformer_decoder_1/self_attention/attention_output/kernel/v/Read/ReadVariableOpUAdam/transformer_decoder_1/self_attention/attention_output/bias/v/Read/ReadVariableOp"Adam/gamma/v_1/Read/ReadVariableOp!Adam/beta/v_1/Read/ReadVariableOp#Adam/kernel/v_1/Read/ReadVariableOp!Adam/bias/v_1/Read/ReadVariableOp!Adam/kernel/v/Read/ReadVariableOpAdam/bias/v/Read/ReadVariableOp Adam/gamma/v/Read/ReadVariableOpAdam/beta/v/Read/ReadVariableOpConst*�
Tin{
y2w	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_47226
� 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias)token_and_position_embedding/embeddings_1'token_and_position_embedding/embeddings/transformer_decoder/self_attention/query/kernel-transformer_decoder/self_attention/query/bias-transformer_decoder/self_attention/key/kernel+transformer_decoder/self_attention/key/bias/transformer_decoder/self_attention/value/kernel-transformer_decoder/self_attention/value/bias:transformer_decoder/self_attention/attention_output/kernel8transformer_decoder/self_attention/attention_output/biasgamma_3beta_3kernel_3bias_3kernel_2bias_2gamma_2beta_21transformer_decoder_1/self_attention/query/kernel/transformer_decoder_1/self_attention/query/bias/transformer_decoder_1/self_attention/key/kernel-transformer_decoder_1/self_attention/key/bias1transformer_decoder_1/self_attention/value/kernel/transformer_decoder_1/self_attention/value/bias<transformer_decoder_1/self_attention/attention_output/kernel:transformer_decoder_1/self_attention/attention_output/biasgamma_1beta_1kernel_1bias_1kernelbiasgammabeta	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountaggregate_crossentropynumber_of_samplesAdam/dense/kernel/mAdam/dense/bias/m0Adam/token_and_position_embedding/embeddings/m_1.Adam/token_and_position_embedding/embeddings/m6Adam/transformer_decoder/self_attention/query/kernel/m4Adam/transformer_decoder/self_attention/query/bias/m4Adam/transformer_decoder/self_attention/key/kernel/m2Adam/transformer_decoder/self_attention/key/bias/m6Adam/transformer_decoder/self_attention/value/kernel/m4Adam/transformer_decoder/self_attention/value/bias/mAAdam/transformer_decoder/self_attention/attention_output/kernel/m?Adam/transformer_decoder/self_attention/attention_output/bias/mAdam/gamma/m_3Adam/beta/m_3Adam/kernel/m_3Adam/bias/m_3Adam/kernel/m_2Adam/bias/m_2Adam/gamma/m_2Adam/beta/m_28Adam/transformer_decoder_1/self_attention/query/kernel/m6Adam/transformer_decoder_1/self_attention/query/bias/m6Adam/transformer_decoder_1/self_attention/key/kernel/m4Adam/transformer_decoder_1/self_attention/key/bias/m8Adam/transformer_decoder_1/self_attention/value/kernel/m6Adam/transformer_decoder_1/self_attention/value/bias/mCAdam/transformer_decoder_1/self_attention/attention_output/kernel/mAAdam/transformer_decoder_1/self_attention/attention_output/bias/mAdam/gamma/m_1Adam/beta/m_1Adam/kernel/m_1Adam/bias/m_1Adam/kernel/mAdam/bias/mAdam/gamma/mAdam/beta/mAdam/dense/kernel/vAdam/dense/bias/v0Adam/token_and_position_embedding/embeddings/v_1.Adam/token_and_position_embedding/embeddings/v6Adam/transformer_decoder/self_attention/query/kernel/v4Adam/transformer_decoder/self_attention/query/bias/v4Adam/transformer_decoder/self_attention/key/kernel/v2Adam/transformer_decoder/self_attention/key/bias/v6Adam/transformer_decoder/self_attention/value/kernel/v4Adam/transformer_decoder/self_attention/value/bias/vAAdam/transformer_decoder/self_attention/attention_output/kernel/v?Adam/transformer_decoder/self_attention/attention_output/bias/vAdam/gamma/v_3Adam/beta/v_3Adam/kernel/v_3Adam/bias/v_3Adam/kernel/v_2Adam/bias/v_2Adam/gamma/v_2Adam/beta/v_28Adam/transformer_decoder_1/self_attention/query/kernel/v6Adam/transformer_decoder_1/self_attention/query/bias/v6Adam/transformer_decoder_1/self_attention/key/kernel/v4Adam/transformer_decoder_1/self_attention/key/bias/v8Adam/transformer_decoder_1/self_attention/value/kernel/v6Adam/transformer_decoder_1/self_attention/value/bias/vCAdam/transformer_decoder_1/self_attention/attention_output/kernel/vAAdam/transformer_decoder_1/self_attention/attention_output/bias/vAdam/gamma/v_1Adam/beta/v_1Adam/kernel/v_1Adam/bias/v_1Adam/kernel/vAdam/bias/vAdam/gamma/vAdam/beta/v*�
Tinz
x2v*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_47587�)
��
�
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_43729
decoder_sequenceQ
:self_attention_query_einsum_einsum_readvariableop_resource:�UB
0self_attention_query_add_readvariableop_resource:UO
8self_attention_key_einsum_einsum_readvariableop_resource:�U@
.self_attention_key_add_readvariableop_resource:UQ
:self_attention_value_einsum_einsum_readvariableop_resource:�UB
0self_attention_value_add_readvariableop_resource:U\
Eself_attention_attention_output_einsum_einsum_readvariableop_resource:U�J
;self_attention_attention_output_add_readvariableop_resource:	�N
?self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�J
;self_attention_layer_norm_batchnorm_readvariableop_resource:	�T
@feedforward_intermediate_dense_tensordot_readvariableop_resource:
��M
>feedforward_intermediate_dense_biasadd_readvariableop_resource:	�N
:feedforward_output_dense_tensordot_readvariableop_resource:
��G
8feedforward_output_dense_biasadd_readvariableop_resource:	�K
<feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�G
8feedforward_layer_norm_batchnorm_readvariableop_resource:	�
identity��5feedforward_intermediate_dense/BiasAdd/ReadVariableOp�7feedforward_intermediate_dense/Tensordot/ReadVariableOp�/feedforward_layer_norm/batchnorm/ReadVariableOp�3feedforward_layer_norm/batchnorm/mul/ReadVariableOp�/feedforward_output_dense/BiasAdd/ReadVariableOp�1feedforward_output_dense/Tensordot/ReadVariableOp�2self_attention/attention_output/add/ReadVariableOp�<self_attention/attention_output/einsum/Einsum/ReadVariableOp�%self_attention/key/add/ReadVariableOp�/self_attention/key/einsum/Einsum/ReadVariableOp�'self_attention/query/add/ReadVariableOp�1self_attention/query/einsum/Einsum/ReadVariableOp�'self_attention/value/add/ReadVariableOp�1self_attention/value/einsum/Einsum/ReadVariableOp�2self_attention_layer_norm/batchnorm/ReadVariableOp�6self_attention_layer_norm/batchnorm/mul/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
Shape_1Shapedecoder_sequence*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\

range/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: {
rangeRangerange/start:output:0range/Cast:y:0range/delta:output:0*

Tidx0*#
_output_shapes
:���������H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B : M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
addAddV2range:output:0Cast:y:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :l

ExpandDims
ExpandDimsadd:z:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������R
range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    R
range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
range_1/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
range_1Rangerange_1/start:output:0range_1/Cast:y:0range_1/delta:output:0*

Tidx0*#
_output_shapes
:���������~
GreaterEqualGreaterEqualExpandDims:output:0range_1:output:0*
T0*0
_output_shapes
:������������������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsGreaterEqual:z:0ExpandDims_1/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
packedPackstrided_slice:output:0strided_slice_3:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :�
BroadcastToBroadcastToExpandDims_1:output:0packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
1self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/query/einsum/EinsumEinsumdecoder_sequence9self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/query/add/ReadVariableOpReadVariableOp0self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/query/addAddV2+self_attention/query/einsum/Einsum:output:0/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp8self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
 self_attention/key/einsum/EinsumEinsumdecoder_sequence7self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
%self_attention/key/add/ReadVariableOpReadVariableOp.self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/key/addAddV2)self_attention/key/einsum/Einsum:output:0-self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
1self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/value/einsum/EinsumEinsumdecoder_sequence9self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/value/add/ReadVariableOpReadVariableOp0self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/value/addAddV2+self_attention/value/einsum/Einsum:output:0/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������UW
self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Uk
self_attention/CastCastself_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: U
self_attention/SqrtSqrtself_attention/Cast:y:0*
T0*
_output_shapes
: ]
self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
self_attention/truedivRealDiv!self_attention/truediv/x:output:0self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
self_attention/MulMulself_attention/query/add:z:0self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
self_attention/einsum/EinsumEinsumself_attention/key/add:z:0self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbeh
self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
self_attention/ExpandDims
ExpandDimsBroadcastTo:output:0&self_attention/ExpandDims/dim:output:0*
T0
*A
_output_shapes/
-:+����������������������������
self_attention/softmax_1/CastCast"self_attention/ExpandDims:output:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+���������������������������c
self_attention/softmax_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
self_attention/softmax_1/subSub'self_attention/softmax_1/sub/x:output:0!self_attention/softmax_1/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������c
self_attention/softmax_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
self_attention/softmax_1/mulMul self_attention/softmax_1/sub:z:0'self_attention/softmax_1/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax_1/addAddV2%self_attention/einsum/Einsum:output:0 self_attention/softmax_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
 self_attention/softmax_1/SoftmaxSoftmax self_attention/softmax_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
!self_attention/dropout_1/IdentityIdentity*self_attention/softmax_1/Softmax:softmax:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/einsum_1/EinsumEinsum*self_attention/dropout_1/Identity:output:0self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
<self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpEself_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
-self_attention/attention_output/einsum/EinsumEinsum'self_attention/einsum_1/Einsum:output:0Dself_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
2self_attention/attention_output/add/ReadVariableOpReadVariableOp;self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#self_attention/attention_output/addAddV26self_attention/attention_output/einsum/Einsum:output:0:self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
self_attention_dropout/IdentityIdentity'self_attention/attention_output/add:z:0*
T0*5
_output_shapes#
!:��������������������
add_1AddV2(self_attention_dropout/Identity:output:0decoder_sequence*
T0*5
_output_shapes#
!:��������������������
8self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&self_attention_layer_norm/moments/meanMean	add_1:z:0Aself_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
.self_attention_layer_norm/moments/StopGradientStopGradient/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
3self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference	add_1:z:07self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
<self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
*self_attention_layer_norm/moments/varianceMean7self_attention_layer_norm/moments/SquaredDifference:z:0Eself_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(n
)self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
'self_attention_layer_norm/batchnorm/addAddV23self_attention_layer_norm/moments/variance:output:02self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
)self_attention_layer_norm/batchnorm/RsqrtRsqrt+self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
6self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp?self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/mulMul-self_attention_layer_norm/batchnorm/Rsqrt:y:0>self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_1Mul	add_1:z:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_2Mul/self_attention_layer_norm/moments/mean:output:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
2self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOp;self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/subSub:self_attention_layer_norm/batchnorm/ReadVariableOp:value:0-self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/add_1AddV2-self_attention_layer_norm/batchnorm/mul_1:z:0+self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
7feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOp@feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
-feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
.feedforward_intermediate_dense/Tensordot/ShapeShape-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:x
6feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/GatherV2GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/free:output:0?feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/axes:output:0Afeedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
-feedforward_intermediate_dense/Tensordot/ProdProd:feedforward_intermediate_dense/Tensordot/GatherV2:output:07feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
/feedforward_intermediate_dense/Tensordot/Prod_1Prod<feedforward_intermediate_dense/Tensordot/GatherV2_1:output:09feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/feedforward_intermediate_dense/Tensordot/concatConcatV26feedforward_intermediate_dense/Tensordot/free:output:06feedforward_intermediate_dense/Tensordot/axes:output:0=feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
.feedforward_intermediate_dense/Tensordot/stackPack6feedforward_intermediate_dense/Tensordot/Prod:output:08feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
2feedforward_intermediate_dense/Tensordot/transpose	Transpose-self_attention_layer_norm/batchnorm/add_1:z:08feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
0feedforward_intermediate_dense/Tensordot/ReshapeReshape6feedforward_intermediate_dense/Tensordot/transpose:y:07feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
/feedforward_intermediate_dense/Tensordot/MatMulMatMul9feedforward_intermediate_dense/Tensordot/Reshape:output:0?feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
0feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�x
6feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/concat_1ConcatV2:feedforward_intermediate_dense/Tensordot/GatherV2:output:09feedforward_intermediate_dense/Tensordot/Const_2:output:0?feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_intermediate_dense/TensordotReshape9feedforward_intermediate_dense/Tensordot/MatMul:product:0:feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
5feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOp>feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&feedforward_intermediate_dense/BiasAddBiasAdd1feedforward_intermediate_dense/Tensordot:output:0=feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
#feedforward_intermediate_dense/ReluRelu/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
1feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOp:feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0q
'feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(feedforward_output_dense/Tensordot/ShapeShape1feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:r
0feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/GatherV2GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/free:output:09feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-feedforward_output_dense/Tensordot/GatherV2_1GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/axes:output:0;feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'feedforward_output_dense/Tensordot/ProdProd4feedforward_output_dense/Tensordot/GatherV2:output:01feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)feedforward_output_dense/Tensordot/Prod_1Prod6feedforward_output_dense/Tensordot/GatherV2_1:output:03feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)feedforward_output_dense/Tensordot/concatConcatV20feedforward_output_dense/Tensordot/free:output:00feedforward_output_dense/Tensordot/axes:output:07feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_output_dense/Tensordot/stackPack0feedforward_output_dense/Tensordot/Prod:output:02feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,feedforward_output_dense/Tensordot/transpose	Transpose1feedforward_intermediate_dense/Relu:activations:02feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
*feedforward_output_dense/Tensordot/ReshapeReshape0feedforward_output_dense/Tensordot/transpose:y:01feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)feedforward_output_dense/Tensordot/MatMulMatMul3feedforward_output_dense/Tensordot/Reshape:output:09feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
*feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�r
0feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/concat_1ConcatV24feedforward_output_dense/Tensordot/GatherV2:output:03feedforward_output_dense/Tensordot/Const_2:output:09feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"feedforward_output_dense/TensordotReshape3feedforward_output_dense/Tensordot/MatMul:product:04feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOp8feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 feedforward_output_dense/BiasAddBiasAdd+feedforward_output_dense/Tensordot:output:07feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
feedforward_dropout/IdentityIdentity)feedforward_output_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
add_2AddV2%feedforward_dropout/Identity:output:0-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:�������������������
5feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#feedforward_layer_norm/moments/meanMean	add_2:z:0>feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
+feedforward_layer_norm/moments/StopGradientStopGradient,feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
0feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference	add_2:z:04feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
9feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'feedforward_layer_norm/moments/varianceMean4feedforward_layer_norm/moments/SquaredDifference:z:0Bfeedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(k
&feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$feedforward_layer_norm/batchnorm/addAddV20feedforward_layer_norm/moments/variance:output:0/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
&feedforward_layer_norm/batchnorm/RsqrtRsqrt(feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
3feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp<feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/mulMul*feedforward_layer_norm/batchnorm/Rsqrt:y:0;feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_1Mul	add_2:z:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_2Mul,feedforward_layer_norm/moments/mean:output:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOp8feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/subSub7feedforward_layer_norm/batchnorm/ReadVariableOp:value:0*feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/add_1AddV2*feedforward_layer_norm/batchnorm/mul_1:z:0(feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentity*feedforward_layer_norm/batchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp6^feedforward_intermediate_dense/BiasAdd/ReadVariableOp8^feedforward_intermediate_dense/Tensordot/ReadVariableOp0^feedforward_layer_norm/batchnorm/ReadVariableOp4^feedforward_layer_norm/batchnorm/mul/ReadVariableOp0^feedforward_output_dense/BiasAdd/ReadVariableOp2^feedforward_output_dense/Tensordot/ReadVariableOp3^self_attention/attention_output/add/ReadVariableOp=^self_attention/attention_output/einsum/Einsum/ReadVariableOp&^self_attention/key/add/ReadVariableOp0^self_attention/key/einsum/Einsum/ReadVariableOp(^self_attention/query/add/ReadVariableOp2^self_attention/query/einsum/Einsum/ReadVariableOp(^self_attention/value/add/ReadVariableOp2^self_attention/value/einsum/Einsum/ReadVariableOp3^self_attention_layer_norm/batchnorm/ReadVariableOp7^self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 2n
5feedforward_intermediate_dense/BiasAdd/ReadVariableOp5feedforward_intermediate_dense/BiasAdd/ReadVariableOp2r
7feedforward_intermediate_dense/Tensordot/ReadVariableOp7feedforward_intermediate_dense/Tensordot/ReadVariableOp2b
/feedforward_layer_norm/batchnorm/ReadVariableOp/feedforward_layer_norm/batchnorm/ReadVariableOp2j
3feedforward_layer_norm/batchnorm/mul/ReadVariableOp3feedforward_layer_norm/batchnorm/mul/ReadVariableOp2b
/feedforward_output_dense/BiasAdd/ReadVariableOp/feedforward_output_dense/BiasAdd/ReadVariableOp2f
1feedforward_output_dense/Tensordot/ReadVariableOp1feedforward_output_dense/Tensordot/ReadVariableOp2h
2self_attention/attention_output/add/ReadVariableOp2self_attention/attention_output/add/ReadVariableOp2|
<self_attention/attention_output/einsum/Einsum/ReadVariableOp<self_attention/attention_output/einsum/Einsum/ReadVariableOp2N
%self_attention/key/add/ReadVariableOp%self_attention/key/add/ReadVariableOp2b
/self_attention/key/einsum/Einsum/ReadVariableOp/self_attention/key/einsum/Einsum/ReadVariableOp2R
'self_attention/query/add/ReadVariableOp'self_attention/query/add/ReadVariableOp2f
1self_attention/query/einsum/Einsum/ReadVariableOp1self_attention/query/einsum/Einsum/ReadVariableOp2R
'self_attention/value/add/ReadVariableOp'self_attention/value/add/ReadVariableOp2f
1self_attention/value/einsum/Einsum/ReadVariableOp1self_attention/value/einsum/Einsum/ReadVariableOp2h
2self_attention_layer_norm/batchnorm/ReadVariableOp2self_attention_layer_norm/batchnorm/ReadVariableOp2p
6self_attention_layer_norm/batchnorm/mul/ReadVariableOp6self_attention_layer_norm/batchnorm/mul/ReadVariableOp:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
��
�
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_46813
decoder_sequenceQ
:self_attention_query_einsum_einsum_readvariableop_resource:�UB
0self_attention_query_add_readvariableop_resource:UO
8self_attention_key_einsum_einsum_readvariableop_resource:�U@
.self_attention_key_add_readvariableop_resource:UQ
:self_attention_value_einsum_einsum_readvariableop_resource:�UB
0self_attention_value_add_readvariableop_resource:U\
Eself_attention_attention_output_einsum_einsum_readvariableop_resource:U�J
;self_attention_attention_output_add_readvariableop_resource:	�N
?self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�J
;self_attention_layer_norm_batchnorm_readvariableop_resource:	�T
@feedforward_intermediate_dense_tensordot_readvariableop_resource:
��M
>feedforward_intermediate_dense_biasadd_readvariableop_resource:	�N
:feedforward_output_dense_tensordot_readvariableop_resource:
��G
8feedforward_output_dense_biasadd_readvariableop_resource:	�K
<feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�G
8feedforward_layer_norm_batchnorm_readvariableop_resource:	�
identity��5feedforward_intermediate_dense/BiasAdd/ReadVariableOp�7feedforward_intermediate_dense/Tensordot/ReadVariableOp�/feedforward_layer_norm/batchnorm/ReadVariableOp�3feedforward_layer_norm/batchnorm/mul/ReadVariableOp�/feedforward_output_dense/BiasAdd/ReadVariableOp�1feedforward_output_dense/Tensordot/ReadVariableOp�2self_attention/attention_output/add/ReadVariableOp�<self_attention/attention_output/einsum/Einsum/ReadVariableOp�%self_attention/key/add/ReadVariableOp�/self_attention/key/einsum/Einsum/ReadVariableOp�'self_attention/query/add/ReadVariableOp�1self_attention/query/einsum/Einsum/ReadVariableOp�'self_attention/value/add/ReadVariableOp�1self_attention/value/einsum/Einsum/ReadVariableOp�2self_attention_layer_norm/batchnorm/ReadVariableOp�6self_attention_layer_norm/batchnorm/mul/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
Shape_1Shapedecoder_sequence*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\

range/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: {
rangeRangerange/start:output:0range/Cast:y:0range/delta:output:0*

Tidx0*#
_output_shapes
:���������H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B : M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
addAddV2range:output:0Cast:y:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :l

ExpandDims
ExpandDimsadd:z:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������R
range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    R
range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
range_1/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
range_1Rangerange_1/start:output:0range_1/Cast:y:0range_1/delta:output:0*

Tidx0*#
_output_shapes
:���������~
GreaterEqualGreaterEqualExpandDims:output:0range_1:output:0*
T0*0
_output_shapes
:������������������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsGreaterEqual:z:0ExpandDims_1/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
packedPackstrided_slice:output:0strided_slice_3:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :�
BroadcastToBroadcastToExpandDims_1:output:0packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
1self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/query/einsum/EinsumEinsumdecoder_sequence9self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/query/add/ReadVariableOpReadVariableOp0self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/query/addAddV2+self_attention/query/einsum/Einsum:output:0/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp8self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
 self_attention/key/einsum/EinsumEinsumdecoder_sequence7self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
%self_attention/key/add/ReadVariableOpReadVariableOp.self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/key/addAddV2)self_attention/key/einsum/Einsum:output:0-self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
1self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/value/einsum/EinsumEinsumdecoder_sequence9self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/value/add/ReadVariableOpReadVariableOp0self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/value/addAddV2+self_attention/value/einsum/Einsum:output:0/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������UW
self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Uk
self_attention/CastCastself_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: U
self_attention/SqrtSqrtself_attention/Cast:y:0*
T0*
_output_shapes
: ]
self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
self_attention/truedivRealDiv!self_attention/truediv/x:output:0self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
self_attention/MulMulself_attention/query/add:z:0self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
self_attention/einsum/EinsumEinsumself_attention/key/add:z:0self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbeh
self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
self_attention/ExpandDims
ExpandDimsBroadcastTo:output:0&self_attention/ExpandDims/dim:output:0*
T0
*A
_output_shapes/
-:+����������������������������
self_attention/softmax_1/CastCast"self_attention/ExpandDims:output:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+���������������������������c
self_attention/softmax_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
self_attention/softmax_1/subSub'self_attention/softmax_1/sub/x:output:0!self_attention/softmax_1/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������c
self_attention/softmax_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
self_attention/softmax_1/mulMul self_attention/softmax_1/sub:z:0'self_attention/softmax_1/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax_1/addAddV2%self_attention/einsum/Einsum:output:0 self_attention/softmax_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
 self_attention/softmax_1/SoftmaxSoftmax self_attention/softmax_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/einsum_1/EinsumEinsum*self_attention/softmax_1/Softmax:softmax:0self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
<self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpEself_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
-self_attention/attention_output/einsum/EinsumEinsum'self_attention/einsum_1/Einsum:output:0Dself_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
2self_attention/attention_output/add/ReadVariableOpReadVariableOp;self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#self_attention/attention_output/addAddV26self_attention/attention_output/einsum/Einsum:output:0:self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
add_1AddV2'self_attention/attention_output/add:z:0decoder_sequence*
T0*5
_output_shapes#
!:��������������������
8self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&self_attention_layer_norm/moments/meanMean	add_1:z:0Aself_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
.self_attention_layer_norm/moments/StopGradientStopGradient/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
3self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference	add_1:z:07self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
<self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
*self_attention_layer_norm/moments/varianceMean7self_attention_layer_norm/moments/SquaredDifference:z:0Eself_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(n
)self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
'self_attention_layer_norm/batchnorm/addAddV23self_attention_layer_norm/moments/variance:output:02self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
)self_attention_layer_norm/batchnorm/RsqrtRsqrt+self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
6self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp?self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/mulMul-self_attention_layer_norm/batchnorm/Rsqrt:y:0>self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_1Mul	add_1:z:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_2Mul/self_attention_layer_norm/moments/mean:output:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
2self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOp;self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/subSub:self_attention_layer_norm/batchnorm/ReadVariableOp:value:0-self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/add_1AddV2-self_attention_layer_norm/batchnorm/mul_1:z:0+self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
7feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOp@feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
-feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
.feedforward_intermediate_dense/Tensordot/ShapeShape-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:x
6feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/GatherV2GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/free:output:0?feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/axes:output:0Afeedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
-feedforward_intermediate_dense/Tensordot/ProdProd:feedforward_intermediate_dense/Tensordot/GatherV2:output:07feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
/feedforward_intermediate_dense/Tensordot/Prod_1Prod<feedforward_intermediate_dense/Tensordot/GatherV2_1:output:09feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/feedforward_intermediate_dense/Tensordot/concatConcatV26feedforward_intermediate_dense/Tensordot/free:output:06feedforward_intermediate_dense/Tensordot/axes:output:0=feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
.feedforward_intermediate_dense/Tensordot/stackPack6feedforward_intermediate_dense/Tensordot/Prod:output:08feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
2feedforward_intermediate_dense/Tensordot/transpose	Transpose-self_attention_layer_norm/batchnorm/add_1:z:08feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
0feedforward_intermediate_dense/Tensordot/ReshapeReshape6feedforward_intermediate_dense/Tensordot/transpose:y:07feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
/feedforward_intermediate_dense/Tensordot/MatMulMatMul9feedforward_intermediate_dense/Tensordot/Reshape:output:0?feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
0feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�x
6feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/concat_1ConcatV2:feedforward_intermediate_dense/Tensordot/GatherV2:output:09feedforward_intermediate_dense/Tensordot/Const_2:output:0?feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_intermediate_dense/TensordotReshape9feedforward_intermediate_dense/Tensordot/MatMul:product:0:feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
5feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOp>feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&feedforward_intermediate_dense/BiasAddBiasAdd1feedforward_intermediate_dense/Tensordot:output:0=feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
#feedforward_intermediate_dense/ReluRelu/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
1feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOp:feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0q
'feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(feedforward_output_dense/Tensordot/ShapeShape1feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:r
0feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/GatherV2GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/free:output:09feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-feedforward_output_dense/Tensordot/GatherV2_1GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/axes:output:0;feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'feedforward_output_dense/Tensordot/ProdProd4feedforward_output_dense/Tensordot/GatherV2:output:01feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)feedforward_output_dense/Tensordot/Prod_1Prod6feedforward_output_dense/Tensordot/GatherV2_1:output:03feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)feedforward_output_dense/Tensordot/concatConcatV20feedforward_output_dense/Tensordot/free:output:00feedforward_output_dense/Tensordot/axes:output:07feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_output_dense/Tensordot/stackPack0feedforward_output_dense/Tensordot/Prod:output:02feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,feedforward_output_dense/Tensordot/transpose	Transpose1feedforward_intermediate_dense/Relu:activations:02feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
*feedforward_output_dense/Tensordot/ReshapeReshape0feedforward_output_dense/Tensordot/transpose:y:01feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)feedforward_output_dense/Tensordot/MatMulMatMul3feedforward_output_dense/Tensordot/Reshape:output:09feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
*feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�r
0feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/concat_1ConcatV24feedforward_output_dense/Tensordot/GatherV2:output:03feedforward_output_dense/Tensordot/Const_2:output:09feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"feedforward_output_dense/TensordotReshape3feedforward_output_dense/Tensordot/MatMul:product:04feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOp8feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 feedforward_output_dense/BiasAddBiasAdd+feedforward_output_dense/Tensordot:output:07feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
add_2AddV2)feedforward_output_dense/BiasAdd:output:0-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:�������������������
5feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#feedforward_layer_norm/moments/meanMean	add_2:z:0>feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
+feedforward_layer_norm/moments/StopGradientStopGradient,feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
0feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference	add_2:z:04feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
9feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'feedforward_layer_norm/moments/varianceMean4feedforward_layer_norm/moments/SquaredDifference:z:0Bfeedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(k
&feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$feedforward_layer_norm/batchnorm/addAddV20feedforward_layer_norm/moments/variance:output:0/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
&feedforward_layer_norm/batchnorm/RsqrtRsqrt(feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
3feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp<feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/mulMul*feedforward_layer_norm/batchnorm/Rsqrt:y:0;feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_1Mul	add_2:z:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_2Mul,feedforward_layer_norm/moments/mean:output:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOp8feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/subSub7feedforward_layer_norm/batchnorm/ReadVariableOp:value:0*feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/add_1AddV2*feedforward_layer_norm/batchnorm/mul_1:z:0(feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentity*feedforward_layer_norm/batchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp6^feedforward_intermediate_dense/BiasAdd/ReadVariableOp8^feedforward_intermediate_dense/Tensordot/ReadVariableOp0^feedforward_layer_norm/batchnorm/ReadVariableOp4^feedforward_layer_norm/batchnorm/mul/ReadVariableOp0^feedforward_output_dense/BiasAdd/ReadVariableOp2^feedforward_output_dense/Tensordot/ReadVariableOp3^self_attention/attention_output/add/ReadVariableOp=^self_attention/attention_output/einsum/Einsum/ReadVariableOp&^self_attention/key/add/ReadVariableOp0^self_attention/key/einsum/Einsum/ReadVariableOp(^self_attention/query/add/ReadVariableOp2^self_attention/query/einsum/Einsum/ReadVariableOp(^self_attention/value/add/ReadVariableOp2^self_attention/value/einsum/Einsum/ReadVariableOp3^self_attention_layer_norm/batchnorm/ReadVariableOp7^self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 2n
5feedforward_intermediate_dense/BiasAdd/ReadVariableOp5feedforward_intermediate_dense/BiasAdd/ReadVariableOp2r
7feedforward_intermediate_dense/Tensordot/ReadVariableOp7feedforward_intermediate_dense/Tensordot/ReadVariableOp2b
/feedforward_layer_norm/batchnorm/ReadVariableOp/feedforward_layer_norm/batchnorm/ReadVariableOp2j
3feedforward_layer_norm/batchnorm/mul/ReadVariableOp3feedforward_layer_norm/batchnorm/mul/ReadVariableOp2b
/feedforward_output_dense/BiasAdd/ReadVariableOp/feedforward_output_dense/BiasAdd/ReadVariableOp2f
1feedforward_output_dense/Tensordot/ReadVariableOp1feedforward_output_dense/Tensordot/ReadVariableOp2h
2self_attention/attention_output/add/ReadVariableOp2self_attention/attention_output/add/ReadVariableOp2|
<self_attention/attention_output/einsum/Einsum/ReadVariableOp<self_attention/attention_output/einsum/Einsum/ReadVariableOp2N
%self_attention/key/add/ReadVariableOp%self_attention/key/add/ReadVariableOp2b
/self_attention/key/einsum/Einsum/ReadVariableOp/self_attention/key/einsum/Einsum/ReadVariableOp2R
'self_attention/query/add/ReadVariableOp'self_attention/query/add/ReadVariableOp2f
1self_attention/query/einsum/Einsum/ReadVariableOp1self_attention/query/einsum/Einsum/ReadVariableOp2R
'self_attention/value/add/ReadVariableOp'self_attention/value/add/ReadVariableOp2f
1self_attention/value/einsum/Einsum/ReadVariableOp1self_attention/value/einsum/Einsum/ReadVariableOp2h
2self_attention_layer_norm/batchnorm/ReadVariableOp2self_attention_layer_norm/batchnorm/ReadVariableOp2p
6self_attention_layer_norm/batchnorm/mul/ReadVariableOp6self_attention_layer_norm/batchnorm/mul/ReadVariableOp:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
�
�	
%__inference_model_layer_call_fn_45030

inputs
unknown:
�'�
	unknown_0:
�� 
	unknown_1:�U
	unknown_2:U 
	unknown_3:�U
	unknown_4:U 
	unknown_5:�U
	unknown_6:U 
	unknown_7:U�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�!

unknown_17:�U

unknown_18:U!

unknown_19:�U

unknown_20:U!

unknown_21:�U

unknown_22:U!

unknown_23:U�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:
��

unknown_28:	�

unknown_29:
��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:
��'

unknown_34:	�'
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_43800}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�	
#__inference_signature_wrapper_44953
input_1
unknown:
�'�
	unknown_0:
�� 
	unknown_1:�U
	unknown_2:U 
	unknown_3:�U
	unknown_4:U 
	unknown_5:�U
	unknown_6:U 
	unknown_7:U�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�!

unknown_17:�U

unknown_18:U!

unknown_19:�U

unknown_20:U!

unknown_21:�U

unknown_22:U!

unknown_23:U�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:
��

unknown_28:	�

unknown_29:
��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:
��'

unknown_34:	�'
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_43297}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:������������������
!
_user_specified_name	input_1
�
�
3__inference_transformer_decoder_layer_call_fn_46004
decoder_sequence
unknown:�U
	unknown_0:U 
	unknown_1:�U
	unknown_2:U 
	unknown_3:�U
	unknown_4:U 
	unknown_5:U�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldecoder_sequenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_43519}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
�.
�
@__inference_model_layer_call_and_return_conditional_losses_43800

inputs6
"token_and_position_embedding_43336:
�'�6
"token_and_position_embedding_43338:
��0
transformer_decoder_43520:�U+
transformer_decoder_43522:U0
transformer_decoder_43524:�U+
transformer_decoder_43526:U0
transformer_decoder_43528:�U+
transformer_decoder_43530:U0
transformer_decoder_43532:U�(
transformer_decoder_43534:	�(
transformer_decoder_43536:	�(
transformer_decoder_43538:	�-
transformer_decoder_43540:
��(
transformer_decoder_43542:	�-
transformer_decoder_43544:
��(
transformer_decoder_43546:	�(
transformer_decoder_43548:	�(
transformer_decoder_43550:	�2
transformer_decoder_1_43730:�U-
transformer_decoder_1_43732:U2
transformer_decoder_1_43734:�U-
transformer_decoder_1_43736:U2
transformer_decoder_1_43738:�U-
transformer_decoder_1_43740:U2
transformer_decoder_1_43742:U�*
transformer_decoder_1_43744:	�*
transformer_decoder_1_43746:	�*
transformer_decoder_1_43748:	�/
transformer_decoder_1_43750:
��*
transformer_decoder_1_43752:	�/
transformer_decoder_1_43754:
��*
transformer_decoder_1_43756:	�*
transformer_decoder_1_43758:	�*
transformer_decoder_1_43760:	�
dense_43794:
��'
dense_43796:	�'
identity��dense/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�+transformer_decoder/StatefulPartitionedCall�-transformer_decoder_1/StatefulPartitionedCall�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs"token_and_position_embedding_43336"token_and_position_embedding_43338*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_43335i
'token_and_position_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
%token_and_position_embedding/NotEqualNotEqualinputs0token_and_position_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:�������������������
+transformer_decoder/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_decoder_43520transformer_decoder_43522transformer_decoder_43524transformer_decoder_43526transformer_decoder_43528transformer_decoder_43530transformer_decoder_43532transformer_decoder_43534transformer_decoder_43536transformer_decoder_43538transformer_decoder_43540transformer_decoder_43542transformer_decoder_43544transformer_decoder_43546transformer_decoder_43548transformer_decoder_43550*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_43519�
-transformer_decoder_1/StatefulPartitionedCallStatefulPartitionedCall4transformer_decoder/StatefulPartitionedCall:output:0transformer_decoder_1_43730transformer_decoder_1_43732transformer_decoder_1_43734transformer_decoder_1_43736transformer_decoder_1_43738transformer_decoder_1_43740transformer_decoder_1_43742transformer_decoder_1_43744transformer_decoder_1_43746transformer_decoder_1_43748transformer_decoder_1_43750transformer_decoder_1_43752transformer_decoder_1_43754transformer_decoder_1_43756transformer_decoder_1_43758transformer_decoder_1_43760*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_43729�
dense/StatefulPartitionedCallStatefulPartitionedCall6transformer_decoder_1/StatefulPartitionedCall:output:0dense_43794dense_43796*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43793�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'�
NoOpNoOp^dense/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_decoder/StatefulPartitionedCall.^transformer_decoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_decoder/StatefulPartitionedCall+transformer_decoder/StatefulPartitionedCall2^
-transformer_decoder_1/StatefulPartitionedCall-transformer_decoder_1/StatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
��
�T
!__inference__traced_restore_47587
file_prefix1
assignvariableop_dense_kernel:
��',
assignvariableop_1_dense_bias:	�'P
<assignvariableop_2_token_and_position_embedding_embeddings_1:
�'�N
:assignvariableop_3_token_and_position_embedding_embeddings:
��Y
Bassignvariableop_4_transformer_decoder_self_attention_query_kernel:�UR
@assignvariableop_5_transformer_decoder_self_attention_query_bias:UW
@assignvariableop_6_transformer_decoder_self_attention_key_kernel:�UP
>assignvariableop_7_transformer_decoder_self_attention_key_bias:UY
Bassignvariableop_8_transformer_decoder_self_attention_value_kernel:�UR
@assignvariableop_9_transformer_decoder_self_attention_value_bias:Ue
Nassignvariableop_10_transformer_decoder_self_attention_attention_output_kernel:U�[
Lassignvariableop_11_transformer_decoder_self_attention_attention_output_bias:	�*
assignvariableop_12_gamma_3:	�)
assignvariableop_13_beta_3:	�0
assignvariableop_14_kernel_3:
��)
assignvariableop_15_bias_3:	�0
assignvariableop_16_kernel_2:
��)
assignvariableop_17_bias_2:	�*
assignvariableop_18_gamma_2:	�)
assignvariableop_19_beta_2:	�\
Eassignvariableop_20_transformer_decoder_1_self_attention_query_kernel:�UU
Cassignvariableop_21_transformer_decoder_1_self_attention_query_bias:UZ
Cassignvariableop_22_transformer_decoder_1_self_attention_key_kernel:�US
Aassignvariableop_23_transformer_decoder_1_self_attention_key_bias:U\
Eassignvariableop_24_transformer_decoder_1_self_attention_value_kernel:�UU
Cassignvariableop_25_transformer_decoder_1_self_attention_value_bias:Ug
Passignvariableop_26_transformer_decoder_1_self_attention_attention_output_kernel:U�]
Nassignvariableop_27_transformer_decoder_1_self_attention_attention_output_bias:	�*
assignvariableop_28_gamma_1:	�)
assignvariableop_29_beta_1:	�0
assignvariableop_30_kernel_1:
��)
assignvariableop_31_bias_1:	�.
assignvariableop_32_kernel:
��'
assignvariableop_33_bias:	�(
assignvariableop_34_gamma:	�'
assignvariableop_35_beta:	�'
assignvariableop_36_adam_iter:	 )
assignvariableop_37_adam_beta_1: )
assignvariableop_38_adam_beta_2: (
assignvariableop_39_adam_decay: 0
&assignvariableop_40_adam_learning_rate: #
assignvariableop_41_total: #
assignvariableop_42_count: 4
*assignvariableop_43_aggregate_crossentropy: /
%assignvariableop_44_number_of_samples: ;
'assignvariableop_45_adam_dense_kernel_m:
��'4
%assignvariableop_46_adam_dense_bias_m:	�'X
Dassignvariableop_47_adam_token_and_position_embedding_embeddings_m_1:
�'�V
Bassignvariableop_48_adam_token_and_position_embedding_embeddings_m:
��a
Jassignvariableop_49_adam_transformer_decoder_self_attention_query_kernel_m:�UZ
Hassignvariableop_50_adam_transformer_decoder_self_attention_query_bias_m:U_
Hassignvariableop_51_adam_transformer_decoder_self_attention_key_kernel_m:�UX
Fassignvariableop_52_adam_transformer_decoder_self_attention_key_bias_m:Ua
Jassignvariableop_53_adam_transformer_decoder_self_attention_value_kernel_m:�UZ
Hassignvariableop_54_adam_transformer_decoder_self_attention_value_bias_m:Ul
Uassignvariableop_55_adam_transformer_decoder_self_attention_attention_output_kernel_m:U�b
Sassignvariableop_56_adam_transformer_decoder_self_attention_attention_output_bias_m:	�1
"assignvariableop_57_adam_gamma_m_3:	�0
!assignvariableop_58_adam_beta_m_3:	�7
#assignvariableop_59_adam_kernel_m_3:
��0
!assignvariableop_60_adam_bias_m_3:	�7
#assignvariableop_61_adam_kernel_m_2:
��0
!assignvariableop_62_adam_bias_m_2:	�1
"assignvariableop_63_adam_gamma_m_2:	�0
!assignvariableop_64_adam_beta_m_2:	�c
Lassignvariableop_65_adam_transformer_decoder_1_self_attention_query_kernel_m:�U\
Jassignvariableop_66_adam_transformer_decoder_1_self_attention_query_bias_m:Ua
Jassignvariableop_67_adam_transformer_decoder_1_self_attention_key_kernel_m:�UZ
Hassignvariableop_68_adam_transformer_decoder_1_self_attention_key_bias_m:Uc
Lassignvariableop_69_adam_transformer_decoder_1_self_attention_value_kernel_m:�U\
Jassignvariableop_70_adam_transformer_decoder_1_self_attention_value_bias_m:Un
Wassignvariableop_71_adam_transformer_decoder_1_self_attention_attention_output_kernel_m:U�d
Uassignvariableop_72_adam_transformer_decoder_1_self_attention_attention_output_bias_m:	�1
"assignvariableop_73_adam_gamma_m_1:	�0
!assignvariableop_74_adam_beta_m_1:	�7
#assignvariableop_75_adam_kernel_m_1:
��0
!assignvariableop_76_adam_bias_m_1:	�5
!assignvariableop_77_adam_kernel_m:
��.
assignvariableop_78_adam_bias_m:	�/
 assignvariableop_79_adam_gamma_m:	�.
assignvariableop_80_adam_beta_m:	�;
'assignvariableop_81_adam_dense_kernel_v:
��'4
%assignvariableop_82_adam_dense_bias_v:	�'X
Dassignvariableop_83_adam_token_and_position_embedding_embeddings_v_1:
�'�V
Bassignvariableop_84_adam_token_and_position_embedding_embeddings_v:
��a
Jassignvariableop_85_adam_transformer_decoder_self_attention_query_kernel_v:�UZ
Hassignvariableop_86_adam_transformer_decoder_self_attention_query_bias_v:U_
Hassignvariableop_87_adam_transformer_decoder_self_attention_key_kernel_v:�UX
Fassignvariableop_88_adam_transformer_decoder_self_attention_key_bias_v:Ua
Jassignvariableop_89_adam_transformer_decoder_self_attention_value_kernel_v:�UZ
Hassignvariableop_90_adam_transformer_decoder_self_attention_value_bias_v:Ul
Uassignvariableop_91_adam_transformer_decoder_self_attention_attention_output_kernel_v:U�b
Sassignvariableop_92_adam_transformer_decoder_self_attention_attention_output_bias_v:	�1
"assignvariableop_93_adam_gamma_v_3:	�0
!assignvariableop_94_adam_beta_v_3:	�7
#assignvariableop_95_adam_kernel_v_3:
��0
!assignvariableop_96_adam_bias_v_3:	�7
#assignvariableop_97_adam_kernel_v_2:
��0
!assignvariableop_98_adam_bias_v_2:	�1
"assignvariableop_99_adam_gamma_v_2:	�1
"assignvariableop_100_adam_beta_v_2:	�d
Massignvariableop_101_adam_transformer_decoder_1_self_attention_query_kernel_v:�U]
Kassignvariableop_102_adam_transformer_decoder_1_self_attention_query_bias_v:Ub
Kassignvariableop_103_adam_transformer_decoder_1_self_attention_key_kernel_v:�U[
Iassignvariableop_104_adam_transformer_decoder_1_self_attention_key_bias_v:Ud
Massignvariableop_105_adam_transformer_decoder_1_self_attention_value_kernel_v:�U]
Kassignvariableop_106_adam_transformer_decoder_1_self_attention_value_bias_v:Uo
Xassignvariableop_107_adam_transformer_decoder_1_self_attention_attention_output_kernel_v:U�e
Vassignvariableop_108_adam_transformer_decoder_1_self_attention_attention_output_bias_v:	�2
#assignvariableop_109_adam_gamma_v_1:	�1
"assignvariableop_110_adam_beta_v_1:	�8
$assignvariableop_111_adam_kernel_v_1:
��1
"assignvariableop_112_adam_bias_v_1:	�6
"assignvariableop_113_adam_kernel_v:
��/
 assignvariableop_114_adam_bias_v:	�0
!assignvariableop_115_adam_gamma_v:	�/
 assignvariableop_116_adam_beta_v:	�
identity_118��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�7
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:v*
dtype0*�6
value�6B�6vB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBEkeras_api/metrics/1/aggregate_crossentropy/.ATTRIBUTES/VARIABLE_VALUEB@keras_api/metrics/1/number_of_samples/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:v*
dtype0*�
value�B�vB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypesz
x2v	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp<assignvariableop_2_token_and_position_embedding_embeddings_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp:assignvariableop_3_token_and_position_embedding_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpBassignvariableop_4_transformer_decoder_self_attention_query_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp@assignvariableop_5_transformer_decoder_self_attention_query_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp@assignvariableop_6_transformer_decoder_self_attention_key_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp>assignvariableop_7_transformer_decoder_self_attention_key_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpBassignvariableop_8_transformer_decoder_self_attention_value_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp@assignvariableop_9_transformer_decoder_self_attention_value_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpNassignvariableop_10_transformer_decoder_self_attention_attention_output_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpLassignvariableop_11_transformer_decoder_self_attention_attention_output_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_gamma_3Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_beta_3Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_kernel_3Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_bias_3Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_kernel_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_bias_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_gamma_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_beta_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpEassignvariableop_20_transformer_decoder_1_self_attention_query_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpCassignvariableop_21_transformer_decoder_1_self_attention_query_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpCassignvariableop_22_transformer_decoder_1_self_attention_key_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpAassignvariableop_23_transformer_decoder_1_self_attention_key_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpEassignvariableop_24_transformer_decoder_1_self_attention_value_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpCassignvariableop_25_transformer_decoder_1_self_attention_value_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpPassignvariableop_26_transformer_decoder_1_self_attention_attention_output_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpNassignvariableop_27_transformer_decoder_1_self_attention_attention_output_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_gamma_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_beta_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_kernel_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_bias_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_iterIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_beta_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_beta_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_decayIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_learning_rateIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_aggregate_crossentropyIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp%assignvariableop_44_number_of_samplesIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_dense_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpDassignvariableop_47_adam_token_and_position_embedding_embeddings_m_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpBassignvariableop_48_adam_token_and_position_embedding_embeddings_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpJassignvariableop_49_adam_transformer_decoder_self_attention_query_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpHassignvariableop_50_adam_transformer_decoder_self_attention_query_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpHassignvariableop_51_adam_transformer_decoder_self_attention_key_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpFassignvariableop_52_adam_transformer_decoder_self_attention_key_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpJassignvariableop_53_adam_transformer_decoder_self_attention_value_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpHassignvariableop_54_adam_transformer_decoder_self_attention_value_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpUassignvariableop_55_adam_transformer_decoder_self_attention_attention_output_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpSassignvariableop_56_adam_transformer_decoder_self_attention_attention_output_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp"assignvariableop_57_adam_gamma_m_3Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp!assignvariableop_58_adam_beta_m_3Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp#assignvariableop_59_adam_kernel_m_3Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp!assignvariableop_60_adam_bias_m_3Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp#assignvariableop_61_adam_kernel_m_2Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp!assignvariableop_62_adam_bias_m_2Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp"assignvariableop_63_adam_gamma_m_2Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp!assignvariableop_64_adam_beta_m_2Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpLassignvariableop_65_adam_transformer_decoder_1_self_attention_query_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOpJassignvariableop_66_adam_transformer_decoder_1_self_attention_query_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpJassignvariableop_67_adam_transformer_decoder_1_self_attention_key_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpHassignvariableop_68_adam_transformer_decoder_1_self_attention_key_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOpLassignvariableop_69_adam_transformer_decoder_1_self_attention_value_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOpJassignvariableop_70_adam_transformer_decoder_1_self_attention_value_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOpWassignvariableop_71_adam_transformer_decoder_1_self_attention_attention_output_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOpUassignvariableop_72_adam_transformer_decoder_1_self_attention_attention_output_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp"assignvariableop_73_adam_gamma_m_1Identity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp!assignvariableop_74_adam_beta_m_1Identity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp#assignvariableop_75_adam_kernel_m_1Identity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp!assignvariableop_76_adam_bias_m_1Identity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp!assignvariableop_77_adam_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpassignvariableop_78_adam_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp assignvariableop_79_adam_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOpassignvariableop_80_adam_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp'assignvariableop_81_adam_dense_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp%assignvariableop_82_adam_dense_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOpDassignvariableop_83_adam_token_and_position_embedding_embeddings_v_1Identity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOpBassignvariableop_84_adam_token_and_position_embedding_embeddings_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOpJassignvariableop_85_adam_transformer_decoder_self_attention_query_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpHassignvariableop_86_adam_transformer_decoder_self_attention_query_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOpHassignvariableop_87_adam_transformer_decoder_self_attention_key_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOpFassignvariableop_88_adam_transformer_decoder_self_attention_key_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOpJassignvariableop_89_adam_transformer_decoder_self_attention_value_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOpHassignvariableop_90_adam_transformer_decoder_self_attention_value_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOpUassignvariableop_91_adam_transformer_decoder_self_attention_attention_output_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOpSassignvariableop_92_adam_transformer_decoder_self_attention_attention_output_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp"assignvariableop_93_adam_gamma_v_3Identity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp!assignvariableop_94_adam_beta_v_3Identity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp#assignvariableop_95_adam_kernel_v_3Identity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp!assignvariableop_96_adam_bias_v_3Identity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp#assignvariableop_97_adam_kernel_v_2Identity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp!assignvariableop_98_adam_bias_v_2Identity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp"assignvariableop_99_adam_gamma_v_2Identity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp"assignvariableop_100_adam_beta_v_2Identity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOpMassignvariableop_101_adam_transformer_decoder_1_self_attention_query_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOpKassignvariableop_102_adam_transformer_decoder_1_self_attention_query_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOpKassignvariableop_103_adam_transformer_decoder_1_self_attention_key_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOpIassignvariableop_104_adam_transformer_decoder_1_self_attention_key_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOpMassignvariableop_105_adam_transformer_decoder_1_self_attention_value_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOpKassignvariableop_106_adam_transformer_decoder_1_self_attention_value_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOpXassignvariableop_107_adam_transformer_decoder_1_self_attention_attention_output_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOpVassignvariableop_108_adam_transformer_decoder_1_self_attention_attention_output_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp#assignvariableop_109_adam_gamma_v_1Identity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp"assignvariableop_110_adam_beta_v_1Identity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp$assignvariableop_111_adam_kernel_v_1Identity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp"assignvariableop_112_adam_bias_v_1Identity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp"assignvariableop_113_adam_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp assignvariableop_114_adam_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp!assignvariableop_115_adam_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp assignvariableop_116_adam_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_117Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_118IdentityIdentity_117:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_118Identity_118:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�.
�
@__inference_model_layer_call_and_return_conditional_losses_44868
input_16
"token_and_position_embedding_44789:
�'�6
"token_and_position_embedding_44791:
��0
transformer_decoder_44796:�U+
transformer_decoder_44798:U0
transformer_decoder_44800:�U+
transformer_decoder_44802:U0
transformer_decoder_44804:�U+
transformer_decoder_44806:U0
transformer_decoder_44808:U�(
transformer_decoder_44810:	�(
transformer_decoder_44812:	�(
transformer_decoder_44814:	�-
transformer_decoder_44816:
��(
transformer_decoder_44818:	�-
transformer_decoder_44820:
��(
transformer_decoder_44822:	�(
transformer_decoder_44824:	�(
transformer_decoder_44826:	�2
transformer_decoder_1_44829:�U-
transformer_decoder_1_44831:U2
transformer_decoder_1_44833:�U-
transformer_decoder_1_44835:U2
transformer_decoder_1_44837:�U-
transformer_decoder_1_44839:U2
transformer_decoder_1_44841:U�*
transformer_decoder_1_44843:	�*
transformer_decoder_1_44845:	�*
transformer_decoder_1_44847:	�/
transformer_decoder_1_44849:
��*
transformer_decoder_1_44851:	�/
transformer_decoder_1_44853:
��*
transformer_decoder_1_44855:	�*
transformer_decoder_1_44857:	�*
transformer_decoder_1_44859:	�
dense_44862:
��'
dense_44864:	�'
identity��dense/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�+transformer_decoder/StatefulPartitionedCall�-transformer_decoder_1/StatefulPartitionedCall�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1"token_and_position_embedding_44789"token_and_position_embedding_44791*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_43335i
'token_and_position_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
%token_and_position_embedding/NotEqualNotEqualinput_10token_and_position_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:�������������������
+transformer_decoder/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_decoder_44796transformer_decoder_44798transformer_decoder_44800transformer_decoder_44802transformer_decoder_44804transformer_decoder_44806transformer_decoder_44808transformer_decoder_44810transformer_decoder_44812transformer_decoder_44814transformer_decoder_44816transformer_decoder_44818transformer_decoder_44820transformer_decoder_44822transformer_decoder_44824transformer_decoder_44826*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_44346�
-transformer_decoder_1/StatefulPartitionedCallStatefulPartitionedCall4transformer_decoder/StatefulPartitionedCall:output:0transformer_decoder_1_44829transformer_decoder_1_44831transformer_decoder_1_44833transformer_decoder_1_44835transformer_decoder_1_44837transformer_decoder_1_44839transformer_decoder_1_44841transformer_decoder_1_44843transformer_decoder_1_44845transformer_decoder_1_44847transformer_decoder_1_44849transformer_decoder_1_44851transformer_decoder_1_44853transformer_decoder_1_44855transformer_decoder_1_44857transformer_decoder_1_44859*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_44098�
dense/StatefulPartitionedCallStatefulPartitionedCall6transformer_decoder_1/StatefulPartitionedCall:output:0dense_44862dense_44864*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43793�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'�
NoOpNoOp^dense/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_decoder/StatefulPartitionedCall.^transformer_decoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_decoder/StatefulPartitionedCall+transformer_decoder/StatefulPartitionedCall2^
-transformer_decoder_1/StatefulPartitionedCall-transformer_decoder_1/StatefulPartitionedCall:Y U
0
_output_shapes
:������������������
!
_user_specified_name	input_1
�
�
@__inference_dense_layer_call_and_return_conditional_losses_43793

inputs5
!tensordot_readvariableop_resource:
��'.
biasadd_readvariableop_resource:	�'
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��'*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������'\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�'Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������'s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�'*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������'m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
��
�/
@__inference_model_layer_call_and_return_conditional_losses_45520

inputsW
Ctoken_and_position_embedding_token_embedding_embedding_lookup_45110:
�'�a
Mtoken_and_position_embedding_position_embedding_slice_readvariableop_resource:
��e
Ntransformer_decoder_self_attention_query_einsum_einsum_readvariableop_resource:�UV
Dtransformer_decoder_self_attention_query_add_readvariableop_resource:Uc
Ltransformer_decoder_self_attention_key_einsum_einsum_readvariableop_resource:�UT
Btransformer_decoder_self_attention_key_add_readvariableop_resource:Ue
Ntransformer_decoder_self_attention_value_einsum_einsum_readvariableop_resource:�UV
Dtransformer_decoder_self_attention_value_add_readvariableop_resource:Up
Ytransformer_decoder_self_attention_attention_output_einsum_einsum_readvariableop_resource:U�^
Otransformer_decoder_self_attention_attention_output_add_readvariableop_resource:	�b
Stransformer_decoder_self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�^
Otransformer_decoder_self_attention_layer_norm_batchnorm_readvariableop_resource:	�h
Ttransformer_decoder_feedforward_intermediate_dense_tensordot_readvariableop_resource:
��a
Rtransformer_decoder_feedforward_intermediate_dense_biasadd_readvariableop_resource:	�b
Ntransformer_decoder_feedforward_output_dense_tensordot_readvariableop_resource:
��[
Ltransformer_decoder_feedforward_output_dense_biasadd_readvariableop_resource:	�_
Ptransformer_decoder_feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�[
Ltransformer_decoder_feedforward_layer_norm_batchnorm_readvariableop_resource:	�g
Ptransformer_decoder_1_self_attention_query_einsum_einsum_readvariableop_resource:�UX
Ftransformer_decoder_1_self_attention_query_add_readvariableop_resource:Ue
Ntransformer_decoder_1_self_attention_key_einsum_einsum_readvariableop_resource:�UV
Dtransformer_decoder_1_self_attention_key_add_readvariableop_resource:Ug
Ptransformer_decoder_1_self_attention_value_einsum_einsum_readvariableop_resource:�UX
Ftransformer_decoder_1_self_attention_value_add_readvariableop_resource:Ur
[transformer_decoder_1_self_attention_attention_output_einsum_einsum_readvariableop_resource:U�`
Qtransformer_decoder_1_self_attention_attention_output_add_readvariableop_resource:	�d
Utransformer_decoder_1_self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�`
Qtransformer_decoder_1_self_attention_layer_norm_batchnorm_readvariableop_resource:	�j
Vtransformer_decoder_1_feedforward_intermediate_dense_tensordot_readvariableop_resource:
��c
Ttransformer_decoder_1_feedforward_intermediate_dense_biasadd_readvariableop_resource:	�d
Ptransformer_decoder_1_feedforward_output_dense_tensordot_readvariableop_resource:
��]
Ntransformer_decoder_1_feedforward_output_dense_biasadd_readvariableop_resource:	�a
Rtransformer_decoder_1_feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�]
Ntransformer_decoder_1_feedforward_layer_norm_batchnorm_readvariableop_resource:	�;
'dense_tensordot_readvariableop_resource:
��'4
%dense_biasadd_readvariableop_resource:	�'
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�Dtoken_and_position_embedding/position_embedding/Slice/ReadVariableOp�=token_and_position_embedding/token_embedding/embedding_lookup�Itransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOp�Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOp�Ctransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOp�Gtransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOp�Ctransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOp�Etransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOp�Ftransformer_decoder/self_attention/attention_output/add/ReadVariableOp�Ptransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp�9transformer_decoder/self_attention/key/add/ReadVariableOp�Ctransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp�;transformer_decoder/self_attention/query/add/ReadVariableOp�Etransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp�;transformer_decoder/self_attention/value/add/ReadVariableOp�Etransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOp�Ftransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOp�Jtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOp�Ktransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOp�Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOp�Etransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOp�Itransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOp�Etransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOp�Gtransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOp�Htransformer_decoder_1/self_attention/attention_output/add/ReadVariableOp�Rtransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp�;transformer_decoder_1/self_attention/key/add/ReadVariableOp�Etransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp�=transformer_decoder_1/self_attention/query/add/ReadVariableOp�Gtransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp�=transformer_decoder_1/self_attention/value/add/ReadVariableOp�Gtransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOp�Htransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOp�Ltransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp�
=token_and_position_embedding/token_embedding/embedding_lookupResourceGatherCtoken_and_position_embedding_token_embedding_embedding_lookup_45110inputs*
Tindices0*V
_classL
JHloc:@token_and_position_embedding/token_embedding/embedding_lookup/45110*5
_output_shapes#
!:�������������������*
dtype0�
Ftoken_and_position_embedding/token_embedding/embedding_lookup/IdentityIdentityFtoken_and_position_embedding/token_embedding/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding/token_embedding/embedding_lookup/45110*5
_output_shapes#
!:��������������������
Htoken_and_position_embedding/token_embedding/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding/token_embedding/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:�������������������y
7token_and_position_embedding/token_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
5token_and_position_embedding/token_embedding/NotEqualNotEqualinputs@token_and_position_embedding/token_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:�������������������
5token_and_position_embedding/position_embedding/ShapeShapeQtoken_and_position_embedding/token_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:�
Ctoken_and_position_embedding/position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Etoken_and_position_embedding/position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Etoken_and_position_embedding/position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=token_and_position_embedding/position_embedding/strided_sliceStridedSlice>token_and_position_embedding/position_embedding/Shape:output:0Ltoken_and_position_embedding/position_embedding/strided_slice/stack:output:0Ntoken_and_position_embedding/position_embedding/strided_slice/stack_1:output:0Ntoken_and_position_embedding/position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Etoken_and_position_embedding/position_embedding/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Gtoken_and_position_embedding/position_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gtoken_and_position_embedding/position_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?token_and_position_embedding/position_embedding/strided_slice_1StridedSlice>token_and_position_embedding/position_embedding/Shape:output:0Ntoken_and_position_embedding/position_embedding/strided_slice_1/stack:output:0Ptoken_and_position_embedding/position_embedding/strided_slice_1/stack_1:output:0Ptoken_and_position_embedding/position_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Dtoken_and_position_embedding/position_embedding/Slice/ReadVariableOpReadVariableOpMtoken_and_position_embedding_position_embedding_slice_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;token_and_position_embedding/position_embedding/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        
<token_and_position_embedding/position_embedding/Slice/size/1Const*
_output_shapes
: *
dtype0*
value
B :��
:token_and_position_embedding/position_embedding/Slice/sizePackHtoken_and_position_embedding/position_embedding/strided_slice_1:output:0Etoken_and_position_embedding/position_embedding/Slice/size/1:output:0*
N*
T0*
_output_shapes
:�
5token_and_position_embedding/position_embedding/SliceSliceLtoken_and_position_embedding/position_embedding/Slice/ReadVariableOp:value:0Dtoken_and_position_embedding/position_embedding/Slice/begin:output:0Ctoken_and_position_embedding/position_embedding/Slice/size:output:0*
Index0*
T0*(
_output_shapes
:����������{
8token_and_position_embedding/position_embedding/packed/2Const*
_output_shapes
: *
dtype0*
value
B :��
6token_and_position_embedding/position_embedding/packedPackFtoken_and_position_embedding/position_embedding/strided_slice:output:0Htoken_and_position_embedding/position_embedding/strided_slice_1:output:0Atoken_and_position_embedding/position_embedding/packed/2:output:0*
N*
T0*
_output_shapes
:v
4token_and_position_embedding/position_embedding/RankConst*
_output_shapes
: *
dtype0*
value	B :�
;token_and_position_embedding/position_embedding/BroadcastToBroadcastTo>token_and_position_embedding/position_embedding/Slice:output:0?token_and_position_embedding/position_embedding/packed:output:0*
T0*5
_output_shapes#
!:��������������������
 token_and_position_embedding/addAddV2Qtoken_and_position_embedding/token_embedding/embedding_lookup/Identity_1:output:0Dtoken_and_position_embedding/position_embedding/BroadcastTo:output:0*
T0*5
_output_shapes#
!:�������������������i
'token_and_position_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
%token_and_position_embedding/NotEqualNotEqualinputs0token_and_position_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:������������������d
"transformer_decoder/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
transformer_decoder/ExpandDims
ExpandDims)token_and_position_embedding/NotEqual:z:0+transformer_decoder/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
transformer_decoder/CastCast'transformer_decoder/ExpandDims:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :������������������m
transformer_decoder/ShapeShape$token_and_position_embedding/add:z:0*
T0*
_output_shapes
:q
'transformer_decoder/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)transformer_decoder/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)transformer_decoder/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!transformer_decoder/strided_sliceStridedSlice"transformer_decoder/Shape:output:00transformer_decoder/strided_slice/stack:output:02transformer_decoder/strided_slice/stack_1:output:02transformer_decoder/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)transformer_decoder/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#transformer_decoder/strided_slice_1StridedSlice"transformer_decoder/Shape:output:02transformer_decoder/strided_slice_1/stack:output:04transformer_decoder/strided_slice_1/stack_1:output:04transformer_decoder/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
transformer_decoder/Shape_1Shape$token_and_position_embedding/add:z:0*
T0*
_output_shapes
:s
)transformer_decoder/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+transformer_decoder/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#transformer_decoder/strided_slice_2StridedSlice$transformer_decoder/Shape_1:output:02transformer_decoder/strided_slice_2/stack:output:04transformer_decoder/strided_slice_2/stack_1:output:04transformer_decoder/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)transformer_decoder/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#transformer_decoder/strided_slice_3StridedSlice$transformer_decoder/Shape_1:output:02transformer_decoder/strided_slice_3/stack:output:04transformer_decoder/strided_slice_3/stack_1:output:04transformer_decoder/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
transformer_decoder/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    d
transformer_decoder/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
transformer_decoder/range/CastCast,transformer_decoder/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder/rangeRange(transformer_decoder/range/start:output:0"transformer_decoder/range/Cast:y:0(transformer_decoder/range/delta:output:0*

Tidx0*#
_output_shapes
:���������^
transformer_decoder/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : y
transformer_decoder/Cast_1Cast%transformer_decoder/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder/addAddV2"transformer_decoder/range:output:0transformer_decoder/Cast_1:y:0*
T0*#
_output_shapes
:���������f
$transformer_decoder/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
 transformer_decoder/ExpandDims_1
ExpandDimstransformer_decoder/add:z:0-transformer_decoder/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������f
!transformer_decoder/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    f
!transformer_decoder/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 transformer_decoder/range_1/CastCast,transformer_decoder/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder/range_1Range*transformer_decoder/range_1/start:output:0$transformer_decoder/range_1/Cast:y:0*transformer_decoder/range_1/delta:output:0*

Tidx0*#
_output_shapes
:����������
 transformer_decoder/GreaterEqualGreaterEqual)transformer_decoder/ExpandDims_1:output:0$transformer_decoder/range_1:output:0*
T0*0
_output_shapes
:������������������f
$transformer_decoder/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : �
 transformer_decoder/ExpandDims_2
ExpandDims$transformer_decoder/GreaterEqual:z:0-transformer_decoder/ExpandDims_2/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
transformer_decoder/packedPack*transformer_decoder/strided_slice:output:0,transformer_decoder/strided_slice_3:output:0,transformer_decoder/strided_slice_3:output:0*
N*
T0*
_output_shapes
:Z
transformer_decoder/RankConst*
_output_shapes
: *
dtype0*
value	B :�
transformer_decoder/BroadcastToBroadcastTo)transformer_decoder/ExpandDims_2:output:0#transformer_decoder/packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
transformer_decoder/Cast_2Cast(transformer_decoder/BroadcastTo:output:0*

DstT0*

SrcT0
*=
_output_shapes+
):'����������������������������
transformer_decoder/MinimumMinimumtransformer_decoder/Cast:y:0transformer_decoder/Cast_2:y:0*
T0*=
_output_shapes+
):'����������������������������
Etransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_decoder_self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
6transformer_decoder/self_attention/query/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Mtransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
;transformer_decoder/self_attention/query/add/ReadVariableOpReadVariableOpDtransformer_decoder_self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
,transformer_decoder/self_attention/query/addAddV2?transformer_decoder/self_attention/query/einsum/Einsum:output:0Ctransformer_decoder/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Ctransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpLtransformer_decoder_self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
4transformer_decoder/self_attention/key/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Ktransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
9transformer_decoder/self_attention/key/add/ReadVariableOpReadVariableOpBtransformer_decoder_self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
*transformer_decoder/self_attention/key/addAddV2=transformer_decoder/self_attention/key/einsum/Einsum:output:0Atransformer_decoder/self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Etransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_decoder_self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
6transformer_decoder/self_attention/value/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Mtransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
;transformer_decoder/self_attention/value/add/ReadVariableOpReadVariableOpDtransformer_decoder_self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
,transformer_decoder/self_attention/value/addAddV2?transformer_decoder/self_attention/value/einsum/Einsum:output:0Ctransformer_decoder/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������Uk
)transformer_decoder/self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :U�
'transformer_decoder/self_attention/CastCast2transformer_decoder/self_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: }
'transformer_decoder/self_attention/SqrtSqrt+transformer_decoder/self_attention/Cast:y:0*
T0*
_output_shapes
: q
,transformer_decoder/self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*transformer_decoder/self_attention/truedivRealDiv5transformer_decoder/self_attention/truediv/x:output:0+transformer_decoder/self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
&transformer_decoder/self_attention/MulMul0transformer_decoder/self_attention/query/add:z:0.transformer_decoder/self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
0transformer_decoder/self_attention/einsum/EinsumEinsum.transformer_decoder/self_attention/key/add:z:0*transformer_decoder/self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbe|
1transformer_decoder/self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
-transformer_decoder/self_attention/ExpandDims
ExpandDimstransformer_decoder/Minimum:z:0:transformer_decoder/self_attention/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+����������������������������
/transformer_decoder/self_attention/softmax/CastCast6transformer_decoder/self_attention/ExpandDims:output:0*

DstT0*

SrcT0*A
_output_shapes/
-:+���������������������������u
0transformer_decoder/self_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.transformer_decoder/self_attention/softmax/subSub9transformer_decoder/self_attention/softmax/sub/x:output:03transformer_decoder/self_attention/softmax/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������u
0transformer_decoder/self_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
.transformer_decoder/self_attention/softmax/mulMul2transformer_decoder/self_attention/softmax/sub:z:09transformer_decoder/self_attention/softmax/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
.transformer_decoder/self_attention/softmax/addAddV29transformer_decoder/self_attention/einsum/Einsum:output:02transformer_decoder/self_attention/softmax/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
2transformer_decoder/self_attention/softmax/SoftmaxSoftmax2transformer_decoder/self_attention/softmax/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
3transformer_decoder/self_attention/dropout/IdentityIdentity<transformer_decoder/self_attention/softmax/Softmax:softmax:0*
T0*A
_output_shapes/
-:+����������������������������
2transformer_decoder/self_attention/einsum_1/EinsumEinsum<transformer_decoder/self_attention/dropout/Identity:output:00transformer_decoder/self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
Ptransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpYtransformer_decoder_self_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
Atransformer_decoder/self_attention/attention_output/einsum/EinsumEinsum;transformer_decoder/self_attention/einsum_1/Einsum:output:0Xtransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
Ftransformer_decoder/self_attention/attention_output/add/ReadVariableOpReadVariableOpOtransformer_decoder_self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_decoder/self_attention/attention_output/addAddV2Jtransformer_decoder/self_attention/attention_output/einsum/Einsum:output:0Ntransformer_decoder/self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
3transformer_decoder/self_attention_dropout/IdentityIdentity;transformer_decoder/self_attention/attention_output/add:z:0*
T0*5
_output_shapes#
!:��������������������
transformer_decoder/add_1AddV2<transformer_decoder/self_attention_dropout/Identity:output:0$token_and_position_embedding/add:z:0*
T0*5
_output_shapes#
!:��������������������
Ltransformer_decoder/self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_decoder/self_attention_layer_norm/moments/meanMeantransformer_decoder/add_1:z:0Utransformer_decoder/self_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Btransformer_decoder/self_attention_layer_norm/moments/StopGradientStopGradientCtransformer_decoder/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Gtransformer_decoder/self_attention_layer_norm/moments/SquaredDifferenceSquaredDifferencetransformer_decoder/add_1:z:0Ktransformer_decoder/self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Ptransformer_decoder/self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>transformer_decoder/self_attention_layer_norm/moments/varianceMeanKtransformer_decoder/self_attention_layer_norm/moments/SquaredDifference:z:0Ytransformer_decoder/self_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
=transformer_decoder/self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
;transformer_decoder/self_attention_layer_norm/batchnorm/addAddV2Gtransformer_decoder/self_attention_layer_norm/moments/variance:output:0Ftransformer_decoder/self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
=transformer_decoder/self_attention_layer_norm/batchnorm/RsqrtRsqrt?transformer_decoder/self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Jtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpStransformer_decoder_self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;transformer_decoder/self_attention_layer_norm/batchnorm/mulMulAtransformer_decoder/self_attention_layer_norm/batchnorm/Rsqrt:y:0Rtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
=transformer_decoder/self_attention_layer_norm/batchnorm/mul_1Multransformer_decoder/add_1:z:0?transformer_decoder/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
=transformer_decoder/self_attention_layer_norm/batchnorm/mul_2MulCtransformer_decoder/self_attention_layer_norm/moments/mean:output:0?transformer_decoder/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Ftransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOpOtransformer_decoder_self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;transformer_decoder/self_attention_layer_norm/batchnorm/subSubNtransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOp:value:0Atransformer_decoder/self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
=transformer_decoder/self_attention_layer_norm/batchnorm/add_1AddV2Atransformer_decoder/self_attention_layer_norm/batchnorm/mul_1:z:0?transformer_decoder/self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOpTtransformer_decoder_feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Atransformer_decoder/feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_decoder/feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Btransformer_decoder/feedforward_intermediate_dense/Tensordot/ShapeShapeAtransformer_decoder/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Etransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2GatherV2Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/Shape:output:0Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/free:output:0Stransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ltransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Gtransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV2Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/Shape:output:0Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/axes:output:0Utransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Btransformer_decoder/feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Atransformer_decoder/feedforward_intermediate_dense/Tensordot/ProdProdNtransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Dtransformer_decoder/feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Ctransformer_decoder/feedforward_intermediate_dense/Tensordot/Prod_1ProdPtransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1:output:0Mtransformer_decoder/feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Htransformer_decoder/feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Ctransformer_decoder/feedforward_intermediate_dense/Tensordot/concatConcatV2Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/free:output:0Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/axes:output:0Qtransformer_decoder/feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Btransformer_decoder/feedforward_intermediate_dense/Tensordot/stackPackJtransformer_decoder/feedforward_intermediate_dense/Tensordot/Prod:output:0Ltransformer_decoder/feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Ftransformer_decoder/feedforward_intermediate_dense/Tensordot/transpose	TransposeAtransformer_decoder/self_attention_layer_norm/batchnorm/add_1:z:0Ltransformer_decoder/feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Dtransformer_decoder/feedforward_intermediate_dense/Tensordot/ReshapeReshapeJtransformer_decoder/feedforward_intermediate_dense/Tensordot/transpose:y:0Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Ctransformer_decoder/feedforward_intermediate_dense/Tensordot/MatMulMatMulMtransformer_decoder/feedforward_intermediate_dense/Tensordot/Reshape:output:0Stransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Dtransformer_decoder/feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Etransformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1ConcatV2Ntransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Mtransformer_decoder/feedforward_intermediate_dense/Tensordot/Const_2:output:0Stransformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
<transformer_decoder/feedforward_intermediate_dense/TensordotReshapeMtransformer_decoder/feedforward_intermediate_dense/Tensordot/MatMul:product:0Ntransformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Itransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOpRtransformer_decoder_feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:transformer_decoder/feedforward_intermediate_dense/BiasAddBiasAddEtransformer_decoder/feedforward_intermediate_dense/Tensordot:output:0Qtransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
7transformer_decoder/feedforward_intermediate_dense/ReluReluCtransformer_decoder/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
Etransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOpNtransformer_decoder_feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;transformer_decoder/feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
;transformer_decoder/feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
<transformer_decoder/feedforward_output_dense/Tensordot/ShapeShapeEtransformer_decoder/feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:�
Dtransformer_decoder/feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?transformer_decoder/feedforward_output_dense/Tensordot/GatherV2GatherV2Etransformer_decoder/feedforward_output_dense/Tensordot/Shape:output:0Dtransformer_decoder/feedforward_output_dense/Tensordot/free:output:0Mtransformer_decoder/feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ftransformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Atransformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1GatherV2Etransformer_decoder/feedforward_output_dense/Tensordot/Shape:output:0Dtransformer_decoder/feedforward_output_dense/Tensordot/axes:output:0Otransformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
<transformer_decoder/feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
;transformer_decoder/feedforward_output_dense/Tensordot/ProdProdHtransformer_decoder/feedforward_output_dense/Tensordot/GatherV2:output:0Etransformer_decoder/feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
>transformer_decoder/feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
=transformer_decoder/feedforward_output_dense/Tensordot/Prod_1ProdJtransformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1:output:0Gtransformer_decoder/feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Btransformer_decoder/feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=transformer_decoder/feedforward_output_dense/Tensordot/concatConcatV2Dtransformer_decoder/feedforward_output_dense/Tensordot/free:output:0Dtransformer_decoder/feedforward_output_dense/Tensordot/axes:output:0Ktransformer_decoder/feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
<transformer_decoder/feedforward_output_dense/Tensordot/stackPackDtransformer_decoder/feedforward_output_dense/Tensordot/Prod:output:0Ftransformer_decoder/feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
@transformer_decoder/feedforward_output_dense/Tensordot/transpose	TransposeEtransformer_decoder/feedforward_intermediate_dense/Relu:activations:0Ftransformer_decoder/feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
>transformer_decoder/feedforward_output_dense/Tensordot/ReshapeReshapeDtransformer_decoder/feedforward_output_dense/Tensordot/transpose:y:0Etransformer_decoder/feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
=transformer_decoder/feedforward_output_dense/Tensordot/MatMulMatMulGtransformer_decoder/feedforward_output_dense/Tensordot/Reshape:output:0Mtransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
>transformer_decoder/feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Dtransformer_decoder/feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?transformer_decoder/feedforward_output_dense/Tensordot/concat_1ConcatV2Htransformer_decoder/feedforward_output_dense/Tensordot/GatherV2:output:0Gtransformer_decoder/feedforward_output_dense/Tensordot/Const_2:output:0Mtransformer_decoder/feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_decoder/feedforward_output_dense/TensordotReshapeGtransformer_decoder/feedforward_output_dense/Tensordot/MatMul:product:0Htransformer_decoder/feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Ctransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOpLtransformer_decoder_feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4transformer_decoder/feedforward_output_dense/BiasAddBiasAdd?transformer_decoder/feedforward_output_dense/Tensordot:output:0Ktransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
0transformer_decoder/feedforward_dropout/IdentityIdentity=transformer_decoder/feedforward_output_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
transformer_decoder/add_2AddV29transformer_decoder/feedforward_dropout/Identity:output:0Atransformer_decoder/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Itransformer_decoder/feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_decoder/feedforward_layer_norm/moments/meanMeantransformer_decoder/add_2:z:0Rtransformer_decoder/feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
?transformer_decoder/feedforward_layer_norm/moments/StopGradientStopGradient@transformer_decoder/feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Dtransformer_decoder/feedforward_layer_norm/moments/SquaredDifferenceSquaredDifferencetransformer_decoder/add_2:z:0Htransformer_decoder/feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Mtransformer_decoder/feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
;transformer_decoder/feedforward_layer_norm/moments/varianceMeanHtransformer_decoder/feedforward_layer_norm/moments/SquaredDifference:z:0Vtransformer_decoder/feedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(
:transformer_decoder/feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
8transformer_decoder/feedforward_layer_norm/batchnorm/addAddV2Dtransformer_decoder/feedforward_layer_norm/moments/variance:output:0Ctransformer_decoder/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
:transformer_decoder/feedforward_layer_norm/batchnorm/RsqrtRsqrt<transformer_decoder/feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Gtransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_decoder_feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8transformer_decoder/feedforward_layer_norm/batchnorm/mulMul>transformer_decoder/feedforward_layer_norm/batchnorm/Rsqrt:y:0Otransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
:transformer_decoder/feedforward_layer_norm/batchnorm/mul_1Multransformer_decoder/add_2:z:0<transformer_decoder/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
:transformer_decoder/feedforward_layer_norm/batchnorm/mul_2Mul@transformer_decoder/feedforward_layer_norm/moments/mean:output:0<transformer_decoder/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Ctransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOpLtransformer_decoder_feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8transformer_decoder/feedforward_layer_norm/batchnorm/subSubKtransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOp:value:0>transformer_decoder/feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
:transformer_decoder/feedforward_layer_norm/batchnorm/add_1AddV2>transformer_decoder/feedforward_layer_norm/batchnorm/mul_1:z:0<transformer_decoder/feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������f
$transformer_decoder_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
 transformer_decoder_1/ExpandDims
ExpandDims)token_and_position_embedding/NotEqual:z:0-transformer_decoder_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
transformer_decoder_1/CastCast)transformer_decoder_1/ExpandDims:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :�������������������
transformer_decoder_1/ShapeShape>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:s
)transformer_decoder_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+transformer_decoder_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#transformer_decoder_1/strided_sliceStridedSlice$transformer_decoder_1/Shape:output:02transformer_decoder_1/strided_slice/stack:output:04transformer_decoder_1/strided_slice/stack_1:output:04transformer_decoder_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
+transformer_decoder_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%transformer_decoder_1/strided_slice_1StridedSlice$transformer_decoder_1/Shape:output:04transformer_decoder_1/strided_slice_1/stack:output:06transformer_decoder_1/strided_slice_1/stack_1:output:06transformer_decoder_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
transformer_decoder_1/Shape_1Shape>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:u
+transformer_decoder_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-transformer_decoder_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%transformer_decoder_1/strided_slice_2StridedSlice&transformer_decoder_1/Shape_1:output:04transformer_decoder_1/strided_slice_2/stack:output:06transformer_decoder_1/strided_slice_2/stack_1:output:06transformer_decoder_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
+transformer_decoder_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%transformer_decoder_1/strided_slice_3StridedSlice&transformer_decoder_1/Shape_1:output:04transformer_decoder_1/strided_slice_3/stack:output:06transformer_decoder_1/strided_slice_3/stack_1:output:06transformer_decoder_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
!transformer_decoder_1/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    f
!transformer_decoder_1/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 transformer_decoder_1/range/CastCast.transformer_decoder_1/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder_1/rangeRange*transformer_decoder_1/range/start:output:0$transformer_decoder_1/range/Cast:y:0*transformer_decoder_1/range/delta:output:0*

Tidx0*#
_output_shapes
:���������`
transformer_decoder_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : }
transformer_decoder_1/Cast_1Cast'transformer_decoder_1/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder_1/addAddV2$transformer_decoder_1/range:output:0 transformer_decoder_1/Cast_1:y:0*
T0*#
_output_shapes
:���������h
&transformer_decoder_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
"transformer_decoder_1/ExpandDims_1
ExpandDimstransformer_decoder_1/add:z:0/transformer_decoder_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������h
#transformer_decoder_1/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#transformer_decoder_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"transformer_decoder_1/range_1/CastCast.transformer_decoder_1/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder_1/range_1Range,transformer_decoder_1/range_1/start:output:0&transformer_decoder_1/range_1/Cast:y:0,transformer_decoder_1/range_1/delta:output:0*

Tidx0*#
_output_shapes
:����������
"transformer_decoder_1/GreaterEqualGreaterEqual+transformer_decoder_1/ExpandDims_1:output:0&transformer_decoder_1/range_1:output:0*
T0*0
_output_shapes
:������������������h
&transformer_decoder_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"transformer_decoder_1/ExpandDims_2
ExpandDims&transformer_decoder_1/GreaterEqual:z:0/transformer_decoder_1/ExpandDims_2/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
transformer_decoder_1/packedPack,transformer_decoder_1/strided_slice:output:0.transformer_decoder_1/strided_slice_3:output:0.transformer_decoder_1/strided_slice_3:output:0*
N*
T0*
_output_shapes
:\
transformer_decoder_1/RankConst*
_output_shapes
: *
dtype0*
value	B :�
!transformer_decoder_1/BroadcastToBroadcastTo+transformer_decoder_1/ExpandDims_2:output:0%transformer_decoder_1/packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
transformer_decoder_1/Cast_2Cast*transformer_decoder_1/BroadcastTo:output:0*

DstT0*

SrcT0
*=
_output_shapes+
):'����������������������������
transformer_decoder_1/MinimumMinimumtransformer_decoder_1/Cast:y:0 transformer_decoder_1/Cast_2:y:0*
T0*=
_output_shapes+
):'����������������������������
Gtransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_decoder_1_self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
8transformer_decoder_1/self_attention/query/einsum/EinsumEinsum>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0Otransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
=transformer_decoder_1/self_attention/query/add/ReadVariableOpReadVariableOpFtransformer_decoder_1_self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
.transformer_decoder_1/self_attention/query/addAddV2Atransformer_decoder_1/self_attention/query/einsum/Einsum:output:0Etransformer_decoder_1/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Etransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_decoder_1_self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
6transformer_decoder_1/self_attention/key/einsum/EinsumEinsum>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0Mtransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
;transformer_decoder_1/self_attention/key/add/ReadVariableOpReadVariableOpDtransformer_decoder_1_self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
,transformer_decoder_1/self_attention/key/addAddV2?transformer_decoder_1/self_attention/key/einsum/Einsum:output:0Ctransformer_decoder_1/self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Gtransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_decoder_1_self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
8transformer_decoder_1/self_attention/value/einsum/EinsumEinsum>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0Otransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
=transformer_decoder_1/self_attention/value/add/ReadVariableOpReadVariableOpFtransformer_decoder_1_self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
.transformer_decoder_1/self_attention/value/addAddV2Atransformer_decoder_1/self_attention/value/einsum/Einsum:output:0Etransformer_decoder_1/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������Um
+transformer_decoder_1/self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :U�
)transformer_decoder_1/self_attention/CastCast4transformer_decoder_1/self_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
)transformer_decoder_1/self_attention/SqrtSqrt-transformer_decoder_1/self_attention/Cast:y:0*
T0*
_output_shapes
: s
.transformer_decoder_1/self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,transformer_decoder_1/self_attention/truedivRealDiv7transformer_decoder_1/self_attention/truediv/x:output:0-transformer_decoder_1/self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
(transformer_decoder_1/self_attention/MulMul2transformer_decoder_1/self_attention/query/add:z:00transformer_decoder_1/self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
2transformer_decoder_1/self_attention/einsum/EinsumEinsum0transformer_decoder_1/self_attention/key/add:z:0,transformer_decoder_1/self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbe~
3transformer_decoder_1/self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
/transformer_decoder_1/self_attention/ExpandDims
ExpandDims!transformer_decoder_1/Minimum:z:0<transformer_decoder_1/self_attention/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+����������������������������
3transformer_decoder_1/self_attention/softmax_1/CastCast8transformer_decoder_1/self_attention/ExpandDims:output:0*

DstT0*

SrcT0*A
_output_shapes/
-:+���������������������������y
4transformer_decoder_1/self_attention/softmax_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
2transformer_decoder_1/self_attention/softmax_1/subSub=transformer_decoder_1/self_attention/softmax_1/sub/x:output:07transformer_decoder_1/self_attention/softmax_1/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������y
4transformer_decoder_1/self_attention/softmax_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
2transformer_decoder_1/self_attention/softmax_1/mulMul6transformer_decoder_1/self_attention/softmax_1/sub:z:0=transformer_decoder_1/self_attention/softmax_1/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
2transformer_decoder_1/self_attention/softmax_1/addAddV2;transformer_decoder_1/self_attention/einsum/Einsum:output:06transformer_decoder_1/self_attention/softmax_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
6transformer_decoder_1/self_attention/softmax_1/SoftmaxSoftmax6transformer_decoder_1/self_attention/softmax_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
7transformer_decoder_1/self_attention/dropout_1/IdentityIdentity@transformer_decoder_1/self_attention/softmax_1/Softmax:softmax:0*
T0*A
_output_shapes/
-:+����������������������������
4transformer_decoder_1/self_attention/einsum_1/EinsumEinsum@transformer_decoder_1/self_attention/dropout_1/Identity:output:02transformer_decoder_1/self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
Rtransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp[transformer_decoder_1_self_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
Ctransformer_decoder_1/self_attention/attention_output/einsum/EinsumEinsum=transformer_decoder_1/self_attention/einsum_1/Einsum:output:0Ztransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
Htransformer_decoder_1/self_attention/attention_output/add/ReadVariableOpReadVariableOpQtransformer_decoder_1_self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
9transformer_decoder_1/self_attention/attention_output/addAddV2Ltransformer_decoder_1/self_attention/attention_output/einsum/Einsum:output:0Ptransformer_decoder_1/self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
5transformer_decoder_1/self_attention_dropout/IdentityIdentity=transformer_decoder_1/self_attention/attention_output/add:z:0*
T0*5
_output_shapes#
!:��������������������
transformer_decoder_1/add_1AddV2>transformer_decoder_1/self_attention_dropout/Identity:output:0>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Ntransformer_decoder_1/self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
<transformer_decoder_1/self_attention_layer_norm/moments/meanMeantransformer_decoder_1/add_1:z:0Wtransformer_decoder_1/self_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Dtransformer_decoder_1/self_attention_layer_norm/moments/StopGradientStopGradientEtransformer_decoder_1/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Itransformer_decoder_1/self_attention_layer_norm/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_1/add_1:z:0Mtransformer_decoder_1/self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Rtransformer_decoder_1/self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
@transformer_decoder_1/self_attention_layer_norm/moments/varianceMeanMtransformer_decoder_1/self_attention_layer_norm/moments/SquaredDifference:z:0[transformer_decoder_1/self_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
?transformer_decoder_1/self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
=transformer_decoder_1/self_attention_layer_norm/batchnorm/addAddV2Itransformer_decoder_1/self_attention_layer_norm/moments/variance:output:0Htransformer_decoder_1/self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
?transformer_decoder_1/self_attention_layer_norm/batchnorm/RsqrtRsqrtAtransformer_decoder_1/self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Ltransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpUtransformer_decoder_1_self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_decoder_1/self_attention_layer_norm/batchnorm/mulMulCtransformer_decoder_1/self_attention_layer_norm/batchnorm/Rsqrt:y:0Ttransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
?transformer_decoder_1/self_attention_layer_norm/batchnorm/mul_1Multransformer_decoder_1/add_1:z:0Atransformer_decoder_1/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
?transformer_decoder_1/self_attention_layer_norm/batchnorm/mul_2MulEtransformer_decoder_1/self_attention_layer_norm/moments/mean:output:0Atransformer_decoder_1/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Htransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOpQtransformer_decoder_1_self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_decoder_1/self_attention_layer_norm/batchnorm/subSubPtransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOp:value:0Ctransformer_decoder_1/self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
?transformer_decoder_1/self_attention_layer_norm/batchnorm/add_1AddV2Ctransformer_decoder_1/self_attention_layer_norm/batchnorm/mul_1:z:0Atransformer_decoder_1/self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOpVtransformer_decoder_1_feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Ctransformer_decoder_1/feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Ctransformer_decoder_1/feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Dtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ShapeShapeCtransformer_decoder_1/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Gtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2GatherV2Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Shape:output:0Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/free:output:0Utransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ntransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Itransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV2Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Shape:output:0Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/axes:output:0Wtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Dtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Ctransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ProdProdPtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Ftransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Etransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Prod_1ProdRtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1:output:0Otransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Jtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Etransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concatConcatV2Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/free:output:0Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/axes:output:0Stransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/stackPackLtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Prod:output:0Ntransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Htransformer_decoder_1/feedforward_intermediate_dense/Tensordot/transpose	TransposeCtransformer_decoder_1/self_attention_layer_norm/batchnorm/add_1:z:0Ntransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Ftransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReshapeReshapeLtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/transpose:y:0Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Etransformer_decoder_1/feedforward_intermediate_dense/Tensordot/MatMulMatMulOtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Reshape:output:0Utransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Ftransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Gtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1ConcatV2Ptransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Otransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_2:output:0Utransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
>transformer_decoder_1/feedforward_intermediate_dense/TensordotReshapeOtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/MatMul:product:0Ptransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Ktransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOpTtransformer_decoder_1_feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<transformer_decoder_1/feedforward_intermediate_dense/BiasAddBiasAddGtransformer_decoder_1/feedforward_intermediate_dense/Tensordot:output:0Stransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
9transformer_decoder_1/feedforward_intermediate_dense/ReluReluEtransformer_decoder_1/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
Gtransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOpPtransformer_decoder_1_feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
=transformer_decoder_1/feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
=transformer_decoder_1/feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
>transformer_decoder_1/feedforward_output_dense/Tensordot/ShapeShapeGtransformer_decoder_1/feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:�
Ftransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Atransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2GatherV2Gtransformer_decoder_1/feedforward_output_dense/Tensordot/Shape:output:0Ftransformer_decoder_1/feedforward_output_dense/Tensordot/free:output:0Otransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Htransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Ctransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1GatherV2Gtransformer_decoder_1/feedforward_output_dense/Tensordot/Shape:output:0Ftransformer_decoder_1/feedforward_output_dense/Tensordot/axes:output:0Qtransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_decoder_1/feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
=transformer_decoder_1/feedforward_output_dense/Tensordot/ProdProdJtransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2:output:0Gtransformer_decoder_1/feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
@transformer_decoder_1/feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
?transformer_decoder_1/feedforward_output_dense/Tensordot/Prod_1ProdLtransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1:output:0Itransformer_decoder_1/feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Dtransformer_decoder_1/feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?transformer_decoder_1/feedforward_output_dense/Tensordot/concatConcatV2Ftransformer_decoder_1/feedforward_output_dense/Tensordot/free:output:0Ftransformer_decoder_1/feedforward_output_dense/Tensordot/axes:output:0Mtransformer_decoder_1/feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
>transformer_decoder_1/feedforward_output_dense/Tensordot/stackPackFtransformer_decoder_1/feedforward_output_dense/Tensordot/Prod:output:0Htransformer_decoder_1/feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Btransformer_decoder_1/feedforward_output_dense/Tensordot/transpose	TransposeGtransformer_decoder_1/feedforward_intermediate_dense/Relu:activations:0Htransformer_decoder_1/feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
@transformer_decoder_1/feedforward_output_dense/Tensordot/ReshapeReshapeFtransformer_decoder_1/feedforward_output_dense/Tensordot/transpose:y:0Gtransformer_decoder_1/feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
?transformer_decoder_1/feedforward_output_dense/Tensordot/MatMulMatMulItransformer_decoder_1/feedforward_output_dense/Tensordot/Reshape:output:0Otransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
@transformer_decoder_1/feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Ftransformer_decoder_1/feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Atransformer_decoder_1/feedforward_output_dense/Tensordot/concat_1ConcatV2Jtransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2:output:0Itransformer_decoder_1/feedforward_output_dense/Tensordot/Const_2:output:0Otransformer_decoder_1/feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
8transformer_decoder_1/feedforward_output_dense/TensordotReshapeItransformer_decoder_1/feedforward_output_dense/Tensordot/MatMul:product:0Jtransformer_decoder_1/feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Etransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOpNtransformer_decoder_1_feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6transformer_decoder_1/feedforward_output_dense/BiasAddBiasAddAtransformer_decoder_1/feedforward_output_dense/Tensordot:output:0Mtransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
2transformer_decoder_1/feedforward_dropout/IdentityIdentity?transformer_decoder_1/feedforward_output_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
transformer_decoder_1/add_2AddV2;transformer_decoder_1/feedforward_dropout/Identity:output:0Ctransformer_decoder_1/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Ktransformer_decoder_1/feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
9transformer_decoder_1/feedforward_layer_norm/moments/meanMeantransformer_decoder_1/add_2:z:0Ttransformer_decoder_1/feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Atransformer_decoder_1/feedforward_layer_norm/moments/StopGradientStopGradientBtransformer_decoder_1/feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Ftransformer_decoder_1/feedforward_layer_norm/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_1/add_2:z:0Jtransformer_decoder_1/feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Otransformer_decoder_1/feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
=transformer_decoder_1/feedforward_layer_norm/moments/varianceMeanJtransformer_decoder_1/feedforward_layer_norm/moments/SquaredDifference:z:0Xtransformer_decoder_1/feedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
<transformer_decoder_1/feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
:transformer_decoder_1/feedforward_layer_norm/batchnorm/addAddV2Ftransformer_decoder_1/feedforward_layer_norm/moments/variance:output:0Etransformer_decoder_1/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
<transformer_decoder_1/feedforward_layer_norm/batchnorm/RsqrtRsqrt>transformer_decoder_1/feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Itransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_decoder_1_feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:transformer_decoder_1/feedforward_layer_norm/batchnorm/mulMul@transformer_decoder_1/feedforward_layer_norm/batchnorm/Rsqrt:y:0Qtransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
<transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_1Multransformer_decoder_1/add_2:z:0>transformer_decoder_1/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
<transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_2MulBtransformer_decoder_1/feedforward_layer_norm/moments/mean:output:0>transformer_decoder_1/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Etransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOpNtransformer_decoder_1_feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:transformer_decoder_1/feedforward_layer_norm/batchnorm/subSubMtransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOp:value:0@transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
<transformer_decoder_1/feedforward_layer_norm/batchnorm/add_1AddV2@transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_1:z:0>transformer_decoder_1/feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
��'*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/Tensordot/ShapeShape@transformer_decoder_1/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transpose@transformer_decoder_1/feedforward_layer_norm/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������'b
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�'_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������'
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�'*
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������'s
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOpE^token_and_position_embedding/position_embedding/Slice/ReadVariableOp>^token_and_position_embedding/token_embedding/embedding_lookupJ^transformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOpL^transformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOpD^transformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOpH^transformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOpD^transformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOpF^transformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOpG^transformer_decoder/self_attention/attention_output/add/ReadVariableOpQ^transformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp:^transformer_decoder/self_attention/key/add/ReadVariableOpD^transformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp<^transformer_decoder/self_attention/query/add/ReadVariableOpF^transformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp<^transformer_decoder/self_attention/value/add/ReadVariableOpF^transformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOpG^transformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOpK^transformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOpL^transformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOpN^transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOpF^transformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOpJ^transformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOpF^transformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOpH^transformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOpI^transformer_decoder_1/self_attention/attention_output/add/ReadVariableOpS^transformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp<^transformer_decoder_1/self_attention/key/add/ReadVariableOpF^transformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp>^transformer_decoder_1/self_attention/query/add/ReadVariableOpH^transformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp>^transformer_decoder_1/self_attention/value/add/ReadVariableOpH^transformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOpI^transformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOpM^transformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2�
Dtoken_and_position_embedding/position_embedding/Slice/ReadVariableOpDtoken_and_position_embedding/position_embedding/Slice/ReadVariableOp2~
=token_and_position_embedding/token_embedding/embedding_lookup=token_and_position_embedding/token_embedding/embedding_lookup2�
Itransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOpItransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOp2�
Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOpKtransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOp2�
Ctransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOpCtransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOp2�
Gtransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOpGtransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOp2�
Ctransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOpCtransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOp2�
Etransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOpEtransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOp2�
Ftransformer_decoder/self_attention/attention_output/add/ReadVariableOpFtransformer_decoder/self_attention/attention_output/add/ReadVariableOp2�
Ptransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOpPtransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp2v
9transformer_decoder/self_attention/key/add/ReadVariableOp9transformer_decoder/self_attention/key/add/ReadVariableOp2�
Ctransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOpCtransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp2z
;transformer_decoder/self_attention/query/add/ReadVariableOp;transformer_decoder/self_attention/query/add/ReadVariableOp2�
Etransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOpEtransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp2z
;transformer_decoder/self_attention/value/add/ReadVariableOp;transformer_decoder/self_attention/value/add/ReadVariableOp2�
Etransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOpEtransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOp2�
Ftransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOpFtransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOp2�
Jtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOpJtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOp2�
Ktransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOpKtransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOp2�
Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOpMtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOp2�
Etransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOpEtransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOp2�
Itransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOpItransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOp2�
Etransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOpEtransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOp2�
Gtransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOpGtransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOp2�
Htransformer_decoder_1/self_attention/attention_output/add/ReadVariableOpHtransformer_decoder_1/self_attention/attention_output/add/ReadVariableOp2�
Rtransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOpRtransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp2z
;transformer_decoder_1/self_attention/key/add/ReadVariableOp;transformer_decoder_1/self_attention/key/add/ReadVariableOp2�
Etransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOpEtransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp2~
=transformer_decoder_1/self_attention/query/add/ReadVariableOp=transformer_decoder_1/self_attention/query/add/ReadVariableOp2�
Gtransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOpGtransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp2~
=transformer_decoder_1/self_attention/value/add/ReadVariableOp=transformer_decoder_1/self_attention/value/add/ReadVariableOp2�
Gtransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOpGtransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOp2�
Htransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOpHtransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOp2�
Ltransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOpLtransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
��
�
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_44098
decoder_sequenceQ
:self_attention_query_einsum_einsum_readvariableop_resource:�UB
0self_attention_query_add_readvariableop_resource:UO
8self_attention_key_einsum_einsum_readvariableop_resource:�U@
.self_attention_key_add_readvariableop_resource:UQ
:self_attention_value_einsum_einsum_readvariableop_resource:�UB
0self_attention_value_add_readvariableop_resource:U\
Eself_attention_attention_output_einsum_einsum_readvariableop_resource:U�J
;self_attention_attention_output_add_readvariableop_resource:	�N
?self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�J
;self_attention_layer_norm_batchnorm_readvariableop_resource:	�T
@feedforward_intermediate_dense_tensordot_readvariableop_resource:
��M
>feedforward_intermediate_dense_biasadd_readvariableop_resource:	�N
:feedforward_output_dense_tensordot_readvariableop_resource:
��G
8feedforward_output_dense_biasadd_readvariableop_resource:	�K
<feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�G
8feedforward_layer_norm_batchnorm_readvariableop_resource:	�
identity��5feedforward_intermediate_dense/BiasAdd/ReadVariableOp�7feedforward_intermediate_dense/Tensordot/ReadVariableOp�/feedforward_layer_norm/batchnorm/ReadVariableOp�3feedforward_layer_norm/batchnorm/mul/ReadVariableOp�/feedforward_output_dense/BiasAdd/ReadVariableOp�1feedforward_output_dense/Tensordot/ReadVariableOp�2self_attention/attention_output/add/ReadVariableOp�<self_attention/attention_output/einsum/Einsum/ReadVariableOp�%self_attention/key/add/ReadVariableOp�/self_attention/key/einsum/Einsum/ReadVariableOp�'self_attention/query/add/ReadVariableOp�1self_attention/query/einsum/Einsum/ReadVariableOp�'self_attention/value/add/ReadVariableOp�1self_attention/value/einsum/Einsum/ReadVariableOp�2self_attention_layer_norm/batchnorm/ReadVariableOp�6self_attention_layer_norm/batchnorm/mul/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
Shape_1Shapedecoder_sequence*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\

range/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: {
rangeRangerange/start:output:0range/Cast:y:0range/delta:output:0*

Tidx0*#
_output_shapes
:���������H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B : M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
addAddV2range:output:0Cast:y:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :l

ExpandDims
ExpandDimsadd:z:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������R
range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    R
range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
range_1/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
range_1Rangerange_1/start:output:0range_1/Cast:y:0range_1/delta:output:0*

Tidx0*#
_output_shapes
:���������~
GreaterEqualGreaterEqualExpandDims:output:0range_1:output:0*
T0*0
_output_shapes
:������������������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsGreaterEqual:z:0ExpandDims_1/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
packedPackstrided_slice:output:0strided_slice_3:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :�
BroadcastToBroadcastToExpandDims_1:output:0packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
1self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/query/einsum/EinsumEinsumdecoder_sequence9self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/query/add/ReadVariableOpReadVariableOp0self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/query/addAddV2+self_attention/query/einsum/Einsum:output:0/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp8self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
 self_attention/key/einsum/EinsumEinsumdecoder_sequence7self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
%self_attention/key/add/ReadVariableOpReadVariableOp.self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/key/addAddV2)self_attention/key/einsum/Einsum:output:0-self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
1self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/value/einsum/EinsumEinsumdecoder_sequence9self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/value/add/ReadVariableOpReadVariableOp0self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/value/addAddV2+self_attention/value/einsum/Einsum:output:0/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������UW
self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Uk
self_attention/CastCastself_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: U
self_attention/SqrtSqrtself_attention/Cast:y:0*
T0*
_output_shapes
: ]
self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
self_attention/truedivRealDiv!self_attention/truediv/x:output:0self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
self_attention/MulMulself_attention/query/add:z:0self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
self_attention/einsum/EinsumEinsumself_attention/key/add:z:0self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbeh
self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
self_attention/ExpandDims
ExpandDimsBroadcastTo:output:0&self_attention/ExpandDims/dim:output:0*
T0
*A
_output_shapes/
-:+����������������������������
self_attention/softmax_1/CastCast"self_attention/ExpandDims:output:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+���������������������������c
self_attention/softmax_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
self_attention/softmax_1/subSub'self_attention/softmax_1/sub/x:output:0!self_attention/softmax_1/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������c
self_attention/softmax_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
self_attention/softmax_1/mulMul self_attention/softmax_1/sub:z:0'self_attention/softmax_1/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax_1/addAddV2%self_attention/einsum/Einsum:output:0 self_attention/softmax_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
 self_attention/softmax_1/SoftmaxSoftmax self_attention/softmax_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/einsum_1/EinsumEinsum*self_attention/softmax_1/Softmax:softmax:0self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
<self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpEself_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
-self_attention/attention_output/einsum/EinsumEinsum'self_attention/einsum_1/Einsum:output:0Dself_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
2self_attention/attention_output/add/ReadVariableOpReadVariableOp;self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#self_attention/attention_output/addAddV26self_attention/attention_output/einsum/Einsum:output:0:self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
add_1AddV2'self_attention/attention_output/add:z:0decoder_sequence*
T0*5
_output_shapes#
!:��������������������
8self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&self_attention_layer_norm/moments/meanMean	add_1:z:0Aself_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
.self_attention_layer_norm/moments/StopGradientStopGradient/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
3self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference	add_1:z:07self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
<self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
*self_attention_layer_norm/moments/varianceMean7self_attention_layer_norm/moments/SquaredDifference:z:0Eself_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(n
)self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
'self_attention_layer_norm/batchnorm/addAddV23self_attention_layer_norm/moments/variance:output:02self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
)self_attention_layer_norm/batchnorm/RsqrtRsqrt+self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
6self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp?self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/mulMul-self_attention_layer_norm/batchnorm/Rsqrt:y:0>self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_1Mul	add_1:z:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_2Mul/self_attention_layer_norm/moments/mean:output:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
2self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOp;self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/subSub:self_attention_layer_norm/batchnorm/ReadVariableOp:value:0-self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/add_1AddV2-self_attention_layer_norm/batchnorm/mul_1:z:0+self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
7feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOp@feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
-feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
.feedforward_intermediate_dense/Tensordot/ShapeShape-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:x
6feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/GatherV2GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/free:output:0?feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/axes:output:0Afeedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
-feedforward_intermediate_dense/Tensordot/ProdProd:feedforward_intermediate_dense/Tensordot/GatherV2:output:07feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
/feedforward_intermediate_dense/Tensordot/Prod_1Prod<feedforward_intermediate_dense/Tensordot/GatherV2_1:output:09feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/feedforward_intermediate_dense/Tensordot/concatConcatV26feedforward_intermediate_dense/Tensordot/free:output:06feedforward_intermediate_dense/Tensordot/axes:output:0=feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
.feedforward_intermediate_dense/Tensordot/stackPack6feedforward_intermediate_dense/Tensordot/Prod:output:08feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
2feedforward_intermediate_dense/Tensordot/transpose	Transpose-self_attention_layer_norm/batchnorm/add_1:z:08feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
0feedforward_intermediate_dense/Tensordot/ReshapeReshape6feedforward_intermediate_dense/Tensordot/transpose:y:07feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
/feedforward_intermediate_dense/Tensordot/MatMulMatMul9feedforward_intermediate_dense/Tensordot/Reshape:output:0?feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
0feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�x
6feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/concat_1ConcatV2:feedforward_intermediate_dense/Tensordot/GatherV2:output:09feedforward_intermediate_dense/Tensordot/Const_2:output:0?feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_intermediate_dense/TensordotReshape9feedforward_intermediate_dense/Tensordot/MatMul:product:0:feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
5feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOp>feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&feedforward_intermediate_dense/BiasAddBiasAdd1feedforward_intermediate_dense/Tensordot:output:0=feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
#feedforward_intermediate_dense/ReluRelu/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
1feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOp:feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0q
'feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(feedforward_output_dense/Tensordot/ShapeShape1feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:r
0feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/GatherV2GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/free:output:09feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-feedforward_output_dense/Tensordot/GatherV2_1GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/axes:output:0;feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'feedforward_output_dense/Tensordot/ProdProd4feedforward_output_dense/Tensordot/GatherV2:output:01feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)feedforward_output_dense/Tensordot/Prod_1Prod6feedforward_output_dense/Tensordot/GatherV2_1:output:03feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)feedforward_output_dense/Tensordot/concatConcatV20feedforward_output_dense/Tensordot/free:output:00feedforward_output_dense/Tensordot/axes:output:07feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_output_dense/Tensordot/stackPack0feedforward_output_dense/Tensordot/Prod:output:02feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,feedforward_output_dense/Tensordot/transpose	Transpose1feedforward_intermediate_dense/Relu:activations:02feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
*feedforward_output_dense/Tensordot/ReshapeReshape0feedforward_output_dense/Tensordot/transpose:y:01feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)feedforward_output_dense/Tensordot/MatMulMatMul3feedforward_output_dense/Tensordot/Reshape:output:09feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
*feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�r
0feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/concat_1ConcatV24feedforward_output_dense/Tensordot/GatherV2:output:03feedforward_output_dense/Tensordot/Const_2:output:09feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"feedforward_output_dense/TensordotReshape3feedforward_output_dense/Tensordot/MatMul:product:04feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOp8feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 feedforward_output_dense/BiasAddBiasAdd+feedforward_output_dense/Tensordot:output:07feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
add_2AddV2)feedforward_output_dense/BiasAdd:output:0-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:�������������������
5feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#feedforward_layer_norm/moments/meanMean	add_2:z:0>feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
+feedforward_layer_norm/moments/StopGradientStopGradient,feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
0feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference	add_2:z:04feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
9feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'feedforward_layer_norm/moments/varianceMean4feedforward_layer_norm/moments/SquaredDifference:z:0Bfeedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(k
&feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$feedforward_layer_norm/batchnorm/addAddV20feedforward_layer_norm/moments/variance:output:0/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
&feedforward_layer_norm/batchnorm/RsqrtRsqrt(feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
3feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp<feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/mulMul*feedforward_layer_norm/batchnorm/Rsqrt:y:0;feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_1Mul	add_2:z:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_2Mul,feedforward_layer_norm/moments/mean:output:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOp8feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/subSub7feedforward_layer_norm/batchnorm/ReadVariableOp:value:0*feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/add_1AddV2*feedforward_layer_norm/batchnorm/mul_1:z:0(feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentity*feedforward_layer_norm/batchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp6^feedforward_intermediate_dense/BiasAdd/ReadVariableOp8^feedforward_intermediate_dense/Tensordot/ReadVariableOp0^feedforward_layer_norm/batchnorm/ReadVariableOp4^feedforward_layer_norm/batchnorm/mul/ReadVariableOp0^feedforward_output_dense/BiasAdd/ReadVariableOp2^feedforward_output_dense/Tensordot/ReadVariableOp3^self_attention/attention_output/add/ReadVariableOp=^self_attention/attention_output/einsum/Einsum/ReadVariableOp&^self_attention/key/add/ReadVariableOp0^self_attention/key/einsum/Einsum/ReadVariableOp(^self_attention/query/add/ReadVariableOp2^self_attention/query/einsum/Einsum/ReadVariableOp(^self_attention/value/add/ReadVariableOp2^self_attention/value/einsum/Einsum/ReadVariableOp3^self_attention_layer_norm/batchnorm/ReadVariableOp7^self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 2n
5feedforward_intermediate_dense/BiasAdd/ReadVariableOp5feedforward_intermediate_dense/BiasAdd/ReadVariableOp2r
7feedforward_intermediate_dense/Tensordot/ReadVariableOp7feedforward_intermediate_dense/Tensordot/ReadVariableOp2b
/feedforward_layer_norm/batchnorm/ReadVariableOp/feedforward_layer_norm/batchnorm/ReadVariableOp2j
3feedforward_layer_norm/batchnorm/mul/ReadVariableOp3feedforward_layer_norm/batchnorm/mul/ReadVariableOp2b
/feedforward_output_dense/BiasAdd/ReadVariableOp/feedforward_output_dense/BiasAdd/ReadVariableOp2f
1feedforward_output_dense/Tensordot/ReadVariableOp1feedforward_output_dense/Tensordot/ReadVariableOp2h
2self_attention/attention_output/add/ReadVariableOp2self_attention/attention_output/add/ReadVariableOp2|
<self_attention/attention_output/einsum/Einsum/ReadVariableOp<self_attention/attention_output/einsum/Einsum/ReadVariableOp2N
%self_attention/key/add/ReadVariableOp%self_attention/key/add/ReadVariableOp2b
/self_attention/key/einsum/Einsum/ReadVariableOp/self_attention/key/einsum/Einsum/ReadVariableOp2R
'self_attention/query/add/ReadVariableOp'self_attention/query/add/ReadVariableOp2f
1self_attention/query/einsum/Einsum/ReadVariableOp1self_attention/query/einsum/Einsum/ReadVariableOp2R
'self_attention/value/add/ReadVariableOp'self_attention/value/add/ReadVariableOp2f
1self_attention/value/einsum/Einsum/ReadVariableOp1self_attention/value/einsum/Einsum/ReadVariableOp2h
2self_attention_layer_norm/batchnorm/ReadVariableOp2self_attention_layer_norm/batchnorm/ReadVariableOp2p
6self_attention_layer_norm/batchnorm/mul/ReadVariableOp6self_attention_layer_norm/batchnorm/mul/ReadVariableOp:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
�
�	
%__inference_model_layer_call_fn_45107

inputs
unknown:
�'�
	unknown_0:
�� 
	unknown_1:�U
	unknown_2:U 
	unknown_3:�U
	unknown_4:U 
	unknown_5:�U
	unknown_6:U 
	unknown_7:U�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�!

unknown_17:�U

unknown_18:U!

unknown_19:�U

unknown_20:U!

unknown_21:�U

unknown_22:U!

unknown_23:U�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:
��

unknown_28:	�

unknown_29:
��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:
��'

unknown_34:	�'
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_44552}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�%
�
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_45967

inputs:
&token_embedding_embedding_lookup_45939:
�'�D
0position_embedding_slice_readvariableop_resource:
��
identity��'position_embedding/Slice/ReadVariableOp� token_embedding/embedding_lookup�
 token_embedding/embedding_lookupResourceGather&token_embedding_embedding_lookup_45939inputs*
Tindices0*9
_class/
-+loc:@token_embedding/embedding_lookup/45939*5
_output_shapes#
!:�������������������*
dtype0�
)token_embedding/embedding_lookup/IdentityIdentity)token_embedding/embedding_lookup:output:0*
T0*9
_class/
-+loc:@token_embedding/embedding_lookup/45939*5
_output_shapes#
!:��������������������
+token_embedding/embedding_lookup/Identity_1Identity2token_embedding/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:�������������������\
token_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
token_embedding/NotEqualNotEqualinputs#token_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:������������������|
position_embedding/ShapeShape4token_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:p
&position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 position_embedding/strided_sliceStridedSlice!position_embedding/Shape:output:0/position_embedding/strided_slice/stack:output:01position_embedding/strided_slice/stack_1:output:01position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
(position_embedding/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*position_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*position_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"position_embedding/strided_slice_1StridedSlice!position_embedding/Shape:output:01position_embedding/strided_slice_1/stack:output:03position_embedding/strided_slice_1/stack_1:output:03position_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
'position_embedding/Slice/ReadVariableOpReadVariableOp0position_embedding_slice_readvariableop_resource* 
_output_shapes
:
��*
dtype0o
position_embedding/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        b
position_embedding/Slice/size/1Const*
_output_shapes
: *
dtype0*
value
B :��
position_embedding/Slice/sizePack+position_embedding/strided_slice_1:output:0(position_embedding/Slice/size/1:output:0*
N*
T0*
_output_shapes
:�
position_embedding/SliceSlice/position_embedding/Slice/ReadVariableOp:value:0'position_embedding/Slice/begin:output:0&position_embedding/Slice/size:output:0*
Index0*
T0*(
_output_shapes
:����������^
position_embedding/packed/2Const*
_output_shapes
: *
dtype0*
value
B :��
position_embedding/packedPack)position_embedding/strided_slice:output:0+position_embedding/strided_slice_1:output:0$position_embedding/packed/2:output:0*
N*
T0*
_output_shapes
:Y
position_embedding/RankConst*
_output_shapes
: *
dtype0*
value	B :�
position_embedding/BroadcastToBroadcastTo!position_embedding/Slice:output:0"position_embedding/packed:output:0*
T0*5
_output_shapes#
!:��������������������
addAddV24token_embedding/embedding_lookup/Identity_1:output:0'position_embedding/BroadcastTo:output:0*
T0*5
_output_shapes#
!:�������������������d
IdentityIdentityadd:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp(^position_embedding/Slice/ReadVariableOp!^token_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������: : 2R
'position_embedding/Slice/ReadVariableOp'position_embedding/Slice/ReadVariableOp2D
 token_embedding/embedding_lookup token_embedding/embedding_lookup:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
��
�:
__inference__traced_save_47226
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableopH
Dsavev2_token_and_position_embedding_embeddings_1_read_readvariableopF
Bsavev2_token_and_position_embedding_embeddings_read_readvariableopN
Jsavev2_transformer_decoder_self_attention_query_kernel_read_readvariableopL
Hsavev2_transformer_decoder_self_attention_query_bias_read_readvariableopL
Hsavev2_transformer_decoder_self_attention_key_kernel_read_readvariableopJ
Fsavev2_transformer_decoder_self_attention_key_bias_read_readvariableopN
Jsavev2_transformer_decoder_self_attention_value_kernel_read_readvariableopL
Hsavev2_transformer_decoder_self_attention_value_bias_read_readvariableopY
Usavev2_transformer_decoder_self_attention_attention_output_kernel_read_readvariableopW
Ssavev2_transformer_decoder_self_attention_attention_output_bias_read_readvariableop&
"savev2_gamma_3_read_readvariableop%
!savev2_beta_3_read_readvariableop'
#savev2_kernel_3_read_readvariableop%
!savev2_bias_3_read_readvariableop'
#savev2_kernel_2_read_readvariableop%
!savev2_bias_2_read_readvariableop&
"savev2_gamma_2_read_readvariableop%
!savev2_beta_2_read_readvariableopP
Lsavev2_transformer_decoder_1_self_attention_query_kernel_read_readvariableopN
Jsavev2_transformer_decoder_1_self_attention_query_bias_read_readvariableopN
Jsavev2_transformer_decoder_1_self_attention_key_kernel_read_readvariableopL
Hsavev2_transformer_decoder_1_self_attention_key_bias_read_readvariableopP
Lsavev2_transformer_decoder_1_self_attention_value_kernel_read_readvariableopN
Jsavev2_transformer_decoder_1_self_attention_value_bias_read_readvariableop[
Wsavev2_transformer_decoder_1_self_attention_attention_output_kernel_read_readvariableopY
Usavev2_transformer_decoder_1_self_attention_attention_output_bias_read_readvariableop&
"savev2_gamma_1_read_readvariableop%
!savev2_beta_1_read_readvariableop'
#savev2_kernel_1_read_readvariableop%
!savev2_bias_1_read_readvariableop%
!savev2_kernel_read_readvariableop#
savev2_bias_read_readvariableop$
 savev2_gamma_read_readvariableop#
savev2_beta_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_aggregate_crossentropy_read_readvariableop0
,savev2_number_of_samples_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableopO
Ksavev2_adam_token_and_position_embedding_embeddings_m_1_read_readvariableopM
Isavev2_adam_token_and_position_embedding_embeddings_m_read_readvariableopU
Qsavev2_adam_transformer_decoder_self_attention_query_kernel_m_read_readvariableopS
Osavev2_adam_transformer_decoder_self_attention_query_bias_m_read_readvariableopS
Osavev2_adam_transformer_decoder_self_attention_key_kernel_m_read_readvariableopQ
Msavev2_adam_transformer_decoder_self_attention_key_bias_m_read_readvariableopU
Qsavev2_adam_transformer_decoder_self_attention_value_kernel_m_read_readvariableopS
Osavev2_adam_transformer_decoder_self_attention_value_bias_m_read_readvariableop`
\savev2_adam_transformer_decoder_self_attention_attention_output_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_decoder_self_attention_attention_output_bias_m_read_readvariableop-
)savev2_adam_gamma_m_3_read_readvariableop,
(savev2_adam_beta_m_3_read_readvariableop.
*savev2_adam_kernel_m_3_read_readvariableop,
(savev2_adam_bias_m_3_read_readvariableop.
*savev2_adam_kernel_m_2_read_readvariableop,
(savev2_adam_bias_m_2_read_readvariableop-
)savev2_adam_gamma_m_2_read_readvariableop,
(savev2_adam_beta_m_2_read_readvariableopW
Ssavev2_adam_transformer_decoder_1_self_attention_query_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_decoder_1_self_attention_query_bias_m_read_readvariableopU
Qsavev2_adam_transformer_decoder_1_self_attention_key_kernel_m_read_readvariableopS
Osavev2_adam_transformer_decoder_1_self_attention_key_bias_m_read_readvariableopW
Ssavev2_adam_transformer_decoder_1_self_attention_value_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_decoder_1_self_attention_value_bias_m_read_readvariableopb
^savev2_adam_transformer_decoder_1_self_attention_attention_output_kernel_m_read_readvariableop`
\savev2_adam_transformer_decoder_1_self_attention_attention_output_bias_m_read_readvariableop-
)savev2_adam_gamma_m_1_read_readvariableop,
(savev2_adam_beta_m_1_read_readvariableop.
*savev2_adam_kernel_m_1_read_readvariableop,
(savev2_adam_bias_m_1_read_readvariableop,
(savev2_adam_kernel_m_read_readvariableop*
&savev2_adam_bias_m_read_readvariableop+
'savev2_adam_gamma_m_read_readvariableop*
&savev2_adam_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableopO
Ksavev2_adam_token_and_position_embedding_embeddings_v_1_read_readvariableopM
Isavev2_adam_token_and_position_embedding_embeddings_v_read_readvariableopU
Qsavev2_adam_transformer_decoder_self_attention_query_kernel_v_read_readvariableopS
Osavev2_adam_transformer_decoder_self_attention_query_bias_v_read_readvariableopS
Osavev2_adam_transformer_decoder_self_attention_key_kernel_v_read_readvariableopQ
Msavev2_adam_transformer_decoder_self_attention_key_bias_v_read_readvariableopU
Qsavev2_adam_transformer_decoder_self_attention_value_kernel_v_read_readvariableopS
Osavev2_adam_transformer_decoder_self_attention_value_bias_v_read_readvariableop`
\savev2_adam_transformer_decoder_self_attention_attention_output_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_decoder_self_attention_attention_output_bias_v_read_readvariableop-
)savev2_adam_gamma_v_3_read_readvariableop,
(savev2_adam_beta_v_3_read_readvariableop.
*savev2_adam_kernel_v_3_read_readvariableop,
(savev2_adam_bias_v_3_read_readvariableop.
*savev2_adam_kernel_v_2_read_readvariableop,
(savev2_adam_bias_v_2_read_readvariableop-
)savev2_adam_gamma_v_2_read_readvariableop,
(savev2_adam_beta_v_2_read_readvariableopW
Ssavev2_adam_transformer_decoder_1_self_attention_query_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_decoder_1_self_attention_query_bias_v_read_readvariableopU
Qsavev2_adam_transformer_decoder_1_self_attention_key_kernel_v_read_readvariableopS
Osavev2_adam_transformer_decoder_1_self_attention_key_bias_v_read_readvariableopW
Ssavev2_adam_transformer_decoder_1_self_attention_value_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_decoder_1_self_attention_value_bias_v_read_readvariableopb
^savev2_adam_transformer_decoder_1_self_attention_attention_output_kernel_v_read_readvariableop`
\savev2_adam_transformer_decoder_1_self_attention_attention_output_bias_v_read_readvariableop-
)savev2_adam_gamma_v_1_read_readvariableop,
(savev2_adam_beta_v_1_read_readvariableop.
*savev2_adam_kernel_v_1_read_readvariableop,
(savev2_adam_bias_v_1_read_readvariableop,
(savev2_adam_kernel_v_read_readvariableop*
&savev2_adam_bias_v_read_readvariableop+
'savev2_adam_gamma_v_read_readvariableop*
&savev2_adam_beta_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �7
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:v*
dtype0*�6
value�6B�6vB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBEkeras_api/metrics/1/aggregate_crossentropy/.ATTRIBUTES/VARIABLE_VALUEB@keras_api/metrics/1/number_of_samples/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:v*
dtype0*�
value�B�vB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �8
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopDsavev2_token_and_position_embedding_embeddings_1_read_readvariableopBsavev2_token_and_position_embedding_embeddings_read_readvariableopJsavev2_transformer_decoder_self_attention_query_kernel_read_readvariableopHsavev2_transformer_decoder_self_attention_query_bias_read_readvariableopHsavev2_transformer_decoder_self_attention_key_kernel_read_readvariableopFsavev2_transformer_decoder_self_attention_key_bias_read_readvariableopJsavev2_transformer_decoder_self_attention_value_kernel_read_readvariableopHsavev2_transformer_decoder_self_attention_value_bias_read_readvariableopUsavev2_transformer_decoder_self_attention_attention_output_kernel_read_readvariableopSsavev2_transformer_decoder_self_attention_attention_output_bias_read_readvariableop"savev2_gamma_3_read_readvariableop!savev2_beta_3_read_readvariableop#savev2_kernel_3_read_readvariableop!savev2_bias_3_read_readvariableop#savev2_kernel_2_read_readvariableop!savev2_bias_2_read_readvariableop"savev2_gamma_2_read_readvariableop!savev2_beta_2_read_readvariableopLsavev2_transformer_decoder_1_self_attention_query_kernel_read_readvariableopJsavev2_transformer_decoder_1_self_attention_query_bias_read_readvariableopJsavev2_transformer_decoder_1_self_attention_key_kernel_read_readvariableopHsavev2_transformer_decoder_1_self_attention_key_bias_read_readvariableopLsavev2_transformer_decoder_1_self_attention_value_kernel_read_readvariableopJsavev2_transformer_decoder_1_self_attention_value_bias_read_readvariableopWsavev2_transformer_decoder_1_self_attention_attention_output_kernel_read_readvariableopUsavev2_transformer_decoder_1_self_attention_attention_output_bias_read_readvariableop"savev2_gamma_1_read_readvariableop!savev2_beta_1_read_readvariableop#savev2_kernel_1_read_readvariableop!savev2_bias_1_read_readvariableop!savev2_kernel_read_readvariableopsavev2_bias_read_readvariableop savev2_gamma_read_readvariableopsavev2_beta_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_aggregate_crossentropy_read_readvariableop,savev2_number_of_samples_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableopKsavev2_adam_token_and_position_embedding_embeddings_m_1_read_readvariableopIsavev2_adam_token_and_position_embedding_embeddings_m_read_readvariableopQsavev2_adam_transformer_decoder_self_attention_query_kernel_m_read_readvariableopOsavev2_adam_transformer_decoder_self_attention_query_bias_m_read_readvariableopOsavev2_adam_transformer_decoder_self_attention_key_kernel_m_read_readvariableopMsavev2_adam_transformer_decoder_self_attention_key_bias_m_read_readvariableopQsavev2_adam_transformer_decoder_self_attention_value_kernel_m_read_readvariableopOsavev2_adam_transformer_decoder_self_attention_value_bias_m_read_readvariableop\savev2_adam_transformer_decoder_self_attention_attention_output_kernel_m_read_readvariableopZsavev2_adam_transformer_decoder_self_attention_attention_output_bias_m_read_readvariableop)savev2_adam_gamma_m_3_read_readvariableop(savev2_adam_beta_m_3_read_readvariableop*savev2_adam_kernel_m_3_read_readvariableop(savev2_adam_bias_m_3_read_readvariableop*savev2_adam_kernel_m_2_read_readvariableop(savev2_adam_bias_m_2_read_readvariableop)savev2_adam_gamma_m_2_read_readvariableop(savev2_adam_beta_m_2_read_readvariableopSsavev2_adam_transformer_decoder_1_self_attention_query_kernel_m_read_readvariableopQsavev2_adam_transformer_decoder_1_self_attention_query_bias_m_read_readvariableopQsavev2_adam_transformer_decoder_1_self_attention_key_kernel_m_read_readvariableopOsavev2_adam_transformer_decoder_1_self_attention_key_bias_m_read_readvariableopSsavev2_adam_transformer_decoder_1_self_attention_value_kernel_m_read_readvariableopQsavev2_adam_transformer_decoder_1_self_attention_value_bias_m_read_readvariableop^savev2_adam_transformer_decoder_1_self_attention_attention_output_kernel_m_read_readvariableop\savev2_adam_transformer_decoder_1_self_attention_attention_output_bias_m_read_readvariableop)savev2_adam_gamma_m_1_read_readvariableop(savev2_adam_beta_m_1_read_readvariableop*savev2_adam_kernel_m_1_read_readvariableop(savev2_adam_bias_m_1_read_readvariableop(savev2_adam_kernel_m_read_readvariableop&savev2_adam_bias_m_read_readvariableop'savev2_adam_gamma_m_read_readvariableop&savev2_adam_beta_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopKsavev2_adam_token_and_position_embedding_embeddings_v_1_read_readvariableopIsavev2_adam_token_and_position_embedding_embeddings_v_read_readvariableopQsavev2_adam_transformer_decoder_self_attention_query_kernel_v_read_readvariableopOsavev2_adam_transformer_decoder_self_attention_query_bias_v_read_readvariableopOsavev2_adam_transformer_decoder_self_attention_key_kernel_v_read_readvariableopMsavev2_adam_transformer_decoder_self_attention_key_bias_v_read_readvariableopQsavev2_adam_transformer_decoder_self_attention_value_kernel_v_read_readvariableopOsavev2_adam_transformer_decoder_self_attention_value_bias_v_read_readvariableop\savev2_adam_transformer_decoder_self_attention_attention_output_kernel_v_read_readvariableopZsavev2_adam_transformer_decoder_self_attention_attention_output_bias_v_read_readvariableop)savev2_adam_gamma_v_3_read_readvariableop(savev2_adam_beta_v_3_read_readvariableop*savev2_adam_kernel_v_3_read_readvariableop(savev2_adam_bias_v_3_read_readvariableop*savev2_adam_kernel_v_2_read_readvariableop(savev2_adam_bias_v_2_read_readvariableop)savev2_adam_gamma_v_2_read_readvariableop(savev2_adam_beta_v_2_read_readvariableopSsavev2_adam_transformer_decoder_1_self_attention_query_kernel_v_read_readvariableopQsavev2_adam_transformer_decoder_1_self_attention_query_bias_v_read_readvariableopQsavev2_adam_transformer_decoder_1_self_attention_key_kernel_v_read_readvariableopOsavev2_adam_transformer_decoder_1_self_attention_key_bias_v_read_readvariableopSsavev2_adam_transformer_decoder_1_self_attention_value_kernel_v_read_readvariableopQsavev2_adam_transformer_decoder_1_self_attention_value_bias_v_read_readvariableop^savev2_adam_transformer_decoder_1_self_attention_attention_output_kernel_v_read_readvariableop\savev2_adam_transformer_decoder_1_self_attention_attention_output_bias_v_read_readvariableop)savev2_adam_gamma_v_1_read_readvariableop(savev2_adam_beta_v_1_read_readvariableop*savev2_adam_kernel_v_1_read_readvariableop(savev2_adam_bias_v_1_read_readvariableop(savev2_adam_kernel_v_read_readvariableop&savev2_adam_bias_v_read_readvariableop'savev2_adam_gamma_v_read_readvariableop&savev2_adam_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypesz
x2v	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��':�':
�'�:
��:�U:U:�U:U:�U:U:U�:�:�:�:
��:�:
��:�:�:�:�U:U:�U:U:�U:U:U�:�:�:�:
��:�:
��:�:�:�: : : : : : : : : :
��':�':
�'�:
��:�U:U:�U:U:�U:U:U�:�:�:�:
��:�:
��:�:�:�:�U:U:�U:U:�U:U:U�:�:�:�:
��:�:
��:�:�:�:
��':�':
�'�:
��:�U:U:�U:U:�U:U:U�:�:�:�:
��:�:
��:�:�:�:�U:U:�U:U:�U:U:U�:�:�:�:
��:�:
��:�:�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��':!

_output_shapes	
:�':&"
 
_output_shapes
:
�'�:&"
 
_output_shapes
:
��:)%
#
_output_shapes
:�U:$ 

_output_shapes

:U:)%
#
_output_shapes
:�U:$ 

_output_shapes

:U:)	%
#
_output_shapes
:�U:$
 

_output_shapes

:U:)%
#
_output_shapes
:U�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:)%
#
_output_shapes
:�U:$ 

_output_shapes

:U:)%
#
_output_shapes
:�U:$ 

_output_shapes

:U:)%
#
_output_shapes
:�U:$ 

_output_shapes

:U:)%
#
_output_shapes
:U�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:! 

_output_shapes	
:�:&!"
 
_output_shapes
:
��:!"

_output_shapes	
:�:!#

_output_shapes	
:�:!$

_output_shapes	
:�:%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :&."
 
_output_shapes
:
��':!/

_output_shapes	
:�':&0"
 
_output_shapes
:
�'�:&1"
 
_output_shapes
:
��:)2%
#
_output_shapes
:�U:$3 

_output_shapes

:U:)4%
#
_output_shapes
:�U:$5 

_output_shapes

:U:)6%
#
_output_shapes
:�U:$7 

_output_shapes

:U:)8%
#
_output_shapes
:U�:!9

_output_shapes	
:�:!:

_output_shapes	
:�:!;

_output_shapes	
:�:&<"
 
_output_shapes
:
��:!=

_output_shapes	
:�:&>"
 
_output_shapes
:
��:!?

_output_shapes	
:�:!@

_output_shapes	
:�:!A

_output_shapes	
:�:)B%
#
_output_shapes
:�U:$C 

_output_shapes

:U:)D%
#
_output_shapes
:�U:$E 

_output_shapes

:U:)F%
#
_output_shapes
:�U:$G 

_output_shapes

:U:)H%
#
_output_shapes
:U�:!I

_output_shapes	
:�:!J

_output_shapes	
:�:!K

_output_shapes	
:�:&L"
 
_output_shapes
:
��:!M

_output_shapes	
:�:&N"
 
_output_shapes
:
��:!O

_output_shapes	
:�:!P

_output_shapes	
:�:!Q

_output_shapes	
:�:&R"
 
_output_shapes
:
��':!S

_output_shapes	
:�':&T"
 
_output_shapes
:
�'�:&U"
 
_output_shapes
:
��:)V%
#
_output_shapes
:�U:$W 

_output_shapes

:U:)X%
#
_output_shapes
:�U:$Y 

_output_shapes

:U:)Z%
#
_output_shapes
:�U:$[ 

_output_shapes

:U:)\%
#
_output_shapes
:U�:!]

_output_shapes	
:�:!^

_output_shapes	
:�:!_

_output_shapes	
:�:&`"
 
_output_shapes
:
��:!a

_output_shapes	
:�:&b"
 
_output_shapes
:
��:!c

_output_shapes	
:�:!d

_output_shapes	
:�:!e

_output_shapes	
:�:)f%
#
_output_shapes
:�U:$g 

_output_shapes

:U:)h%
#
_output_shapes
:�U:$i 

_output_shapes

:U:)j%
#
_output_shapes
:�U:$k 

_output_shapes

:U:)l%
#
_output_shapes
:U�:!m

_output_shapes	
:�:!n

_output_shapes	
:�:!o

_output_shapes	
:�:&p"
 
_output_shapes
:
��:!q

_output_shapes	
:�:&r"
 
_output_shapes
:
��:!s

_output_shapes	
:�:!t

_output_shapes	
:�:!u

_output_shapes	
:�:v

_output_shapes
: 
�
�
@__inference_dense_layer_call_and_return_conditional_losses_46852

inputs5
!tensordot_readvariableop_resource:
��'.
biasadd_readvariableop_resource:	�'
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��'*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������'\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�'Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������'s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�'*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������'m
IdentityIdentityBiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
<__inference_token_and_position_embedding_layer_call_fn_45936

inputs
unknown:
�'�
	unknown_0:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_43335}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
��
�
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_46217
decoder_sequenceQ
:self_attention_query_einsum_einsum_readvariableop_resource:�UB
0self_attention_query_add_readvariableop_resource:UO
8self_attention_key_einsum_einsum_readvariableop_resource:�U@
.self_attention_key_add_readvariableop_resource:UQ
:self_attention_value_einsum_einsum_readvariableop_resource:�UB
0self_attention_value_add_readvariableop_resource:U\
Eself_attention_attention_output_einsum_einsum_readvariableop_resource:U�J
;self_attention_attention_output_add_readvariableop_resource:	�N
?self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�J
;self_attention_layer_norm_batchnorm_readvariableop_resource:	�T
@feedforward_intermediate_dense_tensordot_readvariableop_resource:
��M
>feedforward_intermediate_dense_biasadd_readvariableop_resource:	�N
:feedforward_output_dense_tensordot_readvariableop_resource:
��G
8feedforward_output_dense_biasadd_readvariableop_resource:	�K
<feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�G
8feedforward_layer_norm_batchnorm_readvariableop_resource:	�
identity��5feedforward_intermediate_dense/BiasAdd/ReadVariableOp�7feedforward_intermediate_dense/Tensordot/ReadVariableOp�/feedforward_layer_norm/batchnorm/ReadVariableOp�3feedforward_layer_norm/batchnorm/mul/ReadVariableOp�/feedforward_output_dense/BiasAdd/ReadVariableOp�1feedforward_output_dense/Tensordot/ReadVariableOp�2self_attention/attention_output/add/ReadVariableOp�<self_attention/attention_output/einsum/Einsum/ReadVariableOp�%self_attention/key/add/ReadVariableOp�/self_attention/key/einsum/Einsum/ReadVariableOp�'self_attention/query/add/ReadVariableOp�1self_attention/query/einsum/Einsum/ReadVariableOp�'self_attention/value/add/ReadVariableOp�1self_attention/value/einsum/Einsum/ReadVariableOp�2self_attention_layer_norm/batchnorm/ReadVariableOp�6self_attention_layer_norm/batchnorm/mul/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
Shape_1Shapedecoder_sequence*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\

range/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: {
rangeRangerange/start:output:0range/Cast:y:0range/delta:output:0*

Tidx0*#
_output_shapes
:���������H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B : M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
addAddV2range:output:0Cast:y:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :l

ExpandDims
ExpandDimsadd:z:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������R
range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    R
range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
range_1/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
range_1Rangerange_1/start:output:0range_1/Cast:y:0range_1/delta:output:0*

Tidx0*#
_output_shapes
:���������~
GreaterEqualGreaterEqualExpandDims:output:0range_1:output:0*
T0*0
_output_shapes
:������������������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsGreaterEqual:z:0ExpandDims_1/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
packedPackstrided_slice:output:0strided_slice_3:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :�
BroadcastToBroadcastToExpandDims_1:output:0packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
1self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/query/einsum/EinsumEinsumdecoder_sequence9self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/query/add/ReadVariableOpReadVariableOp0self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/query/addAddV2+self_attention/query/einsum/Einsum:output:0/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp8self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
 self_attention/key/einsum/EinsumEinsumdecoder_sequence7self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
%self_attention/key/add/ReadVariableOpReadVariableOp.self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/key/addAddV2)self_attention/key/einsum/Einsum:output:0-self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
1self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/value/einsum/EinsumEinsumdecoder_sequence9self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/value/add/ReadVariableOpReadVariableOp0self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/value/addAddV2+self_attention/value/einsum/Einsum:output:0/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������UW
self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Uk
self_attention/CastCastself_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: U
self_attention/SqrtSqrtself_attention/Cast:y:0*
T0*
_output_shapes
: ]
self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
self_attention/truedivRealDiv!self_attention/truediv/x:output:0self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
self_attention/MulMulself_attention/query/add:z:0self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
self_attention/einsum/EinsumEinsumself_attention/key/add:z:0self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbeh
self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
self_attention/ExpandDims
ExpandDimsBroadcastTo:output:0&self_attention/ExpandDims/dim:output:0*
T0
*A
_output_shapes/
-:+����������������������������
self_attention/softmax/CastCast"self_attention/ExpandDims:output:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+���������������������������a
self_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
self_attention/softmax/subSub%self_attention/softmax/sub/x:output:0self_attention/softmax/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������a
self_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
self_attention/softmax/mulMulself_attention/softmax/sub:z:0%self_attention/softmax/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax/addAddV2%self_attention/einsum/Einsum:output:0self_attention/softmax/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax/SoftmaxSoftmaxself_attention/softmax/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/dropout/IdentityIdentity(self_attention/softmax/Softmax:softmax:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/einsum_1/EinsumEinsum(self_attention/dropout/Identity:output:0self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
<self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpEself_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
-self_attention/attention_output/einsum/EinsumEinsum'self_attention/einsum_1/Einsum:output:0Dself_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
2self_attention/attention_output/add/ReadVariableOpReadVariableOp;self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#self_attention/attention_output/addAddV26self_attention/attention_output/einsum/Einsum:output:0:self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
self_attention_dropout/IdentityIdentity'self_attention/attention_output/add:z:0*
T0*5
_output_shapes#
!:��������������������
add_1AddV2(self_attention_dropout/Identity:output:0decoder_sequence*
T0*5
_output_shapes#
!:��������������������
8self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&self_attention_layer_norm/moments/meanMean	add_1:z:0Aself_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
.self_attention_layer_norm/moments/StopGradientStopGradient/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
3self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference	add_1:z:07self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
<self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
*self_attention_layer_norm/moments/varianceMean7self_attention_layer_norm/moments/SquaredDifference:z:0Eself_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(n
)self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
'self_attention_layer_norm/batchnorm/addAddV23self_attention_layer_norm/moments/variance:output:02self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
)self_attention_layer_norm/batchnorm/RsqrtRsqrt+self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
6self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp?self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/mulMul-self_attention_layer_norm/batchnorm/Rsqrt:y:0>self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_1Mul	add_1:z:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_2Mul/self_attention_layer_norm/moments/mean:output:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
2self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOp;self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/subSub:self_attention_layer_norm/batchnorm/ReadVariableOp:value:0-self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/add_1AddV2-self_attention_layer_norm/batchnorm/mul_1:z:0+self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
7feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOp@feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
-feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
.feedforward_intermediate_dense/Tensordot/ShapeShape-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:x
6feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/GatherV2GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/free:output:0?feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/axes:output:0Afeedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
-feedforward_intermediate_dense/Tensordot/ProdProd:feedforward_intermediate_dense/Tensordot/GatherV2:output:07feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
/feedforward_intermediate_dense/Tensordot/Prod_1Prod<feedforward_intermediate_dense/Tensordot/GatherV2_1:output:09feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/feedforward_intermediate_dense/Tensordot/concatConcatV26feedforward_intermediate_dense/Tensordot/free:output:06feedforward_intermediate_dense/Tensordot/axes:output:0=feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
.feedforward_intermediate_dense/Tensordot/stackPack6feedforward_intermediate_dense/Tensordot/Prod:output:08feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
2feedforward_intermediate_dense/Tensordot/transpose	Transpose-self_attention_layer_norm/batchnorm/add_1:z:08feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
0feedforward_intermediate_dense/Tensordot/ReshapeReshape6feedforward_intermediate_dense/Tensordot/transpose:y:07feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
/feedforward_intermediate_dense/Tensordot/MatMulMatMul9feedforward_intermediate_dense/Tensordot/Reshape:output:0?feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
0feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�x
6feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/concat_1ConcatV2:feedforward_intermediate_dense/Tensordot/GatherV2:output:09feedforward_intermediate_dense/Tensordot/Const_2:output:0?feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_intermediate_dense/TensordotReshape9feedforward_intermediate_dense/Tensordot/MatMul:product:0:feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
5feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOp>feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&feedforward_intermediate_dense/BiasAddBiasAdd1feedforward_intermediate_dense/Tensordot:output:0=feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
#feedforward_intermediate_dense/ReluRelu/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
1feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOp:feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0q
'feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(feedforward_output_dense/Tensordot/ShapeShape1feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:r
0feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/GatherV2GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/free:output:09feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-feedforward_output_dense/Tensordot/GatherV2_1GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/axes:output:0;feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'feedforward_output_dense/Tensordot/ProdProd4feedforward_output_dense/Tensordot/GatherV2:output:01feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)feedforward_output_dense/Tensordot/Prod_1Prod6feedforward_output_dense/Tensordot/GatherV2_1:output:03feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)feedforward_output_dense/Tensordot/concatConcatV20feedforward_output_dense/Tensordot/free:output:00feedforward_output_dense/Tensordot/axes:output:07feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_output_dense/Tensordot/stackPack0feedforward_output_dense/Tensordot/Prod:output:02feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,feedforward_output_dense/Tensordot/transpose	Transpose1feedforward_intermediate_dense/Relu:activations:02feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
*feedforward_output_dense/Tensordot/ReshapeReshape0feedforward_output_dense/Tensordot/transpose:y:01feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)feedforward_output_dense/Tensordot/MatMulMatMul3feedforward_output_dense/Tensordot/Reshape:output:09feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
*feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�r
0feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/concat_1ConcatV24feedforward_output_dense/Tensordot/GatherV2:output:03feedforward_output_dense/Tensordot/Const_2:output:09feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"feedforward_output_dense/TensordotReshape3feedforward_output_dense/Tensordot/MatMul:product:04feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOp8feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 feedforward_output_dense/BiasAddBiasAdd+feedforward_output_dense/Tensordot:output:07feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
feedforward_dropout/IdentityIdentity)feedforward_output_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
add_2AddV2%feedforward_dropout/Identity:output:0-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:�������������������
5feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#feedforward_layer_norm/moments/meanMean	add_2:z:0>feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
+feedforward_layer_norm/moments/StopGradientStopGradient,feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
0feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference	add_2:z:04feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
9feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'feedforward_layer_norm/moments/varianceMean4feedforward_layer_norm/moments/SquaredDifference:z:0Bfeedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(k
&feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$feedforward_layer_norm/batchnorm/addAddV20feedforward_layer_norm/moments/variance:output:0/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
&feedforward_layer_norm/batchnorm/RsqrtRsqrt(feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
3feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp<feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/mulMul*feedforward_layer_norm/batchnorm/Rsqrt:y:0;feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_1Mul	add_2:z:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_2Mul,feedforward_layer_norm/moments/mean:output:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOp8feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/subSub7feedforward_layer_norm/batchnorm/ReadVariableOp:value:0*feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/add_1AddV2*feedforward_layer_norm/batchnorm/mul_1:z:0(feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentity*feedforward_layer_norm/batchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp6^feedforward_intermediate_dense/BiasAdd/ReadVariableOp8^feedforward_intermediate_dense/Tensordot/ReadVariableOp0^feedforward_layer_norm/batchnorm/ReadVariableOp4^feedforward_layer_norm/batchnorm/mul/ReadVariableOp0^feedforward_output_dense/BiasAdd/ReadVariableOp2^feedforward_output_dense/Tensordot/ReadVariableOp3^self_attention/attention_output/add/ReadVariableOp=^self_attention/attention_output/einsum/Einsum/ReadVariableOp&^self_attention/key/add/ReadVariableOp0^self_attention/key/einsum/Einsum/ReadVariableOp(^self_attention/query/add/ReadVariableOp2^self_attention/query/einsum/Einsum/ReadVariableOp(^self_attention/value/add/ReadVariableOp2^self_attention/value/einsum/Einsum/ReadVariableOp3^self_attention_layer_norm/batchnorm/ReadVariableOp7^self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 2n
5feedforward_intermediate_dense/BiasAdd/ReadVariableOp5feedforward_intermediate_dense/BiasAdd/ReadVariableOp2r
7feedforward_intermediate_dense/Tensordot/ReadVariableOp7feedforward_intermediate_dense/Tensordot/ReadVariableOp2b
/feedforward_layer_norm/batchnorm/ReadVariableOp/feedforward_layer_norm/batchnorm/ReadVariableOp2j
3feedforward_layer_norm/batchnorm/mul/ReadVariableOp3feedforward_layer_norm/batchnorm/mul/ReadVariableOp2b
/feedforward_output_dense/BiasAdd/ReadVariableOp/feedforward_output_dense/BiasAdd/ReadVariableOp2f
1feedforward_output_dense/Tensordot/ReadVariableOp1feedforward_output_dense/Tensordot/ReadVariableOp2h
2self_attention/attention_output/add/ReadVariableOp2self_attention/attention_output/add/ReadVariableOp2|
<self_attention/attention_output/einsum/Einsum/ReadVariableOp<self_attention/attention_output/einsum/Einsum/ReadVariableOp2N
%self_attention/key/add/ReadVariableOp%self_attention/key/add/ReadVariableOp2b
/self_attention/key/einsum/Einsum/ReadVariableOp/self_attention/key/einsum/Einsum/ReadVariableOp2R
'self_attention/query/add/ReadVariableOp'self_attention/query/add/ReadVariableOp2f
1self_attention/query/einsum/Einsum/ReadVariableOp1self_attention/query/einsum/Einsum/ReadVariableOp2R
'self_attention/value/add/ReadVariableOp'self_attention/value/add/ReadVariableOp2f
1self_attention/value/einsum/Einsum/ReadVariableOp1self_attention/value/einsum/Einsum/ReadVariableOp2h
2self_attention_layer_norm/batchnorm/ReadVariableOp2self_attention_layer_norm/batchnorm/ReadVariableOp2p
6self_attention_layer_norm/batchnorm/mul/ReadVariableOp6self_attention_layer_norm/batchnorm/mul/ReadVariableOp:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
��
�2
 __inference__wrapped_model_43297
input_1]
Imodel_token_and_position_embedding_token_embedding_embedding_lookup_42887:
�'�g
Smodel_token_and_position_embedding_position_embedding_slice_readvariableop_resource:
��k
Tmodel_transformer_decoder_self_attention_query_einsum_einsum_readvariableop_resource:�U\
Jmodel_transformer_decoder_self_attention_query_add_readvariableop_resource:Ui
Rmodel_transformer_decoder_self_attention_key_einsum_einsum_readvariableop_resource:�UZ
Hmodel_transformer_decoder_self_attention_key_add_readvariableop_resource:Uk
Tmodel_transformer_decoder_self_attention_value_einsum_einsum_readvariableop_resource:�U\
Jmodel_transformer_decoder_self_attention_value_add_readvariableop_resource:Uv
_model_transformer_decoder_self_attention_attention_output_einsum_einsum_readvariableop_resource:U�d
Umodel_transformer_decoder_self_attention_attention_output_add_readvariableop_resource:	�h
Ymodel_transformer_decoder_self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�d
Umodel_transformer_decoder_self_attention_layer_norm_batchnorm_readvariableop_resource:	�n
Zmodel_transformer_decoder_feedforward_intermediate_dense_tensordot_readvariableop_resource:
��g
Xmodel_transformer_decoder_feedforward_intermediate_dense_biasadd_readvariableop_resource:	�h
Tmodel_transformer_decoder_feedforward_output_dense_tensordot_readvariableop_resource:
��a
Rmodel_transformer_decoder_feedforward_output_dense_biasadd_readvariableop_resource:	�e
Vmodel_transformer_decoder_feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�a
Rmodel_transformer_decoder_feedforward_layer_norm_batchnorm_readvariableop_resource:	�m
Vmodel_transformer_decoder_1_self_attention_query_einsum_einsum_readvariableop_resource:�U^
Lmodel_transformer_decoder_1_self_attention_query_add_readvariableop_resource:Uk
Tmodel_transformer_decoder_1_self_attention_key_einsum_einsum_readvariableop_resource:�U\
Jmodel_transformer_decoder_1_self_attention_key_add_readvariableop_resource:Um
Vmodel_transformer_decoder_1_self_attention_value_einsum_einsum_readvariableop_resource:�U^
Lmodel_transformer_decoder_1_self_attention_value_add_readvariableop_resource:Ux
amodel_transformer_decoder_1_self_attention_attention_output_einsum_einsum_readvariableop_resource:U�f
Wmodel_transformer_decoder_1_self_attention_attention_output_add_readvariableop_resource:	�j
[model_transformer_decoder_1_self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�f
Wmodel_transformer_decoder_1_self_attention_layer_norm_batchnorm_readvariableop_resource:	�p
\model_transformer_decoder_1_feedforward_intermediate_dense_tensordot_readvariableop_resource:
��i
Zmodel_transformer_decoder_1_feedforward_intermediate_dense_biasadd_readvariableop_resource:	�j
Vmodel_transformer_decoder_1_feedforward_output_dense_tensordot_readvariableop_resource:
��c
Tmodel_transformer_decoder_1_feedforward_output_dense_biasadd_readvariableop_resource:	�g
Xmodel_transformer_decoder_1_feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�c
Tmodel_transformer_decoder_1_feedforward_layer_norm_batchnorm_readvariableop_resource:	�A
-model_dense_tensordot_readvariableop_resource:
��':
+model_dense_biasadd_readvariableop_resource:	�'
identity��"model/dense/BiasAdd/ReadVariableOp�$model/dense/Tensordot/ReadVariableOp�Jmodel/token_and_position_embedding/position_embedding/Slice/ReadVariableOp�Cmodel/token_and_position_embedding/token_embedding/embedding_lookup�Omodel/transformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOp�Qmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOp�Imodel/transformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOp�Mmodel/transformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOp�Imodel/transformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOp�Kmodel/transformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOp�Lmodel/transformer_decoder/self_attention/attention_output/add/ReadVariableOp�Vmodel/transformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp�?model/transformer_decoder/self_attention/key/add/ReadVariableOp�Imodel/transformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp�Amodel/transformer_decoder/self_attention/query/add/ReadVariableOp�Kmodel/transformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp�Amodel/transformer_decoder/self_attention/value/add/ReadVariableOp�Kmodel/transformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOp�Lmodel/transformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOp�Pmodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOp�Qmodel/transformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOp�Smodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOp�Kmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOp�Omodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOp�Kmodel/transformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOp�Mmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOp�Nmodel/transformer_decoder_1/self_attention/attention_output/add/ReadVariableOp�Xmodel/transformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp�Amodel/transformer_decoder_1/self_attention/key/add/ReadVariableOp�Kmodel/transformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp�Cmodel/transformer_decoder_1/self_attention/query/add/ReadVariableOp�Mmodel/transformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp�Cmodel/transformer_decoder_1/self_attention/value/add/ReadVariableOp�Mmodel/transformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOp�Nmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOp�Rmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp�
Cmodel/token_and_position_embedding/token_embedding/embedding_lookupResourceGatherImodel_token_and_position_embedding_token_embedding_embedding_lookup_42887input_1*
Tindices0*\
_classR
PNloc:@model/token_and_position_embedding/token_embedding/embedding_lookup/42887*5
_output_shapes#
!:�������������������*
dtype0�
Lmodel/token_and_position_embedding/token_embedding/embedding_lookup/IdentityIdentityLmodel/token_and_position_embedding/token_embedding/embedding_lookup:output:0*
T0*\
_classR
PNloc:@model/token_and_position_embedding/token_embedding/embedding_lookup/42887*5
_output_shapes#
!:��������������������
Nmodel/token_and_position_embedding/token_embedding/embedding_lookup/Identity_1IdentityUmodel/token_and_position_embedding/token_embedding/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:�������������������
=model/token_and_position_embedding/token_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
;model/token_and_position_embedding/token_embedding/NotEqualNotEqualinput_1Fmodel/token_and_position_embedding/token_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:�������������������
;model/token_and_position_embedding/position_embedding/ShapeShapeWmodel/token_and_position_embedding/token_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:�
Imodel/token_and_position_embedding/position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Kmodel/token_and_position_embedding/position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Kmodel/token_and_position_embedding/position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Cmodel/token_and_position_embedding/position_embedding/strided_sliceStridedSliceDmodel/token_and_position_embedding/position_embedding/Shape:output:0Rmodel/token_and_position_embedding/position_embedding/strided_slice/stack:output:0Tmodel/token_and_position_embedding/position_embedding/strided_slice/stack_1:output:0Tmodel/token_and_position_embedding/position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Kmodel/token_and_position_embedding/position_embedding/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Mmodel/token_and_position_embedding/position_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Mmodel/token_and_position_embedding/position_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Emodel/token_and_position_embedding/position_embedding/strided_slice_1StridedSliceDmodel/token_and_position_embedding/position_embedding/Shape:output:0Tmodel/token_and_position_embedding/position_embedding/strided_slice_1/stack:output:0Vmodel/token_and_position_embedding/position_embedding/strided_slice_1/stack_1:output:0Vmodel/token_and_position_embedding/position_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Jmodel/token_and_position_embedding/position_embedding/Slice/ReadVariableOpReadVariableOpSmodel_token_and_position_embedding_position_embedding_slice_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Amodel/token_and_position_embedding/position_embedding/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        �
Bmodel/token_and_position_embedding/position_embedding/Slice/size/1Const*
_output_shapes
: *
dtype0*
value
B :��
@model/token_and_position_embedding/position_embedding/Slice/sizePackNmodel/token_and_position_embedding/position_embedding/strided_slice_1:output:0Kmodel/token_and_position_embedding/position_embedding/Slice/size/1:output:0*
N*
T0*
_output_shapes
:�
;model/token_and_position_embedding/position_embedding/SliceSliceRmodel/token_and_position_embedding/position_embedding/Slice/ReadVariableOp:value:0Jmodel/token_and_position_embedding/position_embedding/Slice/begin:output:0Imodel/token_and_position_embedding/position_embedding/Slice/size:output:0*
Index0*
T0*(
_output_shapes
:�����������
>model/token_and_position_embedding/position_embedding/packed/2Const*
_output_shapes
: *
dtype0*
value
B :��
<model/token_and_position_embedding/position_embedding/packedPackLmodel/token_and_position_embedding/position_embedding/strided_slice:output:0Nmodel/token_and_position_embedding/position_embedding/strided_slice_1:output:0Gmodel/token_and_position_embedding/position_embedding/packed/2:output:0*
N*
T0*
_output_shapes
:|
:model/token_and_position_embedding/position_embedding/RankConst*
_output_shapes
: *
dtype0*
value	B :�
Amodel/token_and_position_embedding/position_embedding/BroadcastToBroadcastToDmodel/token_and_position_embedding/position_embedding/Slice:output:0Emodel/token_and_position_embedding/position_embedding/packed:output:0*
T0*5
_output_shapes#
!:��������������������
&model/token_and_position_embedding/addAddV2Wmodel/token_and_position_embedding/token_embedding/embedding_lookup/Identity_1:output:0Jmodel/token_and_position_embedding/position_embedding/BroadcastTo:output:0*
T0*5
_output_shapes#
!:�������������������o
-model/token_and_position_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
+model/token_and_position_embedding/NotEqualNotEqualinput_16model/token_and_position_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:������������������j
(model/transformer_decoder/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
$model/transformer_decoder/ExpandDims
ExpandDims/model/token_and_position_embedding/NotEqual:z:01model/transformer_decoder/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
model/transformer_decoder/CastCast-model/transformer_decoder/ExpandDims:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :������������������y
model/transformer_decoder/ShapeShape*model/token_and_position_embedding/add:z:0*
T0*
_output_shapes
:w
-model/transformer_decoder/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/transformer_decoder/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/transformer_decoder/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'model/transformer_decoder/strided_sliceStridedSlice(model/transformer_decoder/Shape:output:06model/transformer_decoder/strided_slice/stack:output:08model/transformer_decoder/strided_slice/stack_1:output:08model/transformer_decoder/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
/model/transformer_decoder/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/transformer_decoder/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/transformer_decoder/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)model/transformer_decoder/strided_slice_1StridedSlice(model/transformer_decoder/Shape:output:08model/transformer_decoder/strided_slice_1/stack:output:0:model/transformer_decoder/strided_slice_1/stack_1:output:0:model/transformer_decoder/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
!model/transformer_decoder/Shape_1Shape*model/token_and_position_embedding/add:z:0*
T0*
_output_shapes
:y
/model/transformer_decoder/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/transformer_decoder/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/transformer_decoder/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)model/transformer_decoder/strided_slice_2StridedSlice*model/transformer_decoder/Shape_1:output:08model/transformer_decoder/strided_slice_2/stack:output:0:model/transformer_decoder/strided_slice_2/stack_1:output:0:model/transformer_decoder/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
/model/transformer_decoder/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/transformer_decoder/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/transformer_decoder/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)model/transformer_decoder/strided_slice_3StridedSlice*model/transformer_decoder/Shape_1:output:08model/transformer_decoder/strided_slice_3/stack:output:0:model/transformer_decoder/strided_slice_3/stack_1:output:0:model/transformer_decoder/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
%model/transformer_decoder/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%model/transformer_decoder/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$model/transformer_decoder/range/CastCast2model/transformer_decoder/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
model/transformer_decoder/rangeRange.model/transformer_decoder/range/start:output:0(model/transformer_decoder/range/Cast:y:0.model/transformer_decoder/range/delta:output:0*

Tidx0*#
_output_shapes
:���������d
"model/transformer_decoder/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : �
 model/transformer_decoder/Cast_1Cast+model/transformer_decoder/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
model/transformer_decoder/addAddV2(model/transformer_decoder/range:output:0$model/transformer_decoder/Cast_1:y:0*
T0*#
_output_shapes
:���������l
*model/transformer_decoder/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
&model/transformer_decoder/ExpandDims_1
ExpandDims!model/transformer_decoder/add:z:03model/transformer_decoder/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������l
'model/transformer_decoder/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    l
'model/transformer_decoder/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&model/transformer_decoder/range_1/CastCast2model/transformer_decoder/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!model/transformer_decoder/range_1Range0model/transformer_decoder/range_1/start:output:0*model/transformer_decoder/range_1/Cast:y:00model/transformer_decoder/range_1/delta:output:0*

Tidx0*#
_output_shapes
:����������
&model/transformer_decoder/GreaterEqualGreaterEqual/model/transformer_decoder/ExpandDims_1:output:0*model/transformer_decoder/range_1:output:0*
T0*0
_output_shapes
:������������������l
*model/transformer_decoder/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : �
&model/transformer_decoder/ExpandDims_2
ExpandDims*model/transformer_decoder/GreaterEqual:z:03model/transformer_decoder/ExpandDims_2/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
 model/transformer_decoder/packedPack0model/transformer_decoder/strided_slice:output:02model/transformer_decoder/strided_slice_3:output:02model/transformer_decoder/strided_slice_3:output:0*
N*
T0*
_output_shapes
:`
model/transformer_decoder/RankConst*
_output_shapes
: *
dtype0*
value	B :�
%model/transformer_decoder/BroadcastToBroadcastTo/model/transformer_decoder/ExpandDims_2:output:0)model/transformer_decoder/packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
 model/transformer_decoder/Cast_2Cast.model/transformer_decoder/BroadcastTo:output:0*

DstT0*

SrcT0
*=
_output_shapes+
):'����������������������������
!model/transformer_decoder/MinimumMinimum"model/transformer_decoder/Cast:y:0$model/transformer_decoder/Cast_2:y:0*
T0*=
_output_shapes+
):'����������������������������
Kmodel/transformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpTmodel_transformer_decoder_self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
<model/transformer_decoder/self_attention/query/einsum/EinsumEinsum*model/token_and_position_embedding/add:z:0Smodel/transformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
Amodel/transformer_decoder/self_attention/query/add/ReadVariableOpReadVariableOpJmodel_transformer_decoder_self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
2model/transformer_decoder/self_attention/query/addAddV2Emodel/transformer_decoder/self_attention/query/einsum/Einsum:output:0Imodel/transformer_decoder/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Imodel/transformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpRmodel_transformer_decoder_self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
:model/transformer_decoder/self_attention/key/einsum/EinsumEinsum*model/token_and_position_embedding/add:z:0Qmodel/transformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
?model/transformer_decoder/self_attention/key/add/ReadVariableOpReadVariableOpHmodel_transformer_decoder_self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
0model/transformer_decoder/self_attention/key/addAddV2Cmodel/transformer_decoder/self_attention/key/einsum/Einsum:output:0Gmodel/transformer_decoder/self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Kmodel/transformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpTmodel_transformer_decoder_self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
<model/transformer_decoder/self_attention/value/einsum/EinsumEinsum*model/token_and_position_embedding/add:z:0Smodel/transformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
Amodel/transformer_decoder/self_attention/value/add/ReadVariableOpReadVariableOpJmodel_transformer_decoder_self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
2model/transformer_decoder/self_attention/value/addAddV2Emodel/transformer_decoder/self_attention/value/einsum/Einsum:output:0Imodel/transformer_decoder/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������Uq
/model/transformer_decoder/self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :U�
-model/transformer_decoder/self_attention/CastCast8model/transformer_decoder/self_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
-model/transformer_decoder/self_attention/SqrtSqrt1model/transformer_decoder/self_attention/Cast:y:0*
T0*
_output_shapes
: w
2model/transformer_decoder/self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
0model/transformer_decoder/self_attention/truedivRealDiv;model/transformer_decoder/self_attention/truediv/x:output:01model/transformer_decoder/self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
,model/transformer_decoder/self_attention/MulMul6model/transformer_decoder/self_attention/query/add:z:04model/transformer_decoder/self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
6model/transformer_decoder/self_attention/einsum/EinsumEinsum4model/transformer_decoder/self_attention/key/add:z:00model/transformer_decoder/self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbe�
7model/transformer_decoder/self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
3model/transformer_decoder/self_attention/ExpandDims
ExpandDims%model/transformer_decoder/Minimum:z:0@model/transformer_decoder/self_attention/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+����������������������������
5model/transformer_decoder/self_attention/softmax/CastCast<model/transformer_decoder/self_attention/ExpandDims:output:0*

DstT0*

SrcT0*A
_output_shapes/
-:+���������������������������{
6model/transformer_decoder/self_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
4model/transformer_decoder/self_attention/softmax/subSub?model/transformer_decoder/self_attention/softmax/sub/x:output:09model/transformer_decoder/self_attention/softmax/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������{
6model/transformer_decoder/self_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
4model/transformer_decoder/self_attention/softmax/mulMul8model/transformer_decoder/self_attention/softmax/sub:z:0?model/transformer_decoder/self_attention/softmax/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
4model/transformer_decoder/self_attention/softmax/addAddV2?model/transformer_decoder/self_attention/einsum/Einsum:output:08model/transformer_decoder/self_attention/softmax/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
8model/transformer_decoder/self_attention/softmax/SoftmaxSoftmax8model/transformer_decoder/self_attention/softmax/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
9model/transformer_decoder/self_attention/dropout/IdentityIdentityBmodel/transformer_decoder/self_attention/softmax/Softmax:softmax:0*
T0*A
_output_shapes/
-:+����������������������������
8model/transformer_decoder/self_attention/einsum_1/EinsumEinsumBmodel/transformer_decoder/self_attention/dropout/Identity:output:06model/transformer_decoder/self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
Vmodel/transformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp_model_transformer_decoder_self_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
Gmodel/transformer_decoder/self_attention/attention_output/einsum/EinsumEinsumAmodel/transformer_decoder/self_attention/einsum_1/Einsum:output:0^model/transformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
Lmodel/transformer_decoder/self_attention/attention_output/add/ReadVariableOpReadVariableOpUmodel_transformer_decoder_self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=model/transformer_decoder/self_attention/attention_output/addAddV2Pmodel/transformer_decoder/self_attention/attention_output/einsum/Einsum:output:0Tmodel/transformer_decoder/self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
9model/transformer_decoder/self_attention_dropout/IdentityIdentityAmodel/transformer_decoder/self_attention/attention_output/add:z:0*
T0*5
_output_shapes#
!:��������������������
model/transformer_decoder/add_1AddV2Bmodel/transformer_decoder/self_attention_dropout/Identity:output:0*model/token_and_position_embedding/add:z:0*
T0*5
_output_shapes#
!:��������������������
Rmodel/transformer_decoder/self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
@model/transformer_decoder/self_attention_layer_norm/moments/meanMean#model/transformer_decoder/add_1:z:0[model/transformer_decoder/self_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Hmodel/transformer_decoder/self_attention_layer_norm/moments/StopGradientStopGradientImodel/transformer_decoder/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Mmodel/transformer_decoder/self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference#model/transformer_decoder/add_1:z:0Qmodel/transformer_decoder/self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Vmodel/transformer_decoder/self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Dmodel/transformer_decoder/self_attention_layer_norm/moments/varianceMeanQmodel/transformer_decoder/self_attention_layer_norm/moments/SquaredDifference:z:0_model/transformer_decoder/self_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Cmodel/transformer_decoder/self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
Amodel/transformer_decoder/self_attention_layer_norm/batchnorm/addAddV2Mmodel/transformer_decoder/self_attention_layer_norm/moments/variance:output:0Lmodel/transformer_decoder/self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
Cmodel/transformer_decoder/self_attention_layer_norm/batchnorm/RsqrtRsqrtEmodel/transformer_decoder/self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Pmodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpYmodel_transformer_decoder_self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Amodel/transformer_decoder/self_attention_layer_norm/batchnorm/mulMulGmodel/transformer_decoder/self_attention_layer_norm/batchnorm/Rsqrt:y:0Xmodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Cmodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul_1Mul#model/transformer_decoder/add_1:z:0Emodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Cmodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul_2MulImodel/transformer_decoder/self_attention_layer_norm/moments/mean:output:0Emodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Lmodel/transformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOpUmodel_transformer_decoder_self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Amodel/transformer_decoder/self_attention_layer_norm/batchnorm/subSubTmodel/transformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOp:value:0Gmodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
Cmodel/transformer_decoder/self_attention_layer_norm/batchnorm/add_1AddV2Gmodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul_1:z:0Emodel/transformer_decoder/self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
Qmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOpZmodel_transformer_decoder_feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Gmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Gmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Hmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/ShapeShapeGmodel/transformer_decoder/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Pmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2GatherV2Qmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Shape:output:0Pmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/free:output:0Ymodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Rmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV2Qmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Shape:output:0Pmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/axes:output:0[model/transformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Hmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Gmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/ProdProdTmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Qmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Jmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Imodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Prod_1ProdVmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1:output:0Smodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Nmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Imodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/concatConcatV2Pmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/free:output:0Pmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/axes:output:0Wmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Hmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/stackPackPmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Prod:output:0Rmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Lmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/transpose	TransposeGmodel/transformer_decoder/self_attention_layer_norm/batchnorm/add_1:z:0Rmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Jmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/ReshapeReshapePmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/transpose:y:0Qmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Imodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/MatMulMatMulSmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Reshape:output:0Ymodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Jmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Pmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1ConcatV2Tmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Smodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/Const_2:output:0Ymodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Bmodel/transformer_decoder/feedforward_intermediate_dense/TensordotReshapeSmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/MatMul:product:0Tmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Omodel/transformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOpXmodel_transformer_decoder_feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
@model/transformer_decoder/feedforward_intermediate_dense/BiasAddBiasAddKmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot:output:0Wmodel/transformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
=model/transformer_decoder/feedforward_intermediate_dense/ReluReluImodel/transformer_decoder/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
Kmodel/transformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOpTmodel_transformer_decoder_feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Amodel/transformer_decoder/feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Amodel/transformer_decoder/feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Bmodel/transformer_decoder/feedforward_output_dense/Tensordot/ShapeShapeKmodel/transformer_decoder/feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:�
Jmodel/transformer_decoder/feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Emodel/transformer_decoder/feedforward_output_dense/Tensordot/GatherV2GatherV2Kmodel/transformer_decoder/feedforward_output_dense/Tensordot/Shape:output:0Jmodel/transformer_decoder/feedforward_output_dense/Tensordot/free:output:0Smodel/transformer_decoder/feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Lmodel/transformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Gmodel/transformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1GatherV2Kmodel/transformer_decoder/feedforward_output_dense/Tensordot/Shape:output:0Jmodel/transformer_decoder/feedforward_output_dense/Tensordot/axes:output:0Umodel/transformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Bmodel/transformer_decoder/feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Amodel/transformer_decoder/feedforward_output_dense/Tensordot/ProdProdNmodel/transformer_decoder/feedforward_output_dense/Tensordot/GatherV2:output:0Kmodel/transformer_decoder/feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Dmodel/transformer_decoder/feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Cmodel/transformer_decoder/feedforward_output_dense/Tensordot/Prod_1ProdPmodel/transformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1:output:0Mmodel/transformer_decoder/feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Hmodel/transformer_decoder/feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Cmodel/transformer_decoder/feedforward_output_dense/Tensordot/concatConcatV2Jmodel/transformer_decoder/feedforward_output_dense/Tensordot/free:output:0Jmodel/transformer_decoder/feedforward_output_dense/Tensordot/axes:output:0Qmodel/transformer_decoder/feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Bmodel/transformer_decoder/feedforward_output_dense/Tensordot/stackPackJmodel/transformer_decoder/feedforward_output_dense/Tensordot/Prod:output:0Lmodel/transformer_decoder/feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Fmodel/transformer_decoder/feedforward_output_dense/Tensordot/transpose	TransposeKmodel/transformer_decoder/feedforward_intermediate_dense/Relu:activations:0Lmodel/transformer_decoder/feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Dmodel/transformer_decoder/feedforward_output_dense/Tensordot/ReshapeReshapeJmodel/transformer_decoder/feedforward_output_dense/Tensordot/transpose:y:0Kmodel/transformer_decoder/feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Cmodel/transformer_decoder/feedforward_output_dense/Tensordot/MatMulMatMulMmodel/transformer_decoder/feedforward_output_dense/Tensordot/Reshape:output:0Smodel/transformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Dmodel/transformer_decoder/feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Jmodel/transformer_decoder/feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Emodel/transformer_decoder/feedforward_output_dense/Tensordot/concat_1ConcatV2Nmodel/transformer_decoder/feedforward_output_dense/Tensordot/GatherV2:output:0Mmodel/transformer_decoder/feedforward_output_dense/Tensordot/Const_2:output:0Smodel/transformer_decoder/feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
<model/transformer_decoder/feedforward_output_dense/TensordotReshapeMmodel/transformer_decoder/feedforward_output_dense/Tensordot/MatMul:product:0Nmodel/transformer_decoder/feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Imodel/transformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOpRmodel_transformer_decoder_feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:model/transformer_decoder/feedforward_output_dense/BiasAddBiasAddEmodel/transformer_decoder/feedforward_output_dense/Tensordot:output:0Qmodel/transformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
6model/transformer_decoder/feedforward_dropout/IdentityIdentityCmodel/transformer_decoder/feedforward_output_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
model/transformer_decoder/add_2AddV2?model/transformer_decoder/feedforward_dropout/Identity:output:0Gmodel/transformer_decoder/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Omodel/transformer_decoder/feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
=model/transformer_decoder/feedforward_layer_norm/moments/meanMean#model/transformer_decoder/add_2:z:0Xmodel/transformer_decoder/feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Emodel/transformer_decoder/feedforward_layer_norm/moments/StopGradientStopGradientFmodel/transformer_decoder/feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Jmodel/transformer_decoder/feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference#model/transformer_decoder/add_2:z:0Nmodel/transformer_decoder/feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Smodel/transformer_decoder/feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Amodel/transformer_decoder/feedforward_layer_norm/moments/varianceMeanNmodel/transformer_decoder/feedforward_layer_norm/moments/SquaredDifference:z:0\model/transformer_decoder/feedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
@model/transformer_decoder/feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
>model/transformer_decoder/feedforward_layer_norm/batchnorm/addAddV2Jmodel/transformer_decoder/feedforward_layer_norm/moments/variance:output:0Imodel/transformer_decoder/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
@model/transformer_decoder/feedforward_layer_norm/batchnorm/RsqrtRsqrtBmodel/transformer_decoder/feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Mmodel/transformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpVmodel_transformer_decoder_feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
>model/transformer_decoder/feedforward_layer_norm/batchnorm/mulMulDmodel/transformer_decoder/feedforward_layer_norm/batchnorm/Rsqrt:y:0Umodel/transformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
@model/transformer_decoder/feedforward_layer_norm/batchnorm/mul_1Mul#model/transformer_decoder/add_2:z:0Bmodel/transformer_decoder/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
@model/transformer_decoder/feedforward_layer_norm/batchnorm/mul_2MulFmodel/transformer_decoder/feedforward_layer_norm/moments/mean:output:0Bmodel/transformer_decoder/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Imodel/transformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOpRmodel_transformer_decoder_feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
>model/transformer_decoder/feedforward_layer_norm/batchnorm/subSubQmodel/transformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOp:value:0Dmodel/transformer_decoder/feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
@model/transformer_decoder/feedforward_layer_norm/batchnorm/add_1AddV2Dmodel/transformer_decoder/feedforward_layer_norm/batchnorm/mul_1:z:0Bmodel/transformer_decoder/feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������l
*model/transformer_decoder_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
&model/transformer_decoder_1/ExpandDims
ExpandDims/model/token_and_position_embedding/NotEqual:z:03model/transformer_decoder_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
 model/transformer_decoder_1/CastCast/model/transformer_decoder_1/ExpandDims:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :�������������������
!model/transformer_decoder_1/ShapeShapeDmodel/transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:y
/model/transformer_decoder_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/transformer_decoder_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/transformer_decoder_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)model/transformer_decoder_1/strided_sliceStridedSlice*model/transformer_decoder_1/Shape:output:08model/transformer_decoder_1/strided_slice/stack:output:0:model/transformer_decoder_1/strided_slice/stack_1:output:0:model/transformer_decoder_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/transformer_decoder_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/transformer_decoder_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/transformer_decoder_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model/transformer_decoder_1/strided_slice_1StridedSlice*model/transformer_decoder_1/Shape:output:0:model/transformer_decoder_1/strided_slice_1/stack:output:0<model/transformer_decoder_1/strided_slice_1/stack_1:output:0<model/transformer_decoder_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
#model/transformer_decoder_1/Shape_1ShapeDmodel/transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:{
1model/transformer_decoder_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model/transformer_decoder_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/transformer_decoder_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model/transformer_decoder_1/strided_slice_2StridedSlice,model/transformer_decoder_1/Shape_1:output:0:model/transformer_decoder_1/strided_slice_2/stack:output:0<model/transformer_decoder_1/strided_slice_2/stack_1:output:0<model/transformer_decoder_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
1model/transformer_decoder_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/transformer_decoder_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/transformer_decoder_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+model/transformer_decoder_1/strided_slice_3StridedSlice,model/transformer_decoder_1/Shape_1:output:0:model/transformer_decoder_1/strided_slice_3/stack:output:0<model/transformer_decoder_1/strided_slice_3/stack_1:output:0<model/transformer_decoder_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
'model/transformer_decoder_1/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    l
'model/transformer_decoder_1/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
&model/transformer_decoder_1/range/CastCast4model/transformer_decoder_1/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
!model/transformer_decoder_1/rangeRange0model/transformer_decoder_1/range/start:output:0*model/transformer_decoder_1/range/Cast:y:00model/transformer_decoder_1/range/delta:output:0*

Tidx0*#
_output_shapes
:���������f
$model/transformer_decoder_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : �
"model/transformer_decoder_1/Cast_1Cast-model/transformer_decoder_1/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
model/transformer_decoder_1/addAddV2*model/transformer_decoder_1/range:output:0&model/transformer_decoder_1/Cast_1:y:0*
T0*#
_output_shapes
:���������n
,model/transformer_decoder_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
(model/transformer_decoder_1/ExpandDims_1
ExpandDims#model/transformer_decoder_1/add:z:05model/transformer_decoder_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������n
)model/transformer_decoder_1/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)model/transformer_decoder_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
(model/transformer_decoder_1/range_1/CastCast4model/transformer_decoder_1/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
#model/transformer_decoder_1/range_1Range2model/transformer_decoder_1/range_1/start:output:0,model/transformer_decoder_1/range_1/Cast:y:02model/transformer_decoder_1/range_1/delta:output:0*

Tidx0*#
_output_shapes
:����������
(model/transformer_decoder_1/GreaterEqualGreaterEqual1model/transformer_decoder_1/ExpandDims_1:output:0,model/transformer_decoder_1/range_1:output:0*
T0*0
_output_shapes
:������������������n
,model/transformer_decoder_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : �
(model/transformer_decoder_1/ExpandDims_2
ExpandDims,model/transformer_decoder_1/GreaterEqual:z:05model/transformer_decoder_1/ExpandDims_2/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
"model/transformer_decoder_1/packedPack2model/transformer_decoder_1/strided_slice:output:04model/transformer_decoder_1/strided_slice_3:output:04model/transformer_decoder_1/strided_slice_3:output:0*
N*
T0*
_output_shapes
:b
 model/transformer_decoder_1/RankConst*
_output_shapes
: *
dtype0*
value	B :�
'model/transformer_decoder_1/BroadcastToBroadcastTo1model/transformer_decoder_1/ExpandDims_2:output:0+model/transformer_decoder_1/packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
"model/transformer_decoder_1/Cast_2Cast0model/transformer_decoder_1/BroadcastTo:output:0*

DstT0*

SrcT0
*=
_output_shapes+
):'����������������������������
#model/transformer_decoder_1/MinimumMinimum$model/transformer_decoder_1/Cast:y:0&model/transformer_decoder_1/Cast_2:y:0*
T0*=
_output_shapes+
):'����������������������������
Mmodel/transformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpVmodel_transformer_decoder_1_self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
>model/transformer_decoder_1/self_attention/query/einsum/EinsumEinsumDmodel/transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0Umodel/transformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
Cmodel/transformer_decoder_1/self_attention/query/add/ReadVariableOpReadVariableOpLmodel_transformer_decoder_1_self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
4model/transformer_decoder_1/self_attention/query/addAddV2Gmodel/transformer_decoder_1/self_attention/query/einsum/Einsum:output:0Kmodel/transformer_decoder_1/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Kmodel/transformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpTmodel_transformer_decoder_1_self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
<model/transformer_decoder_1/self_attention/key/einsum/EinsumEinsumDmodel/transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0Smodel/transformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
Amodel/transformer_decoder_1/self_attention/key/add/ReadVariableOpReadVariableOpJmodel_transformer_decoder_1_self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
2model/transformer_decoder_1/self_attention/key/addAddV2Emodel/transformer_decoder_1/self_attention/key/einsum/Einsum:output:0Imodel/transformer_decoder_1/self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Mmodel/transformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpVmodel_transformer_decoder_1_self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
>model/transformer_decoder_1/self_attention/value/einsum/EinsumEinsumDmodel/transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0Umodel/transformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
Cmodel/transformer_decoder_1/self_attention/value/add/ReadVariableOpReadVariableOpLmodel_transformer_decoder_1_self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
4model/transformer_decoder_1/self_attention/value/addAddV2Gmodel/transformer_decoder_1/self_attention/value/einsum/Einsum:output:0Kmodel/transformer_decoder_1/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������Us
1model/transformer_decoder_1/self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :U�
/model/transformer_decoder_1/self_attention/CastCast:model/transformer_decoder_1/self_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
/model/transformer_decoder_1/self_attention/SqrtSqrt3model/transformer_decoder_1/self_attention/Cast:y:0*
T0*
_output_shapes
: y
4model/transformer_decoder_1/self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
2model/transformer_decoder_1/self_attention/truedivRealDiv=model/transformer_decoder_1/self_attention/truediv/x:output:03model/transformer_decoder_1/self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
.model/transformer_decoder_1/self_attention/MulMul8model/transformer_decoder_1/self_attention/query/add:z:06model/transformer_decoder_1/self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
8model/transformer_decoder_1/self_attention/einsum/EinsumEinsum6model/transformer_decoder_1/self_attention/key/add:z:02model/transformer_decoder_1/self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbe�
9model/transformer_decoder_1/self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
5model/transformer_decoder_1/self_attention/ExpandDims
ExpandDims'model/transformer_decoder_1/Minimum:z:0Bmodel/transformer_decoder_1/self_attention/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+����������������������������
9model/transformer_decoder_1/self_attention/softmax_1/CastCast>model/transformer_decoder_1/self_attention/ExpandDims:output:0*

DstT0*

SrcT0*A
_output_shapes/
-:+���������������������������
:model/transformer_decoder_1/self_attention/softmax_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
8model/transformer_decoder_1/self_attention/softmax_1/subSubCmodel/transformer_decoder_1/self_attention/softmax_1/sub/x:output:0=model/transformer_decoder_1/self_attention/softmax_1/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������
:model/transformer_decoder_1/self_attention/softmax_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
8model/transformer_decoder_1/self_attention/softmax_1/mulMul<model/transformer_decoder_1/self_attention/softmax_1/sub:z:0Cmodel/transformer_decoder_1/self_attention/softmax_1/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
8model/transformer_decoder_1/self_attention/softmax_1/addAddV2Amodel/transformer_decoder_1/self_attention/einsum/Einsum:output:0<model/transformer_decoder_1/self_attention/softmax_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
<model/transformer_decoder_1/self_attention/softmax_1/SoftmaxSoftmax<model/transformer_decoder_1/self_attention/softmax_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
=model/transformer_decoder_1/self_attention/dropout_1/IdentityIdentityFmodel/transformer_decoder_1/self_attention/softmax_1/Softmax:softmax:0*
T0*A
_output_shapes/
-:+����������������������������
:model/transformer_decoder_1/self_attention/einsum_1/EinsumEinsumFmodel/transformer_decoder_1/self_attention/dropout_1/Identity:output:08model/transformer_decoder_1/self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
Xmodel/transformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpamodel_transformer_decoder_1_self_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
Imodel/transformer_decoder_1/self_attention/attention_output/einsum/EinsumEinsumCmodel/transformer_decoder_1/self_attention/einsum_1/Einsum:output:0`model/transformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
Nmodel/transformer_decoder_1/self_attention/attention_output/add/ReadVariableOpReadVariableOpWmodel_transformer_decoder_1_self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?model/transformer_decoder_1/self_attention/attention_output/addAddV2Rmodel/transformer_decoder_1/self_attention/attention_output/einsum/Einsum:output:0Vmodel/transformer_decoder_1/self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
;model/transformer_decoder_1/self_attention_dropout/IdentityIdentityCmodel/transformer_decoder_1/self_attention/attention_output/add:z:0*
T0*5
_output_shapes#
!:��������������������
!model/transformer_decoder_1/add_1AddV2Dmodel/transformer_decoder_1/self_attention_dropout/Identity:output:0Dmodel/transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Tmodel/transformer_decoder_1/self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Bmodel/transformer_decoder_1/self_attention_layer_norm/moments/meanMean%model/transformer_decoder_1/add_1:z:0]model/transformer_decoder_1/self_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Jmodel/transformer_decoder_1/self_attention_layer_norm/moments/StopGradientStopGradientKmodel/transformer_decoder_1/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Omodel/transformer_decoder_1/self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference%model/transformer_decoder_1/add_1:z:0Smodel/transformer_decoder_1/self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Xmodel/transformer_decoder_1/self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Fmodel/transformer_decoder_1/self_attention_layer_norm/moments/varianceMeanSmodel/transformer_decoder_1/self_attention_layer_norm/moments/SquaredDifference:z:0amodel/transformer_decoder_1/self_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Emodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
Cmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/addAddV2Omodel/transformer_decoder_1/self_attention_layer_norm/moments/variance:output:0Nmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
Emodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/RsqrtRsqrtGmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Rmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp[model_transformer_decoder_1_self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Cmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mulMulImodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/Rsqrt:y:0Zmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Emodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul_1Mul%model/transformer_decoder_1/add_1:z:0Gmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Emodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul_2MulKmodel/transformer_decoder_1/self_attention_layer_norm/moments/mean:output:0Gmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Nmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOpWmodel_transformer_decoder_1_self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Cmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/subSubVmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOp:value:0Imodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
Emodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/add_1AddV2Imodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul_1:z:0Gmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
Smodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOp\model_transformer_decoder_1_feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Imodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Imodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Jmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ShapeShapeImodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Rmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2GatherV2Smodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Shape:output:0Rmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/free:output:0[model/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Tmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Omodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV2Smodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Shape:output:0Rmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/axes:output:0]model/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Jmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Imodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ProdProdVmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Smodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Lmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Kmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Prod_1ProdXmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1:output:0Umodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Pmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Kmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/concatConcatV2Rmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/free:output:0Rmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/axes:output:0Ymodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Jmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/stackPackRmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Prod:output:0Tmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Nmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/transpose	TransposeImodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/add_1:z:0Tmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Lmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReshapeReshapeRmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/transpose:y:0Smodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Kmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/MatMulMatMulUmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Reshape:output:0[model/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Lmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Rmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Mmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1ConcatV2Vmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Umodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_2:output:0[model/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dmodel/transformer_decoder_1/feedforward_intermediate_dense/TensordotReshapeUmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/MatMul:product:0Vmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Qmodel/transformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOpZmodel_transformer_decoder_1_feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Bmodel/transformer_decoder_1/feedforward_intermediate_dense/BiasAddBiasAddMmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot:output:0Ymodel/transformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
?model/transformer_decoder_1/feedforward_intermediate_dense/ReluReluKmodel/transformer_decoder_1/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
Mmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOpVmodel_transformer_decoder_1_feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Cmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Cmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Dmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/ShapeShapeMmodel/transformer_decoder_1/feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:�
Lmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Gmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2GatherV2Mmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Shape:output:0Lmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/free:output:0Umodel/transformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Nmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Imodel/transformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1GatherV2Mmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Shape:output:0Lmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/axes:output:0Wmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Dmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Cmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/ProdProdPmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2:output:0Mmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Fmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Emodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Prod_1ProdRmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1:output:0Omodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Jmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Emodel/transformer_decoder_1/feedforward_output_dense/Tensordot/concatConcatV2Lmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/free:output:0Lmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/axes:output:0Smodel/transformer_decoder_1/feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/stackPackLmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Prod:output:0Nmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Hmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/transpose	TransposeMmodel/transformer_decoder_1/feedforward_intermediate_dense/Relu:activations:0Nmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Fmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/ReshapeReshapeLmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/transpose:y:0Mmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Emodel/transformer_decoder_1/feedforward_output_dense/Tensordot/MatMulMatMulOmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Reshape:output:0Umodel/transformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Fmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Lmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Gmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/concat_1ConcatV2Pmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2:output:0Omodel/transformer_decoder_1/feedforward_output_dense/Tensordot/Const_2:output:0Umodel/transformer_decoder_1/feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
>model/transformer_decoder_1/feedforward_output_dense/TensordotReshapeOmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/MatMul:product:0Pmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Kmodel/transformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOpTmodel_transformer_decoder_1_feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<model/transformer_decoder_1/feedforward_output_dense/BiasAddBiasAddGmodel/transformer_decoder_1/feedforward_output_dense/Tensordot:output:0Smodel/transformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
8model/transformer_decoder_1/feedforward_dropout/IdentityIdentityEmodel/transformer_decoder_1/feedforward_output_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
!model/transformer_decoder_1/add_2AddV2Amodel/transformer_decoder_1/feedforward_dropout/Identity:output:0Imodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Qmodel/transformer_decoder_1/feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
?model/transformer_decoder_1/feedforward_layer_norm/moments/meanMean%model/transformer_decoder_1/add_2:z:0Zmodel/transformer_decoder_1/feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Gmodel/transformer_decoder_1/feedforward_layer_norm/moments/StopGradientStopGradientHmodel/transformer_decoder_1/feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Lmodel/transformer_decoder_1/feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference%model/transformer_decoder_1/add_2:z:0Pmodel/transformer_decoder_1/feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Umodel/transformer_decoder_1/feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Cmodel/transformer_decoder_1/feedforward_layer_norm/moments/varianceMeanPmodel/transformer_decoder_1/feedforward_layer_norm/moments/SquaredDifference:z:0^model/transformer_decoder_1/feedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Bmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
@model/transformer_decoder_1/feedforward_layer_norm/batchnorm/addAddV2Lmodel/transformer_decoder_1/feedforward_layer_norm/moments/variance:output:0Kmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
Bmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/RsqrtRsqrtDmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Omodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_transformer_decoder_1_feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
@model/transformer_decoder_1/feedforward_layer_norm/batchnorm/mulMulFmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/Rsqrt:y:0Wmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Bmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_1Mul%model/transformer_decoder_1/add_2:z:0Dmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Bmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_2MulHmodel/transformer_decoder_1/feedforward_layer_norm/moments/mean:output:0Dmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Kmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOpTmodel_transformer_decoder_1_feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
@model/transformer_decoder_1/feedforward_layer_norm/batchnorm/subSubSmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOp:value:0Fmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
Bmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/add_1AddV2Fmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_1:z:0Dmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��'*
dtype0d
model/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
model/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
model/dense/Tensordot/ShapeShapeFmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:e
#model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense/Tensordot/GatherV2GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/free:output:0,model/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%model/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense/Tensordot/GatherV2_1GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/axes:output:0.model/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
model/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: g
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
model/dense/Tensordot/transpose	TransposeFmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/add_1:z:0%model/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������'h
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�'e
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������'�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�'*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������'y
IdentityIdentitymodel/dense/BiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'�
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOpK^model/token_and_position_embedding/position_embedding/Slice/ReadVariableOpD^model/token_and_position_embedding/token_embedding/embedding_lookupP^model/transformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOpR^model/transformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOpJ^model/transformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOpN^model/transformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOpJ^model/transformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOpL^model/transformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOpM^model/transformer_decoder/self_attention/attention_output/add/ReadVariableOpW^model/transformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp@^model/transformer_decoder/self_attention/key/add/ReadVariableOpJ^model/transformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOpB^model/transformer_decoder/self_attention/query/add/ReadVariableOpL^model/transformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOpB^model/transformer_decoder/self_attention/value/add/ReadVariableOpL^model/transformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOpM^model/transformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOpQ^model/transformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOpR^model/transformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOpT^model/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOpL^model/transformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOpP^model/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOpL^model/transformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOpN^model/transformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOpO^model/transformer_decoder_1/self_attention/attention_output/add/ReadVariableOpY^model/transformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOpB^model/transformer_decoder_1/self_attention/key/add/ReadVariableOpL^model/transformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOpD^model/transformer_decoder_1/self_attention/query/add/ReadVariableOpN^model/transformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOpD^model/transformer_decoder_1/self_attention/value/add/ReadVariableOpN^model/transformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOpO^model/transformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOpS^model/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2�
Jmodel/token_and_position_embedding/position_embedding/Slice/ReadVariableOpJmodel/token_and_position_embedding/position_embedding/Slice/ReadVariableOp2�
Cmodel/token_and_position_embedding/token_embedding/embedding_lookupCmodel/token_and_position_embedding/token_embedding/embedding_lookup2�
Omodel/transformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOpOmodel/transformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOp2�
Qmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOpQmodel/transformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOp2�
Imodel/transformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOpImodel/transformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOp2�
Mmodel/transformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOpMmodel/transformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOp2�
Imodel/transformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOpImodel/transformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOp2�
Kmodel/transformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOpKmodel/transformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOp2�
Lmodel/transformer_decoder/self_attention/attention_output/add/ReadVariableOpLmodel/transformer_decoder/self_attention/attention_output/add/ReadVariableOp2�
Vmodel/transformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOpVmodel/transformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp2�
?model/transformer_decoder/self_attention/key/add/ReadVariableOp?model/transformer_decoder/self_attention/key/add/ReadVariableOp2�
Imodel/transformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOpImodel/transformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp2�
Amodel/transformer_decoder/self_attention/query/add/ReadVariableOpAmodel/transformer_decoder/self_attention/query/add/ReadVariableOp2�
Kmodel/transformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOpKmodel/transformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp2�
Amodel/transformer_decoder/self_attention/value/add/ReadVariableOpAmodel/transformer_decoder/self_attention/value/add/ReadVariableOp2�
Kmodel/transformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOpKmodel/transformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOp2�
Lmodel/transformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOpLmodel/transformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOp2�
Pmodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOpPmodel/transformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOp2�
Qmodel/transformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOpQmodel/transformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOp2�
Smodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOpSmodel/transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOp2�
Kmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOpKmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOp2�
Omodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOpOmodel/transformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOp2�
Kmodel/transformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOpKmodel/transformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOp2�
Mmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOpMmodel/transformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOp2�
Nmodel/transformer_decoder_1/self_attention/attention_output/add/ReadVariableOpNmodel/transformer_decoder_1/self_attention/attention_output/add/ReadVariableOp2�
Xmodel/transformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOpXmodel/transformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp2�
Amodel/transformer_decoder_1/self_attention/key/add/ReadVariableOpAmodel/transformer_decoder_1/self_attention/key/add/ReadVariableOp2�
Kmodel/transformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOpKmodel/transformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp2�
Cmodel/transformer_decoder_1/self_attention/query/add/ReadVariableOpCmodel/transformer_decoder_1/self_attention/query/add/ReadVariableOp2�
Mmodel/transformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOpMmodel/transformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp2�
Cmodel/transformer_decoder_1/self_attention/value/add/ReadVariableOpCmodel/transformer_decoder_1/self_attention/value/add/ReadVariableOp2�
Mmodel/transformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOpMmodel/transformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOp2�
Nmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOpNmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOp2�
Rmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOpRmodel/transformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp:Y U
0
_output_shapes
:������������������
!
_user_specified_name	input_1
�.
�
@__inference_model_layer_call_and_return_conditional_losses_44552

inputs6
"token_and_position_embedding_44473:
�'�6
"token_and_position_embedding_44475:
��0
transformer_decoder_44480:�U+
transformer_decoder_44482:U0
transformer_decoder_44484:�U+
transformer_decoder_44486:U0
transformer_decoder_44488:�U+
transformer_decoder_44490:U0
transformer_decoder_44492:U�(
transformer_decoder_44494:	�(
transformer_decoder_44496:	�(
transformer_decoder_44498:	�-
transformer_decoder_44500:
��(
transformer_decoder_44502:	�-
transformer_decoder_44504:
��(
transformer_decoder_44506:	�(
transformer_decoder_44508:	�(
transformer_decoder_44510:	�2
transformer_decoder_1_44513:�U-
transformer_decoder_1_44515:U2
transformer_decoder_1_44517:�U-
transformer_decoder_1_44519:U2
transformer_decoder_1_44521:�U-
transformer_decoder_1_44523:U2
transformer_decoder_1_44525:U�*
transformer_decoder_1_44527:	�*
transformer_decoder_1_44529:	�*
transformer_decoder_1_44531:	�/
transformer_decoder_1_44533:
��*
transformer_decoder_1_44535:	�/
transformer_decoder_1_44537:
��*
transformer_decoder_1_44539:	�*
transformer_decoder_1_44541:	�*
transformer_decoder_1_44543:	�
dense_44546:
��'
dense_44548:	�'
identity��dense/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�+transformer_decoder/StatefulPartitionedCall�-transformer_decoder_1/StatefulPartitionedCall�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs"token_and_position_embedding_44473"token_and_position_embedding_44475*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_43335i
'token_and_position_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
%token_and_position_embedding/NotEqualNotEqualinputs0token_and_position_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:�������������������
+transformer_decoder/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_decoder_44480transformer_decoder_44482transformer_decoder_44484transformer_decoder_44486transformer_decoder_44488transformer_decoder_44490transformer_decoder_44492transformer_decoder_44494transformer_decoder_44496transformer_decoder_44498transformer_decoder_44500transformer_decoder_44502transformer_decoder_44504transformer_decoder_44506transformer_decoder_44508transformer_decoder_44510*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_44346�
-transformer_decoder_1/StatefulPartitionedCallStatefulPartitionedCall4transformer_decoder/StatefulPartitionedCall:output:0transformer_decoder_1_44513transformer_decoder_1_44515transformer_decoder_1_44517transformer_decoder_1_44519transformer_decoder_1_44521transformer_decoder_1_44523transformer_decoder_1_44525transformer_decoder_1_44527transformer_decoder_1_44529transformer_decoder_1_44531transformer_decoder_1_44533transformer_decoder_1_44535transformer_decoder_1_44537transformer_decoder_1_44539transformer_decoder_1_44541transformer_decoder_1_44543*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_44098�
dense/StatefulPartitionedCallStatefulPartitionedCall6transformer_decoder_1/StatefulPartitionedCall:output:0dense_44546dense_44548*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43793�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'�
NoOpNoOp^dense/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_decoder/StatefulPartitionedCall.^transformer_decoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_decoder/StatefulPartitionedCall+transformer_decoder/StatefulPartitionedCall2^
-transformer_decoder_1/StatefulPartitionedCall-transformer_decoder_1/StatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
5__inference_transformer_decoder_1_layer_call_fn_46464
decoder_sequence
unknown:�U
	unknown_0:U 
	unknown_1:�U
	unknown_2:U 
	unknown_3:�U
	unknown_4:U 
	unknown_5:U�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldecoder_sequenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_44098}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
�
�
%__inference_dense_layer_call_fn_46822

inputs
unknown:
��'
	unknown_0:	�'
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43793}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:�������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
��
�
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_46390
decoder_sequenceQ
:self_attention_query_einsum_einsum_readvariableop_resource:�UB
0self_attention_query_add_readvariableop_resource:UO
8self_attention_key_einsum_einsum_readvariableop_resource:�U@
.self_attention_key_add_readvariableop_resource:UQ
:self_attention_value_einsum_einsum_readvariableop_resource:�UB
0self_attention_value_add_readvariableop_resource:U\
Eself_attention_attention_output_einsum_einsum_readvariableop_resource:U�J
;self_attention_attention_output_add_readvariableop_resource:	�N
?self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�J
;self_attention_layer_norm_batchnorm_readvariableop_resource:	�T
@feedforward_intermediate_dense_tensordot_readvariableop_resource:
��M
>feedforward_intermediate_dense_biasadd_readvariableop_resource:	�N
:feedforward_output_dense_tensordot_readvariableop_resource:
��G
8feedforward_output_dense_biasadd_readvariableop_resource:	�K
<feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�G
8feedforward_layer_norm_batchnorm_readvariableop_resource:	�
identity��5feedforward_intermediate_dense/BiasAdd/ReadVariableOp�7feedforward_intermediate_dense/Tensordot/ReadVariableOp�/feedforward_layer_norm/batchnorm/ReadVariableOp�3feedforward_layer_norm/batchnorm/mul/ReadVariableOp�/feedforward_output_dense/BiasAdd/ReadVariableOp�1feedforward_output_dense/Tensordot/ReadVariableOp�2self_attention/attention_output/add/ReadVariableOp�<self_attention/attention_output/einsum/Einsum/ReadVariableOp�%self_attention/key/add/ReadVariableOp�/self_attention/key/einsum/Einsum/ReadVariableOp�'self_attention/query/add/ReadVariableOp�1self_attention/query/einsum/Einsum/ReadVariableOp�'self_attention/value/add/ReadVariableOp�1self_attention/value/einsum/Einsum/ReadVariableOp�2self_attention_layer_norm/batchnorm/ReadVariableOp�6self_attention_layer_norm/batchnorm/mul/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
Shape_1Shapedecoder_sequence*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\

range/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: {
rangeRangerange/start:output:0range/Cast:y:0range/delta:output:0*

Tidx0*#
_output_shapes
:���������H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B : M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
addAddV2range:output:0Cast:y:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :l

ExpandDims
ExpandDimsadd:z:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������R
range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    R
range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
range_1/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
range_1Rangerange_1/start:output:0range_1/Cast:y:0range_1/delta:output:0*

Tidx0*#
_output_shapes
:���������~
GreaterEqualGreaterEqualExpandDims:output:0range_1:output:0*
T0*0
_output_shapes
:������������������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsGreaterEqual:z:0ExpandDims_1/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
packedPackstrided_slice:output:0strided_slice_3:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :�
BroadcastToBroadcastToExpandDims_1:output:0packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
1self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/query/einsum/EinsumEinsumdecoder_sequence9self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/query/add/ReadVariableOpReadVariableOp0self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/query/addAddV2+self_attention/query/einsum/Einsum:output:0/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp8self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
 self_attention/key/einsum/EinsumEinsumdecoder_sequence7self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
%self_attention/key/add/ReadVariableOpReadVariableOp.self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/key/addAddV2)self_attention/key/einsum/Einsum:output:0-self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
1self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/value/einsum/EinsumEinsumdecoder_sequence9self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/value/add/ReadVariableOpReadVariableOp0self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/value/addAddV2+self_attention/value/einsum/Einsum:output:0/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������UW
self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Uk
self_attention/CastCastself_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: U
self_attention/SqrtSqrtself_attention/Cast:y:0*
T0*
_output_shapes
: ]
self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
self_attention/truedivRealDiv!self_attention/truediv/x:output:0self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
self_attention/MulMulself_attention/query/add:z:0self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
self_attention/einsum/EinsumEinsumself_attention/key/add:z:0self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbeh
self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
self_attention/ExpandDims
ExpandDimsBroadcastTo:output:0&self_attention/ExpandDims/dim:output:0*
T0
*A
_output_shapes/
-:+����������������������������
self_attention/softmax/CastCast"self_attention/ExpandDims:output:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+���������������������������a
self_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
self_attention/softmax/subSub%self_attention/softmax/sub/x:output:0self_attention/softmax/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������a
self_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
self_attention/softmax/mulMulself_attention/softmax/sub:z:0%self_attention/softmax/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax/addAddV2%self_attention/einsum/Einsum:output:0self_attention/softmax/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax/SoftmaxSoftmaxself_attention/softmax/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/einsum_1/EinsumEinsum(self_attention/softmax/Softmax:softmax:0self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
<self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpEself_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
-self_attention/attention_output/einsum/EinsumEinsum'self_attention/einsum_1/Einsum:output:0Dself_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
2self_attention/attention_output/add/ReadVariableOpReadVariableOp;self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#self_attention/attention_output/addAddV26self_attention/attention_output/einsum/Einsum:output:0:self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
add_1AddV2'self_attention/attention_output/add:z:0decoder_sequence*
T0*5
_output_shapes#
!:��������������������
8self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&self_attention_layer_norm/moments/meanMean	add_1:z:0Aself_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
.self_attention_layer_norm/moments/StopGradientStopGradient/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
3self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference	add_1:z:07self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
<self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
*self_attention_layer_norm/moments/varianceMean7self_attention_layer_norm/moments/SquaredDifference:z:0Eself_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(n
)self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
'self_attention_layer_norm/batchnorm/addAddV23self_attention_layer_norm/moments/variance:output:02self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
)self_attention_layer_norm/batchnorm/RsqrtRsqrt+self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
6self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp?self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/mulMul-self_attention_layer_norm/batchnorm/Rsqrt:y:0>self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_1Mul	add_1:z:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_2Mul/self_attention_layer_norm/moments/mean:output:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
2self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOp;self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/subSub:self_attention_layer_norm/batchnorm/ReadVariableOp:value:0-self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/add_1AddV2-self_attention_layer_norm/batchnorm/mul_1:z:0+self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
7feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOp@feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
-feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
.feedforward_intermediate_dense/Tensordot/ShapeShape-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:x
6feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/GatherV2GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/free:output:0?feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/axes:output:0Afeedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
-feedforward_intermediate_dense/Tensordot/ProdProd:feedforward_intermediate_dense/Tensordot/GatherV2:output:07feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
/feedforward_intermediate_dense/Tensordot/Prod_1Prod<feedforward_intermediate_dense/Tensordot/GatherV2_1:output:09feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/feedforward_intermediate_dense/Tensordot/concatConcatV26feedforward_intermediate_dense/Tensordot/free:output:06feedforward_intermediate_dense/Tensordot/axes:output:0=feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
.feedforward_intermediate_dense/Tensordot/stackPack6feedforward_intermediate_dense/Tensordot/Prod:output:08feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
2feedforward_intermediate_dense/Tensordot/transpose	Transpose-self_attention_layer_norm/batchnorm/add_1:z:08feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
0feedforward_intermediate_dense/Tensordot/ReshapeReshape6feedforward_intermediate_dense/Tensordot/transpose:y:07feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
/feedforward_intermediate_dense/Tensordot/MatMulMatMul9feedforward_intermediate_dense/Tensordot/Reshape:output:0?feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
0feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�x
6feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/concat_1ConcatV2:feedforward_intermediate_dense/Tensordot/GatherV2:output:09feedforward_intermediate_dense/Tensordot/Const_2:output:0?feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_intermediate_dense/TensordotReshape9feedforward_intermediate_dense/Tensordot/MatMul:product:0:feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
5feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOp>feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&feedforward_intermediate_dense/BiasAddBiasAdd1feedforward_intermediate_dense/Tensordot:output:0=feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
#feedforward_intermediate_dense/ReluRelu/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
1feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOp:feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0q
'feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(feedforward_output_dense/Tensordot/ShapeShape1feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:r
0feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/GatherV2GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/free:output:09feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-feedforward_output_dense/Tensordot/GatherV2_1GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/axes:output:0;feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'feedforward_output_dense/Tensordot/ProdProd4feedforward_output_dense/Tensordot/GatherV2:output:01feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)feedforward_output_dense/Tensordot/Prod_1Prod6feedforward_output_dense/Tensordot/GatherV2_1:output:03feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)feedforward_output_dense/Tensordot/concatConcatV20feedforward_output_dense/Tensordot/free:output:00feedforward_output_dense/Tensordot/axes:output:07feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_output_dense/Tensordot/stackPack0feedforward_output_dense/Tensordot/Prod:output:02feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,feedforward_output_dense/Tensordot/transpose	Transpose1feedforward_intermediate_dense/Relu:activations:02feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
*feedforward_output_dense/Tensordot/ReshapeReshape0feedforward_output_dense/Tensordot/transpose:y:01feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)feedforward_output_dense/Tensordot/MatMulMatMul3feedforward_output_dense/Tensordot/Reshape:output:09feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
*feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�r
0feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/concat_1ConcatV24feedforward_output_dense/Tensordot/GatherV2:output:03feedforward_output_dense/Tensordot/Const_2:output:09feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"feedforward_output_dense/TensordotReshape3feedforward_output_dense/Tensordot/MatMul:product:04feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOp8feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 feedforward_output_dense/BiasAddBiasAdd+feedforward_output_dense/Tensordot:output:07feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
add_2AddV2)feedforward_output_dense/BiasAdd:output:0-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:�������������������
5feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#feedforward_layer_norm/moments/meanMean	add_2:z:0>feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
+feedforward_layer_norm/moments/StopGradientStopGradient,feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
0feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference	add_2:z:04feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
9feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'feedforward_layer_norm/moments/varianceMean4feedforward_layer_norm/moments/SquaredDifference:z:0Bfeedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(k
&feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$feedforward_layer_norm/batchnorm/addAddV20feedforward_layer_norm/moments/variance:output:0/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
&feedforward_layer_norm/batchnorm/RsqrtRsqrt(feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
3feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp<feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/mulMul*feedforward_layer_norm/batchnorm/Rsqrt:y:0;feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_1Mul	add_2:z:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_2Mul,feedforward_layer_norm/moments/mean:output:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOp8feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/subSub7feedforward_layer_norm/batchnorm/ReadVariableOp:value:0*feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/add_1AddV2*feedforward_layer_norm/batchnorm/mul_1:z:0(feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentity*feedforward_layer_norm/batchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp6^feedforward_intermediate_dense/BiasAdd/ReadVariableOp8^feedforward_intermediate_dense/Tensordot/ReadVariableOp0^feedforward_layer_norm/batchnorm/ReadVariableOp4^feedforward_layer_norm/batchnorm/mul/ReadVariableOp0^feedforward_output_dense/BiasAdd/ReadVariableOp2^feedforward_output_dense/Tensordot/ReadVariableOp3^self_attention/attention_output/add/ReadVariableOp=^self_attention/attention_output/einsum/Einsum/ReadVariableOp&^self_attention/key/add/ReadVariableOp0^self_attention/key/einsum/Einsum/ReadVariableOp(^self_attention/query/add/ReadVariableOp2^self_attention/query/einsum/Einsum/ReadVariableOp(^self_attention/value/add/ReadVariableOp2^self_attention/value/einsum/Einsum/ReadVariableOp3^self_attention_layer_norm/batchnorm/ReadVariableOp7^self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 2n
5feedforward_intermediate_dense/BiasAdd/ReadVariableOp5feedforward_intermediate_dense/BiasAdd/ReadVariableOp2r
7feedforward_intermediate_dense/Tensordot/ReadVariableOp7feedforward_intermediate_dense/Tensordot/ReadVariableOp2b
/feedforward_layer_norm/batchnorm/ReadVariableOp/feedforward_layer_norm/batchnorm/ReadVariableOp2j
3feedforward_layer_norm/batchnorm/mul/ReadVariableOp3feedforward_layer_norm/batchnorm/mul/ReadVariableOp2b
/feedforward_output_dense/BiasAdd/ReadVariableOp/feedforward_output_dense/BiasAdd/ReadVariableOp2f
1feedforward_output_dense/Tensordot/ReadVariableOp1feedforward_output_dense/Tensordot/ReadVariableOp2h
2self_attention/attention_output/add/ReadVariableOp2self_attention/attention_output/add/ReadVariableOp2|
<self_attention/attention_output/einsum/Einsum/ReadVariableOp<self_attention/attention_output/einsum/Einsum/ReadVariableOp2N
%self_attention/key/add/ReadVariableOp%self_attention/key/add/ReadVariableOp2b
/self_attention/key/einsum/Einsum/ReadVariableOp/self_attention/key/einsum/Einsum/ReadVariableOp2R
'self_attention/query/add/ReadVariableOp'self_attention/query/add/ReadVariableOp2f
1self_attention/query/einsum/Einsum/ReadVariableOp1self_attention/query/einsum/Einsum/ReadVariableOp2R
'self_attention/value/add/ReadVariableOp'self_attention/value/add/ReadVariableOp2f
1self_attention/value/einsum/Einsum/ReadVariableOp1self_attention/value/einsum/Einsum/ReadVariableOp2h
2self_attention_layer_norm/batchnorm/ReadVariableOp2self_attention_layer_norm/batchnorm/ReadVariableOp2p
6self_attention_layer_norm/batchnorm/mul/ReadVariableOp6self_attention_layer_norm/batchnorm/mul/ReadVariableOp:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
�
�	
%__inference_model_layer_call_fn_44704
input_1
unknown:
�'�
	unknown_0:
�� 
	unknown_1:�U
	unknown_2:U 
	unknown_3:�U
	unknown_4:U 
	unknown_5:�U
	unknown_6:U 
	unknown_7:U�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�!

unknown_17:�U

unknown_18:U!

unknown_19:�U

unknown_20:U!

unknown_21:�U

unknown_22:U!

unknown_23:U�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:
��

unknown_28:	�

unknown_29:
��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:
��'

unknown_34:	�'
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_44552}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:������������������
!
_user_specified_name	input_1
��
�
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_46640
decoder_sequenceQ
:self_attention_query_einsum_einsum_readvariableop_resource:�UB
0self_attention_query_add_readvariableop_resource:UO
8self_attention_key_einsum_einsum_readvariableop_resource:�U@
.self_attention_key_add_readvariableop_resource:UQ
:self_attention_value_einsum_einsum_readvariableop_resource:�UB
0self_attention_value_add_readvariableop_resource:U\
Eself_attention_attention_output_einsum_einsum_readvariableop_resource:U�J
;self_attention_attention_output_add_readvariableop_resource:	�N
?self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�J
;self_attention_layer_norm_batchnorm_readvariableop_resource:	�T
@feedforward_intermediate_dense_tensordot_readvariableop_resource:
��M
>feedforward_intermediate_dense_biasadd_readvariableop_resource:	�N
:feedforward_output_dense_tensordot_readvariableop_resource:
��G
8feedforward_output_dense_biasadd_readvariableop_resource:	�K
<feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�G
8feedforward_layer_norm_batchnorm_readvariableop_resource:	�
identity��5feedforward_intermediate_dense/BiasAdd/ReadVariableOp�7feedforward_intermediate_dense/Tensordot/ReadVariableOp�/feedforward_layer_norm/batchnorm/ReadVariableOp�3feedforward_layer_norm/batchnorm/mul/ReadVariableOp�/feedforward_output_dense/BiasAdd/ReadVariableOp�1feedforward_output_dense/Tensordot/ReadVariableOp�2self_attention/attention_output/add/ReadVariableOp�<self_attention/attention_output/einsum/Einsum/ReadVariableOp�%self_attention/key/add/ReadVariableOp�/self_attention/key/einsum/Einsum/ReadVariableOp�'self_attention/query/add/ReadVariableOp�1self_attention/query/einsum/Einsum/ReadVariableOp�'self_attention/value/add/ReadVariableOp�1self_attention/value/einsum/Einsum/ReadVariableOp�2self_attention_layer_norm/batchnorm/ReadVariableOp�6self_attention_layer_norm/batchnorm/mul/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
Shape_1Shapedecoder_sequence*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\

range/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: {
rangeRangerange/start:output:0range/Cast:y:0range/delta:output:0*

Tidx0*#
_output_shapes
:���������H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B : M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
addAddV2range:output:0Cast:y:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :l

ExpandDims
ExpandDimsadd:z:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������R
range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    R
range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
range_1/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
range_1Rangerange_1/start:output:0range_1/Cast:y:0range_1/delta:output:0*

Tidx0*#
_output_shapes
:���������~
GreaterEqualGreaterEqualExpandDims:output:0range_1:output:0*
T0*0
_output_shapes
:������������������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsGreaterEqual:z:0ExpandDims_1/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
packedPackstrided_slice:output:0strided_slice_3:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :�
BroadcastToBroadcastToExpandDims_1:output:0packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
1self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/query/einsum/EinsumEinsumdecoder_sequence9self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/query/add/ReadVariableOpReadVariableOp0self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/query/addAddV2+self_attention/query/einsum/Einsum:output:0/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp8self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
 self_attention/key/einsum/EinsumEinsumdecoder_sequence7self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
%self_attention/key/add/ReadVariableOpReadVariableOp.self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/key/addAddV2)self_attention/key/einsum/Einsum:output:0-self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
1self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/value/einsum/EinsumEinsumdecoder_sequence9self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/value/add/ReadVariableOpReadVariableOp0self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/value/addAddV2+self_attention/value/einsum/Einsum:output:0/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������UW
self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Uk
self_attention/CastCastself_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: U
self_attention/SqrtSqrtself_attention/Cast:y:0*
T0*
_output_shapes
: ]
self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
self_attention/truedivRealDiv!self_attention/truediv/x:output:0self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
self_attention/MulMulself_attention/query/add:z:0self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
self_attention/einsum/EinsumEinsumself_attention/key/add:z:0self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbeh
self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
self_attention/ExpandDims
ExpandDimsBroadcastTo:output:0&self_attention/ExpandDims/dim:output:0*
T0
*A
_output_shapes/
-:+����������������������������
self_attention/softmax_1/CastCast"self_attention/ExpandDims:output:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+���������������������������c
self_attention/softmax_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
self_attention/softmax_1/subSub'self_attention/softmax_1/sub/x:output:0!self_attention/softmax_1/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������c
self_attention/softmax_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
self_attention/softmax_1/mulMul self_attention/softmax_1/sub:z:0'self_attention/softmax_1/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax_1/addAddV2%self_attention/einsum/Einsum:output:0 self_attention/softmax_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
 self_attention/softmax_1/SoftmaxSoftmax self_attention/softmax_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
!self_attention/dropout_1/IdentityIdentity*self_attention/softmax_1/Softmax:softmax:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/einsum_1/EinsumEinsum*self_attention/dropout_1/Identity:output:0self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
<self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpEself_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
-self_attention/attention_output/einsum/EinsumEinsum'self_attention/einsum_1/Einsum:output:0Dself_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
2self_attention/attention_output/add/ReadVariableOpReadVariableOp;self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#self_attention/attention_output/addAddV26self_attention/attention_output/einsum/Einsum:output:0:self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
self_attention_dropout/IdentityIdentity'self_attention/attention_output/add:z:0*
T0*5
_output_shapes#
!:��������������������
add_1AddV2(self_attention_dropout/Identity:output:0decoder_sequence*
T0*5
_output_shapes#
!:��������������������
8self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&self_attention_layer_norm/moments/meanMean	add_1:z:0Aself_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
.self_attention_layer_norm/moments/StopGradientStopGradient/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
3self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference	add_1:z:07self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
<self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
*self_attention_layer_norm/moments/varianceMean7self_attention_layer_norm/moments/SquaredDifference:z:0Eself_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(n
)self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
'self_attention_layer_norm/batchnorm/addAddV23self_attention_layer_norm/moments/variance:output:02self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
)self_attention_layer_norm/batchnorm/RsqrtRsqrt+self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
6self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp?self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/mulMul-self_attention_layer_norm/batchnorm/Rsqrt:y:0>self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_1Mul	add_1:z:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_2Mul/self_attention_layer_norm/moments/mean:output:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
2self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOp;self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/subSub:self_attention_layer_norm/batchnorm/ReadVariableOp:value:0-self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/add_1AddV2-self_attention_layer_norm/batchnorm/mul_1:z:0+self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
7feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOp@feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
-feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
.feedforward_intermediate_dense/Tensordot/ShapeShape-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:x
6feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/GatherV2GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/free:output:0?feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/axes:output:0Afeedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
-feedforward_intermediate_dense/Tensordot/ProdProd:feedforward_intermediate_dense/Tensordot/GatherV2:output:07feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
/feedforward_intermediate_dense/Tensordot/Prod_1Prod<feedforward_intermediate_dense/Tensordot/GatherV2_1:output:09feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/feedforward_intermediate_dense/Tensordot/concatConcatV26feedforward_intermediate_dense/Tensordot/free:output:06feedforward_intermediate_dense/Tensordot/axes:output:0=feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
.feedforward_intermediate_dense/Tensordot/stackPack6feedforward_intermediate_dense/Tensordot/Prod:output:08feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
2feedforward_intermediate_dense/Tensordot/transpose	Transpose-self_attention_layer_norm/batchnorm/add_1:z:08feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
0feedforward_intermediate_dense/Tensordot/ReshapeReshape6feedforward_intermediate_dense/Tensordot/transpose:y:07feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
/feedforward_intermediate_dense/Tensordot/MatMulMatMul9feedforward_intermediate_dense/Tensordot/Reshape:output:0?feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
0feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�x
6feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/concat_1ConcatV2:feedforward_intermediate_dense/Tensordot/GatherV2:output:09feedforward_intermediate_dense/Tensordot/Const_2:output:0?feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_intermediate_dense/TensordotReshape9feedforward_intermediate_dense/Tensordot/MatMul:product:0:feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
5feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOp>feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&feedforward_intermediate_dense/BiasAddBiasAdd1feedforward_intermediate_dense/Tensordot:output:0=feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
#feedforward_intermediate_dense/ReluRelu/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
1feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOp:feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0q
'feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(feedforward_output_dense/Tensordot/ShapeShape1feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:r
0feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/GatherV2GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/free:output:09feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-feedforward_output_dense/Tensordot/GatherV2_1GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/axes:output:0;feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'feedforward_output_dense/Tensordot/ProdProd4feedforward_output_dense/Tensordot/GatherV2:output:01feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)feedforward_output_dense/Tensordot/Prod_1Prod6feedforward_output_dense/Tensordot/GatherV2_1:output:03feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)feedforward_output_dense/Tensordot/concatConcatV20feedforward_output_dense/Tensordot/free:output:00feedforward_output_dense/Tensordot/axes:output:07feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_output_dense/Tensordot/stackPack0feedforward_output_dense/Tensordot/Prod:output:02feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,feedforward_output_dense/Tensordot/transpose	Transpose1feedforward_intermediate_dense/Relu:activations:02feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
*feedforward_output_dense/Tensordot/ReshapeReshape0feedforward_output_dense/Tensordot/transpose:y:01feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)feedforward_output_dense/Tensordot/MatMulMatMul3feedforward_output_dense/Tensordot/Reshape:output:09feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
*feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�r
0feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/concat_1ConcatV24feedforward_output_dense/Tensordot/GatherV2:output:03feedforward_output_dense/Tensordot/Const_2:output:09feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"feedforward_output_dense/TensordotReshape3feedforward_output_dense/Tensordot/MatMul:product:04feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOp8feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 feedforward_output_dense/BiasAddBiasAdd+feedforward_output_dense/Tensordot:output:07feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
feedforward_dropout/IdentityIdentity)feedforward_output_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
add_2AddV2%feedforward_dropout/Identity:output:0-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:�������������������
5feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#feedforward_layer_norm/moments/meanMean	add_2:z:0>feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
+feedforward_layer_norm/moments/StopGradientStopGradient,feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
0feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference	add_2:z:04feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
9feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'feedforward_layer_norm/moments/varianceMean4feedforward_layer_norm/moments/SquaredDifference:z:0Bfeedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(k
&feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$feedforward_layer_norm/batchnorm/addAddV20feedforward_layer_norm/moments/variance:output:0/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
&feedforward_layer_norm/batchnorm/RsqrtRsqrt(feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
3feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp<feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/mulMul*feedforward_layer_norm/batchnorm/Rsqrt:y:0;feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_1Mul	add_2:z:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_2Mul,feedforward_layer_norm/moments/mean:output:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOp8feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/subSub7feedforward_layer_norm/batchnorm/ReadVariableOp:value:0*feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/add_1AddV2*feedforward_layer_norm/batchnorm/mul_1:z:0(feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentity*feedforward_layer_norm/batchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp6^feedforward_intermediate_dense/BiasAdd/ReadVariableOp8^feedforward_intermediate_dense/Tensordot/ReadVariableOp0^feedforward_layer_norm/batchnorm/ReadVariableOp4^feedforward_layer_norm/batchnorm/mul/ReadVariableOp0^feedforward_output_dense/BiasAdd/ReadVariableOp2^feedforward_output_dense/Tensordot/ReadVariableOp3^self_attention/attention_output/add/ReadVariableOp=^self_attention/attention_output/einsum/Einsum/ReadVariableOp&^self_attention/key/add/ReadVariableOp0^self_attention/key/einsum/Einsum/ReadVariableOp(^self_attention/query/add/ReadVariableOp2^self_attention/query/einsum/Einsum/ReadVariableOp(^self_attention/value/add/ReadVariableOp2^self_attention/value/einsum/Einsum/ReadVariableOp3^self_attention_layer_norm/batchnorm/ReadVariableOp7^self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 2n
5feedforward_intermediate_dense/BiasAdd/ReadVariableOp5feedforward_intermediate_dense/BiasAdd/ReadVariableOp2r
7feedforward_intermediate_dense/Tensordot/ReadVariableOp7feedforward_intermediate_dense/Tensordot/ReadVariableOp2b
/feedforward_layer_norm/batchnorm/ReadVariableOp/feedforward_layer_norm/batchnorm/ReadVariableOp2j
3feedforward_layer_norm/batchnorm/mul/ReadVariableOp3feedforward_layer_norm/batchnorm/mul/ReadVariableOp2b
/feedforward_output_dense/BiasAdd/ReadVariableOp/feedforward_output_dense/BiasAdd/ReadVariableOp2f
1feedforward_output_dense/Tensordot/ReadVariableOp1feedforward_output_dense/Tensordot/ReadVariableOp2h
2self_attention/attention_output/add/ReadVariableOp2self_attention/attention_output/add/ReadVariableOp2|
<self_attention/attention_output/einsum/Einsum/ReadVariableOp<self_attention/attention_output/einsum/Einsum/ReadVariableOp2N
%self_attention/key/add/ReadVariableOp%self_attention/key/add/ReadVariableOp2b
/self_attention/key/einsum/Einsum/ReadVariableOp/self_attention/key/einsum/Einsum/ReadVariableOp2R
'self_attention/query/add/ReadVariableOp'self_attention/query/add/ReadVariableOp2f
1self_attention/query/einsum/Einsum/ReadVariableOp1self_attention/query/einsum/Einsum/ReadVariableOp2R
'self_attention/value/add/ReadVariableOp'self_attention/value/add/ReadVariableOp2f
1self_attention/value/einsum/Einsum/ReadVariableOp1self_attention/value/einsum/Einsum/ReadVariableOp2h
2self_attention_layer_norm/batchnorm/ReadVariableOp2self_attention_layer_norm/batchnorm/ReadVariableOp2p
6self_attention_layer_norm/batchnorm/mul/ReadVariableOp6self_attention_layer_norm/batchnorm/mul/ReadVariableOp:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
�
�	
%__inference_model_layer_call_fn_43875
input_1
unknown:
�'�
	unknown_0:
�� 
	unknown_1:�U
	unknown_2:U 
	unknown_3:�U
	unknown_4:U 
	unknown_5:�U
	unknown_6:U 
	unknown_7:U�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:	�

unknown_16:	�!

unknown_17:�U

unknown_18:U!

unknown_19:�U

unknown_20:U!

unknown_21:�U

unknown_22:U!

unknown_23:U�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:
��

unknown_28:	�

unknown_29:
��

unknown_30:	�

unknown_31:	�

unknown_32:	�

unknown_33:
��'

unknown_34:	�'
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_43800}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:������������������
!
_user_specified_name	input_1
�%
�
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_43335

inputs:
&token_embedding_embedding_lookup_43307:
�'�D
0position_embedding_slice_readvariableop_resource:
��
identity��'position_embedding/Slice/ReadVariableOp� token_embedding/embedding_lookup�
 token_embedding/embedding_lookupResourceGather&token_embedding_embedding_lookup_43307inputs*
Tindices0*9
_class/
-+loc:@token_embedding/embedding_lookup/43307*5
_output_shapes#
!:�������������������*
dtype0�
)token_embedding/embedding_lookup/IdentityIdentity)token_embedding/embedding_lookup:output:0*
T0*9
_class/
-+loc:@token_embedding/embedding_lookup/43307*5
_output_shapes#
!:��������������������
+token_embedding/embedding_lookup/Identity_1Identity2token_embedding/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:�������������������\
token_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
token_embedding/NotEqualNotEqualinputs#token_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:������������������|
position_embedding/ShapeShape4token_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:p
&position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 position_embedding/strided_sliceStridedSlice!position_embedding/Shape:output:0/position_embedding/strided_slice/stack:output:01position_embedding/strided_slice/stack_1:output:01position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
(position_embedding/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*position_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*position_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"position_embedding/strided_slice_1StridedSlice!position_embedding/Shape:output:01position_embedding/strided_slice_1/stack:output:03position_embedding/strided_slice_1/stack_1:output:03position_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
'position_embedding/Slice/ReadVariableOpReadVariableOp0position_embedding_slice_readvariableop_resource* 
_output_shapes
:
��*
dtype0o
position_embedding/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        b
position_embedding/Slice/size/1Const*
_output_shapes
: *
dtype0*
value
B :��
position_embedding/Slice/sizePack+position_embedding/strided_slice_1:output:0(position_embedding/Slice/size/1:output:0*
N*
T0*
_output_shapes
:�
position_embedding/SliceSlice/position_embedding/Slice/ReadVariableOp:value:0'position_embedding/Slice/begin:output:0&position_embedding/Slice/size:output:0*
Index0*
T0*(
_output_shapes
:����������^
position_embedding/packed/2Const*
_output_shapes
: *
dtype0*
value
B :��
position_embedding/packedPack)position_embedding/strided_slice:output:0+position_embedding/strided_slice_1:output:0$position_embedding/packed/2:output:0*
N*
T0*
_output_shapes
:Y
position_embedding/RankConst*
_output_shapes
: *
dtype0*
value	B :�
position_embedding/BroadcastToBroadcastTo!position_embedding/Slice:output:0"position_embedding/packed:output:0*
T0*5
_output_shapes#
!:��������������������
addAddV24token_embedding/embedding_lookup/Identity_1:output:0'position_embedding/BroadcastTo:output:0*
T0*5
_output_shapes#
!:�������������������d
IdentityIdentityadd:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp(^position_embedding/Slice/ReadVariableOp!^token_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������: : 2R
'position_embedding/Slice/ReadVariableOp'position_embedding/Slice/ReadVariableOp2D
 token_embedding/embedding_lookup token_embedding/embedding_lookup:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
��
�/
@__inference_model_layer_call_and_return_conditional_losses_45927

inputsW
Ctoken_and_position_embedding_token_embedding_embedding_lookup_45523:
�'�a
Mtoken_and_position_embedding_position_embedding_slice_readvariableop_resource:
��e
Ntransformer_decoder_self_attention_query_einsum_einsum_readvariableop_resource:�UV
Dtransformer_decoder_self_attention_query_add_readvariableop_resource:Uc
Ltransformer_decoder_self_attention_key_einsum_einsum_readvariableop_resource:�UT
Btransformer_decoder_self_attention_key_add_readvariableop_resource:Ue
Ntransformer_decoder_self_attention_value_einsum_einsum_readvariableop_resource:�UV
Dtransformer_decoder_self_attention_value_add_readvariableop_resource:Up
Ytransformer_decoder_self_attention_attention_output_einsum_einsum_readvariableop_resource:U�^
Otransformer_decoder_self_attention_attention_output_add_readvariableop_resource:	�b
Stransformer_decoder_self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�^
Otransformer_decoder_self_attention_layer_norm_batchnorm_readvariableop_resource:	�h
Ttransformer_decoder_feedforward_intermediate_dense_tensordot_readvariableop_resource:
��a
Rtransformer_decoder_feedforward_intermediate_dense_biasadd_readvariableop_resource:	�b
Ntransformer_decoder_feedforward_output_dense_tensordot_readvariableop_resource:
��[
Ltransformer_decoder_feedforward_output_dense_biasadd_readvariableop_resource:	�_
Ptransformer_decoder_feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�[
Ltransformer_decoder_feedforward_layer_norm_batchnorm_readvariableop_resource:	�g
Ptransformer_decoder_1_self_attention_query_einsum_einsum_readvariableop_resource:�UX
Ftransformer_decoder_1_self_attention_query_add_readvariableop_resource:Ue
Ntransformer_decoder_1_self_attention_key_einsum_einsum_readvariableop_resource:�UV
Dtransformer_decoder_1_self_attention_key_add_readvariableop_resource:Ug
Ptransformer_decoder_1_self_attention_value_einsum_einsum_readvariableop_resource:�UX
Ftransformer_decoder_1_self_attention_value_add_readvariableop_resource:Ur
[transformer_decoder_1_self_attention_attention_output_einsum_einsum_readvariableop_resource:U�`
Qtransformer_decoder_1_self_attention_attention_output_add_readvariableop_resource:	�d
Utransformer_decoder_1_self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�`
Qtransformer_decoder_1_self_attention_layer_norm_batchnorm_readvariableop_resource:	�j
Vtransformer_decoder_1_feedforward_intermediate_dense_tensordot_readvariableop_resource:
��c
Ttransformer_decoder_1_feedforward_intermediate_dense_biasadd_readvariableop_resource:	�d
Ptransformer_decoder_1_feedforward_output_dense_tensordot_readvariableop_resource:
��]
Ntransformer_decoder_1_feedforward_output_dense_biasadd_readvariableop_resource:	�a
Rtransformer_decoder_1_feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�]
Ntransformer_decoder_1_feedforward_layer_norm_batchnorm_readvariableop_resource:	�;
'dense_tensordot_readvariableop_resource:
��'4
%dense_biasadd_readvariableop_resource:	�'
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�Dtoken_and_position_embedding/position_embedding/Slice/ReadVariableOp�=token_and_position_embedding/token_embedding/embedding_lookup�Itransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOp�Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOp�Ctransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOp�Gtransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOp�Ctransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOp�Etransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOp�Ftransformer_decoder/self_attention/attention_output/add/ReadVariableOp�Ptransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp�9transformer_decoder/self_attention/key/add/ReadVariableOp�Ctransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp�;transformer_decoder/self_attention/query/add/ReadVariableOp�Etransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp�;transformer_decoder/self_attention/value/add/ReadVariableOp�Etransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOp�Ftransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOp�Jtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOp�Ktransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOp�Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOp�Etransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOp�Itransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOp�Etransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOp�Gtransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOp�Htransformer_decoder_1/self_attention/attention_output/add/ReadVariableOp�Rtransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp�;transformer_decoder_1/self_attention/key/add/ReadVariableOp�Etransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp�=transformer_decoder_1/self_attention/query/add/ReadVariableOp�Gtransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp�=transformer_decoder_1/self_attention/value/add/ReadVariableOp�Gtransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOp�Htransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOp�Ltransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp�
=token_and_position_embedding/token_embedding/embedding_lookupResourceGatherCtoken_and_position_embedding_token_embedding_embedding_lookup_45523inputs*
Tindices0*V
_classL
JHloc:@token_and_position_embedding/token_embedding/embedding_lookup/45523*5
_output_shapes#
!:�������������������*
dtype0�
Ftoken_and_position_embedding/token_embedding/embedding_lookup/IdentityIdentityFtoken_and_position_embedding/token_embedding/embedding_lookup:output:0*
T0*V
_classL
JHloc:@token_and_position_embedding/token_embedding/embedding_lookup/45523*5
_output_shapes#
!:��������������������
Htoken_and_position_embedding/token_embedding/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding/token_embedding/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:�������������������y
7token_and_position_embedding/token_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
5token_and_position_embedding/token_embedding/NotEqualNotEqualinputs@token_and_position_embedding/token_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:�������������������
5token_and_position_embedding/position_embedding/ShapeShapeQtoken_and_position_embedding/token_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:�
Ctoken_and_position_embedding/position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Etoken_and_position_embedding/position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Etoken_and_position_embedding/position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=token_and_position_embedding/position_embedding/strided_sliceStridedSlice>token_and_position_embedding/position_embedding/Shape:output:0Ltoken_and_position_embedding/position_embedding/strided_slice/stack:output:0Ntoken_and_position_embedding/position_embedding/strided_slice/stack_1:output:0Ntoken_and_position_embedding/position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Etoken_and_position_embedding/position_embedding/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
Gtoken_and_position_embedding/position_embedding/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gtoken_and_position_embedding/position_embedding/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?token_and_position_embedding/position_embedding/strided_slice_1StridedSlice>token_and_position_embedding/position_embedding/Shape:output:0Ntoken_and_position_embedding/position_embedding/strided_slice_1/stack:output:0Ptoken_and_position_embedding/position_embedding/strided_slice_1/stack_1:output:0Ptoken_and_position_embedding/position_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Dtoken_and_position_embedding/position_embedding/Slice/ReadVariableOpReadVariableOpMtoken_and_position_embedding_position_embedding_slice_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;token_and_position_embedding/position_embedding/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        
<token_and_position_embedding/position_embedding/Slice/size/1Const*
_output_shapes
: *
dtype0*
value
B :��
:token_and_position_embedding/position_embedding/Slice/sizePackHtoken_and_position_embedding/position_embedding/strided_slice_1:output:0Etoken_and_position_embedding/position_embedding/Slice/size/1:output:0*
N*
T0*
_output_shapes
:�
5token_and_position_embedding/position_embedding/SliceSliceLtoken_and_position_embedding/position_embedding/Slice/ReadVariableOp:value:0Dtoken_and_position_embedding/position_embedding/Slice/begin:output:0Ctoken_and_position_embedding/position_embedding/Slice/size:output:0*
Index0*
T0*(
_output_shapes
:����������{
8token_and_position_embedding/position_embedding/packed/2Const*
_output_shapes
: *
dtype0*
value
B :��
6token_and_position_embedding/position_embedding/packedPackFtoken_and_position_embedding/position_embedding/strided_slice:output:0Htoken_and_position_embedding/position_embedding/strided_slice_1:output:0Atoken_and_position_embedding/position_embedding/packed/2:output:0*
N*
T0*
_output_shapes
:v
4token_and_position_embedding/position_embedding/RankConst*
_output_shapes
: *
dtype0*
value	B :�
;token_and_position_embedding/position_embedding/BroadcastToBroadcastTo>token_and_position_embedding/position_embedding/Slice:output:0?token_and_position_embedding/position_embedding/packed:output:0*
T0*5
_output_shapes#
!:��������������������
 token_and_position_embedding/addAddV2Qtoken_and_position_embedding/token_embedding/embedding_lookup/Identity_1:output:0Dtoken_and_position_embedding/position_embedding/BroadcastTo:output:0*
T0*5
_output_shapes#
!:�������������������i
'token_and_position_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
%token_and_position_embedding/NotEqualNotEqualinputs0token_and_position_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:������������������d
"transformer_decoder/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
transformer_decoder/ExpandDims
ExpandDims)token_and_position_embedding/NotEqual:z:0+transformer_decoder/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
transformer_decoder/CastCast'transformer_decoder/ExpandDims:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :������������������m
transformer_decoder/ShapeShape$token_and_position_embedding/add:z:0*
T0*
_output_shapes
:q
'transformer_decoder/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)transformer_decoder/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)transformer_decoder/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!transformer_decoder/strided_sliceStridedSlice"transformer_decoder/Shape:output:00transformer_decoder/strided_slice/stack:output:02transformer_decoder/strided_slice/stack_1:output:02transformer_decoder/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)transformer_decoder/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#transformer_decoder/strided_slice_1StridedSlice"transformer_decoder/Shape:output:02transformer_decoder/strided_slice_1/stack:output:04transformer_decoder/strided_slice_1/stack_1:output:04transformer_decoder/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
transformer_decoder/Shape_1Shape$token_and_position_embedding/add:z:0*
T0*
_output_shapes
:s
)transformer_decoder/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+transformer_decoder/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#transformer_decoder/strided_slice_2StridedSlice$transformer_decoder/Shape_1:output:02transformer_decoder/strided_slice_2/stack:output:04transformer_decoder/strided_slice_2/stack_1:output:04transformer_decoder/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)transformer_decoder/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#transformer_decoder/strided_slice_3StridedSlice$transformer_decoder/Shape_1:output:02transformer_decoder/strided_slice_3/stack:output:04transformer_decoder/strided_slice_3/stack_1:output:04transformer_decoder/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
transformer_decoder/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    d
transformer_decoder/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
transformer_decoder/range/CastCast,transformer_decoder/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder/rangeRange(transformer_decoder/range/start:output:0"transformer_decoder/range/Cast:y:0(transformer_decoder/range/delta:output:0*

Tidx0*#
_output_shapes
:���������^
transformer_decoder/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : y
transformer_decoder/Cast_1Cast%transformer_decoder/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder/addAddV2"transformer_decoder/range:output:0transformer_decoder/Cast_1:y:0*
T0*#
_output_shapes
:���������f
$transformer_decoder/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
 transformer_decoder/ExpandDims_1
ExpandDimstransformer_decoder/add:z:0-transformer_decoder/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������f
!transformer_decoder/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    f
!transformer_decoder/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 transformer_decoder/range_1/CastCast,transformer_decoder/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder/range_1Range*transformer_decoder/range_1/start:output:0$transformer_decoder/range_1/Cast:y:0*transformer_decoder/range_1/delta:output:0*

Tidx0*#
_output_shapes
:����������
 transformer_decoder/GreaterEqualGreaterEqual)transformer_decoder/ExpandDims_1:output:0$transformer_decoder/range_1:output:0*
T0*0
_output_shapes
:������������������f
$transformer_decoder/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : �
 transformer_decoder/ExpandDims_2
ExpandDims$transformer_decoder/GreaterEqual:z:0-transformer_decoder/ExpandDims_2/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
transformer_decoder/packedPack*transformer_decoder/strided_slice:output:0,transformer_decoder/strided_slice_3:output:0,transformer_decoder/strided_slice_3:output:0*
N*
T0*
_output_shapes
:Z
transformer_decoder/RankConst*
_output_shapes
: *
dtype0*
value	B :�
transformer_decoder/BroadcastToBroadcastTo)transformer_decoder/ExpandDims_2:output:0#transformer_decoder/packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
transformer_decoder/Cast_2Cast(transformer_decoder/BroadcastTo:output:0*

DstT0*

SrcT0
*=
_output_shapes+
):'����������������������������
transformer_decoder/MinimumMinimumtransformer_decoder/Cast:y:0transformer_decoder/Cast_2:y:0*
T0*=
_output_shapes+
):'����������������������������
Etransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_decoder_self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
6transformer_decoder/self_attention/query/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Mtransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
;transformer_decoder/self_attention/query/add/ReadVariableOpReadVariableOpDtransformer_decoder_self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
,transformer_decoder/self_attention/query/addAddV2?transformer_decoder/self_attention/query/einsum/Einsum:output:0Ctransformer_decoder/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Ctransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpLtransformer_decoder_self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
4transformer_decoder/self_attention/key/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Ktransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
9transformer_decoder/self_attention/key/add/ReadVariableOpReadVariableOpBtransformer_decoder_self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
*transformer_decoder/self_attention/key/addAddV2=transformer_decoder/self_attention/key/einsum/Einsum:output:0Atransformer_decoder/self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Etransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_decoder_self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
6transformer_decoder/self_attention/value/einsum/EinsumEinsum$token_and_position_embedding/add:z:0Mtransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
;transformer_decoder/self_attention/value/add/ReadVariableOpReadVariableOpDtransformer_decoder_self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
,transformer_decoder/self_attention/value/addAddV2?transformer_decoder/self_attention/value/einsum/Einsum:output:0Ctransformer_decoder/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������Uk
)transformer_decoder/self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :U�
'transformer_decoder/self_attention/CastCast2transformer_decoder/self_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: }
'transformer_decoder/self_attention/SqrtSqrt+transformer_decoder/self_attention/Cast:y:0*
T0*
_output_shapes
: q
,transformer_decoder/self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
*transformer_decoder/self_attention/truedivRealDiv5transformer_decoder/self_attention/truediv/x:output:0+transformer_decoder/self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
&transformer_decoder/self_attention/MulMul0transformer_decoder/self_attention/query/add:z:0.transformer_decoder/self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
0transformer_decoder/self_attention/einsum/EinsumEinsum.transformer_decoder/self_attention/key/add:z:0*transformer_decoder/self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbe|
1transformer_decoder/self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
-transformer_decoder/self_attention/ExpandDims
ExpandDimstransformer_decoder/Minimum:z:0:transformer_decoder/self_attention/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+����������������������������
/transformer_decoder/self_attention/softmax/CastCast6transformer_decoder/self_attention/ExpandDims:output:0*

DstT0*

SrcT0*A
_output_shapes/
-:+���������������������������u
0transformer_decoder/self_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
.transformer_decoder/self_attention/softmax/subSub9transformer_decoder/self_attention/softmax/sub/x:output:03transformer_decoder/self_attention/softmax/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������u
0transformer_decoder/self_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
.transformer_decoder/self_attention/softmax/mulMul2transformer_decoder/self_attention/softmax/sub:z:09transformer_decoder/self_attention/softmax/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
.transformer_decoder/self_attention/softmax/addAddV29transformer_decoder/self_attention/einsum/Einsum:output:02transformer_decoder/self_attention/softmax/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
2transformer_decoder/self_attention/softmax/SoftmaxSoftmax2transformer_decoder/self_attention/softmax/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
2transformer_decoder/self_attention/einsum_1/EinsumEinsum<transformer_decoder/self_attention/softmax/Softmax:softmax:00transformer_decoder/self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
Ptransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpYtransformer_decoder_self_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
Atransformer_decoder/self_attention/attention_output/einsum/EinsumEinsum;transformer_decoder/self_attention/einsum_1/Einsum:output:0Xtransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
Ftransformer_decoder/self_attention/attention_output/add/ReadVariableOpReadVariableOpOtransformer_decoder_self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7transformer_decoder/self_attention/attention_output/addAddV2Jtransformer_decoder/self_attention/attention_output/einsum/Einsum:output:0Ntransformer_decoder/self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
transformer_decoder/add_1AddV2;transformer_decoder/self_attention/attention_output/add:z:0$token_and_position_embedding/add:z:0*
T0*5
_output_shapes#
!:��������������������
Ltransformer_decoder/self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
:transformer_decoder/self_attention_layer_norm/moments/meanMeantransformer_decoder/add_1:z:0Utransformer_decoder/self_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Btransformer_decoder/self_attention_layer_norm/moments/StopGradientStopGradientCtransformer_decoder/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Gtransformer_decoder/self_attention_layer_norm/moments/SquaredDifferenceSquaredDifferencetransformer_decoder/add_1:z:0Ktransformer_decoder/self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Ptransformer_decoder/self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>transformer_decoder/self_attention_layer_norm/moments/varianceMeanKtransformer_decoder/self_attention_layer_norm/moments/SquaredDifference:z:0Ytransformer_decoder/self_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
=transformer_decoder/self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
;transformer_decoder/self_attention_layer_norm/batchnorm/addAddV2Gtransformer_decoder/self_attention_layer_norm/moments/variance:output:0Ftransformer_decoder/self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
=transformer_decoder/self_attention_layer_norm/batchnorm/RsqrtRsqrt?transformer_decoder/self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Jtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpStransformer_decoder_self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;transformer_decoder/self_attention_layer_norm/batchnorm/mulMulAtransformer_decoder/self_attention_layer_norm/batchnorm/Rsqrt:y:0Rtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
=transformer_decoder/self_attention_layer_norm/batchnorm/mul_1Multransformer_decoder/add_1:z:0?transformer_decoder/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
=transformer_decoder/self_attention_layer_norm/batchnorm/mul_2MulCtransformer_decoder/self_attention_layer_norm/moments/mean:output:0?transformer_decoder/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Ftransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOpOtransformer_decoder_self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
;transformer_decoder/self_attention_layer_norm/batchnorm/subSubNtransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOp:value:0Atransformer_decoder/self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
=transformer_decoder/self_attention_layer_norm/batchnorm/add_1AddV2Atransformer_decoder/self_attention_layer_norm/batchnorm/mul_1:z:0?transformer_decoder/self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOpTtransformer_decoder_feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Atransformer_decoder/feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Atransformer_decoder/feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Btransformer_decoder/feedforward_intermediate_dense/Tensordot/ShapeShapeAtransformer_decoder/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Etransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2GatherV2Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/Shape:output:0Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/free:output:0Stransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ltransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Gtransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV2Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/Shape:output:0Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/axes:output:0Utransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Btransformer_decoder/feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Atransformer_decoder/feedforward_intermediate_dense/Tensordot/ProdProdNtransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Dtransformer_decoder/feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Ctransformer_decoder/feedforward_intermediate_dense/Tensordot/Prod_1ProdPtransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2_1:output:0Mtransformer_decoder/feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Htransformer_decoder/feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Ctransformer_decoder/feedforward_intermediate_dense/Tensordot/concatConcatV2Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/free:output:0Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/axes:output:0Qtransformer_decoder/feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Btransformer_decoder/feedforward_intermediate_dense/Tensordot/stackPackJtransformer_decoder/feedforward_intermediate_dense/Tensordot/Prod:output:0Ltransformer_decoder/feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Ftransformer_decoder/feedforward_intermediate_dense/Tensordot/transpose	TransposeAtransformer_decoder/self_attention_layer_norm/batchnorm/add_1:z:0Ltransformer_decoder/feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Dtransformer_decoder/feedforward_intermediate_dense/Tensordot/ReshapeReshapeJtransformer_decoder/feedforward_intermediate_dense/Tensordot/transpose:y:0Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Ctransformer_decoder/feedforward_intermediate_dense/Tensordot/MatMulMatMulMtransformer_decoder/feedforward_intermediate_dense/Tensordot/Reshape:output:0Stransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Dtransformer_decoder/feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Jtransformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Etransformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1ConcatV2Ntransformer_decoder/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Mtransformer_decoder/feedforward_intermediate_dense/Tensordot/Const_2:output:0Stransformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
<transformer_decoder/feedforward_intermediate_dense/TensordotReshapeMtransformer_decoder/feedforward_intermediate_dense/Tensordot/MatMul:product:0Ntransformer_decoder/feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Itransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOpRtransformer_decoder_feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:transformer_decoder/feedforward_intermediate_dense/BiasAddBiasAddEtransformer_decoder/feedforward_intermediate_dense/Tensordot:output:0Qtransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
7transformer_decoder/feedforward_intermediate_dense/ReluReluCtransformer_decoder/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
Etransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOpNtransformer_decoder_feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
;transformer_decoder/feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
;transformer_decoder/feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
<transformer_decoder/feedforward_output_dense/Tensordot/ShapeShapeEtransformer_decoder/feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:�
Dtransformer_decoder/feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?transformer_decoder/feedforward_output_dense/Tensordot/GatherV2GatherV2Etransformer_decoder/feedforward_output_dense/Tensordot/Shape:output:0Dtransformer_decoder/feedforward_output_dense/Tensordot/free:output:0Mtransformer_decoder/feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ftransformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Atransformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1GatherV2Etransformer_decoder/feedforward_output_dense/Tensordot/Shape:output:0Dtransformer_decoder/feedforward_output_dense/Tensordot/axes:output:0Otransformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
<transformer_decoder/feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
;transformer_decoder/feedforward_output_dense/Tensordot/ProdProdHtransformer_decoder/feedforward_output_dense/Tensordot/GatherV2:output:0Etransformer_decoder/feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
>transformer_decoder/feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
=transformer_decoder/feedforward_output_dense/Tensordot/Prod_1ProdJtransformer_decoder/feedforward_output_dense/Tensordot/GatherV2_1:output:0Gtransformer_decoder/feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Btransformer_decoder/feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
=transformer_decoder/feedforward_output_dense/Tensordot/concatConcatV2Dtransformer_decoder/feedforward_output_dense/Tensordot/free:output:0Dtransformer_decoder/feedforward_output_dense/Tensordot/axes:output:0Ktransformer_decoder/feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
<transformer_decoder/feedforward_output_dense/Tensordot/stackPackDtransformer_decoder/feedforward_output_dense/Tensordot/Prod:output:0Ftransformer_decoder/feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
@transformer_decoder/feedforward_output_dense/Tensordot/transpose	TransposeEtransformer_decoder/feedforward_intermediate_dense/Relu:activations:0Ftransformer_decoder/feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
>transformer_decoder/feedforward_output_dense/Tensordot/ReshapeReshapeDtransformer_decoder/feedforward_output_dense/Tensordot/transpose:y:0Etransformer_decoder/feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
=transformer_decoder/feedforward_output_dense/Tensordot/MatMulMatMulGtransformer_decoder/feedforward_output_dense/Tensordot/Reshape:output:0Mtransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
>transformer_decoder/feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Dtransformer_decoder/feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?transformer_decoder/feedforward_output_dense/Tensordot/concat_1ConcatV2Htransformer_decoder/feedforward_output_dense/Tensordot/GatherV2:output:0Gtransformer_decoder/feedforward_output_dense/Tensordot/Const_2:output:0Mtransformer_decoder/feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
6transformer_decoder/feedforward_output_dense/TensordotReshapeGtransformer_decoder/feedforward_output_dense/Tensordot/MatMul:product:0Htransformer_decoder/feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Ctransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOpLtransformer_decoder_feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4transformer_decoder/feedforward_output_dense/BiasAddBiasAdd?transformer_decoder/feedforward_output_dense/Tensordot:output:0Ktransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
transformer_decoder/add_2AddV2=transformer_decoder/feedforward_output_dense/BiasAdd:output:0Atransformer_decoder/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Itransformer_decoder/feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_decoder/feedforward_layer_norm/moments/meanMeantransformer_decoder/add_2:z:0Rtransformer_decoder/feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
?transformer_decoder/feedforward_layer_norm/moments/StopGradientStopGradient@transformer_decoder/feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Dtransformer_decoder/feedforward_layer_norm/moments/SquaredDifferenceSquaredDifferencetransformer_decoder/add_2:z:0Htransformer_decoder/feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Mtransformer_decoder/feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
;transformer_decoder/feedforward_layer_norm/moments/varianceMeanHtransformer_decoder/feedforward_layer_norm/moments/SquaredDifference:z:0Vtransformer_decoder/feedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(
:transformer_decoder/feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
8transformer_decoder/feedforward_layer_norm/batchnorm/addAddV2Dtransformer_decoder/feedforward_layer_norm/moments/variance:output:0Ctransformer_decoder/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
:transformer_decoder/feedforward_layer_norm/batchnorm/RsqrtRsqrt<transformer_decoder/feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Gtransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_decoder_feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8transformer_decoder/feedforward_layer_norm/batchnorm/mulMul>transformer_decoder/feedforward_layer_norm/batchnorm/Rsqrt:y:0Otransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
:transformer_decoder/feedforward_layer_norm/batchnorm/mul_1Multransformer_decoder/add_2:z:0<transformer_decoder/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
:transformer_decoder/feedforward_layer_norm/batchnorm/mul_2Mul@transformer_decoder/feedforward_layer_norm/moments/mean:output:0<transformer_decoder/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Ctransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOpLtransformer_decoder_feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8transformer_decoder/feedforward_layer_norm/batchnorm/subSubKtransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOp:value:0>transformer_decoder/feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
:transformer_decoder/feedforward_layer_norm/batchnorm/add_1AddV2>transformer_decoder/feedforward_layer_norm/batchnorm/mul_1:z:0<transformer_decoder/feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:�������������������f
$transformer_decoder_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
 transformer_decoder_1/ExpandDims
ExpandDims)token_and_position_embedding/NotEqual:z:0-transformer_decoder_1/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
transformer_decoder_1/CastCast)transformer_decoder_1/ExpandDims:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :�������������������
transformer_decoder_1/ShapeShape>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:s
)transformer_decoder_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+transformer_decoder_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#transformer_decoder_1/strided_sliceStridedSlice$transformer_decoder_1/Shape:output:02transformer_decoder_1/strided_slice/stack:output:04transformer_decoder_1/strided_slice/stack_1:output:04transformer_decoder_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
+transformer_decoder_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%transformer_decoder_1/strided_slice_1StridedSlice$transformer_decoder_1/Shape:output:04transformer_decoder_1/strided_slice_1/stack:output:06transformer_decoder_1/strided_slice_1/stack_1:output:06transformer_decoder_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
transformer_decoder_1/Shape_1Shape>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:u
+transformer_decoder_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-transformer_decoder_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%transformer_decoder_1/strided_slice_2StridedSlice&transformer_decoder_1/Shape_1:output:04transformer_decoder_1/strided_slice_2/stack:output:06transformer_decoder_1/strided_slice_2/stack_1:output:06transformer_decoder_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
+transformer_decoder_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%transformer_decoder_1/strided_slice_3StridedSlice&transformer_decoder_1/Shape_1:output:04transformer_decoder_1/strided_slice_3/stack:output:06transformer_decoder_1/strided_slice_3/stack_1:output:06transformer_decoder_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
!transformer_decoder_1/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    f
!transformer_decoder_1/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
 transformer_decoder_1/range/CastCast.transformer_decoder_1/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder_1/rangeRange*transformer_decoder_1/range/start:output:0$transformer_decoder_1/range/Cast:y:0*transformer_decoder_1/range/delta:output:0*

Tidx0*#
_output_shapes
:���������`
transformer_decoder_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : }
transformer_decoder_1/Cast_1Cast'transformer_decoder_1/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder_1/addAddV2$transformer_decoder_1/range:output:0 transformer_decoder_1/Cast_1:y:0*
T0*#
_output_shapes
:���������h
&transformer_decoder_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
"transformer_decoder_1/ExpandDims_1
ExpandDimstransformer_decoder_1/add:z:0/transformer_decoder_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:���������h
#transformer_decoder_1/range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    h
#transformer_decoder_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"transformer_decoder_1/range_1/CastCast.transformer_decoder_1/strided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
transformer_decoder_1/range_1Range,transformer_decoder_1/range_1/start:output:0&transformer_decoder_1/range_1/Cast:y:0,transformer_decoder_1/range_1/delta:output:0*

Tidx0*#
_output_shapes
:����������
"transformer_decoder_1/GreaterEqualGreaterEqual+transformer_decoder_1/ExpandDims_1:output:0&transformer_decoder_1/range_1:output:0*
T0*0
_output_shapes
:������������������h
&transformer_decoder_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : �
"transformer_decoder_1/ExpandDims_2
ExpandDims&transformer_decoder_1/GreaterEqual:z:0/transformer_decoder_1/ExpandDims_2/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
transformer_decoder_1/packedPack,transformer_decoder_1/strided_slice:output:0.transformer_decoder_1/strided_slice_3:output:0.transformer_decoder_1/strided_slice_3:output:0*
N*
T0*
_output_shapes
:\
transformer_decoder_1/RankConst*
_output_shapes
: *
dtype0*
value	B :�
!transformer_decoder_1/BroadcastToBroadcastTo+transformer_decoder_1/ExpandDims_2:output:0%transformer_decoder_1/packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
transformer_decoder_1/Cast_2Cast*transformer_decoder_1/BroadcastTo:output:0*

DstT0*

SrcT0
*=
_output_shapes+
):'����������������������������
transformer_decoder_1/MinimumMinimumtransformer_decoder_1/Cast:y:0 transformer_decoder_1/Cast_2:y:0*
T0*=
_output_shapes+
):'����������������������������
Gtransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_decoder_1_self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
8transformer_decoder_1/self_attention/query/einsum/EinsumEinsum>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0Otransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
=transformer_decoder_1/self_attention/query/add/ReadVariableOpReadVariableOpFtransformer_decoder_1_self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
.transformer_decoder_1/self_attention/query/addAddV2Atransformer_decoder_1/self_attention/query/einsum/Einsum:output:0Etransformer_decoder_1/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Etransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_decoder_1_self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
6transformer_decoder_1/self_attention/key/einsum/EinsumEinsum>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0Mtransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
;transformer_decoder_1/self_attention/key/add/ReadVariableOpReadVariableOpDtransformer_decoder_1_self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
,transformer_decoder_1/self_attention/key/addAddV2?transformer_decoder_1/self_attention/key/einsum/Einsum:output:0Ctransformer_decoder_1/self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
Gtransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpPtransformer_decoder_1_self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
8transformer_decoder_1/self_attention/value/einsum/EinsumEinsum>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0Otransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
=transformer_decoder_1/self_attention/value/add/ReadVariableOpReadVariableOpFtransformer_decoder_1_self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
.transformer_decoder_1/self_attention/value/addAddV2Atransformer_decoder_1/self_attention/value/einsum/Einsum:output:0Etransformer_decoder_1/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������Um
+transformer_decoder_1/self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :U�
)transformer_decoder_1/self_attention/CastCast4transformer_decoder_1/self_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
)transformer_decoder_1/self_attention/SqrtSqrt-transformer_decoder_1/self_attention/Cast:y:0*
T0*
_output_shapes
: s
.transformer_decoder_1/self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,transformer_decoder_1/self_attention/truedivRealDiv7transformer_decoder_1/self_attention/truediv/x:output:0-transformer_decoder_1/self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
(transformer_decoder_1/self_attention/MulMul2transformer_decoder_1/self_attention/query/add:z:00transformer_decoder_1/self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
2transformer_decoder_1/self_attention/einsum/EinsumEinsum0transformer_decoder_1/self_attention/key/add:z:0,transformer_decoder_1/self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbe~
3transformer_decoder_1/self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
/transformer_decoder_1/self_attention/ExpandDims
ExpandDims!transformer_decoder_1/Minimum:z:0<transformer_decoder_1/self_attention/ExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+����������������������������
3transformer_decoder_1/self_attention/softmax_1/CastCast8transformer_decoder_1/self_attention/ExpandDims:output:0*

DstT0*

SrcT0*A
_output_shapes/
-:+���������������������������y
4transformer_decoder_1/self_attention/softmax_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
2transformer_decoder_1/self_attention/softmax_1/subSub=transformer_decoder_1/self_attention/softmax_1/sub/x:output:07transformer_decoder_1/self_attention/softmax_1/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������y
4transformer_decoder_1/self_attention/softmax_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
2transformer_decoder_1/self_attention/softmax_1/mulMul6transformer_decoder_1/self_attention/softmax_1/sub:z:0=transformer_decoder_1/self_attention/softmax_1/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
2transformer_decoder_1/self_attention/softmax_1/addAddV2;transformer_decoder_1/self_attention/einsum/Einsum:output:06transformer_decoder_1/self_attention/softmax_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
6transformer_decoder_1/self_attention/softmax_1/SoftmaxSoftmax6transformer_decoder_1/self_attention/softmax_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
4transformer_decoder_1/self_attention/einsum_1/EinsumEinsum@transformer_decoder_1/self_attention/softmax_1/Softmax:softmax:02transformer_decoder_1/self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
Rtransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp[transformer_decoder_1_self_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
Ctransformer_decoder_1/self_attention/attention_output/einsum/EinsumEinsum=transformer_decoder_1/self_attention/einsum_1/Einsum:output:0Ztransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
Htransformer_decoder_1/self_attention/attention_output/add/ReadVariableOpReadVariableOpQtransformer_decoder_1_self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
9transformer_decoder_1/self_attention/attention_output/addAddV2Ltransformer_decoder_1/self_attention/attention_output/einsum/Einsum:output:0Ptransformer_decoder_1/self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
transformer_decoder_1/add_1AddV2=transformer_decoder_1/self_attention/attention_output/add:z:0>transformer_decoder/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Ntransformer_decoder_1/self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
<transformer_decoder_1/self_attention_layer_norm/moments/meanMeantransformer_decoder_1/add_1:z:0Wtransformer_decoder_1/self_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Dtransformer_decoder_1/self_attention_layer_norm/moments/StopGradientStopGradientEtransformer_decoder_1/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Itransformer_decoder_1/self_attention_layer_norm/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_1/add_1:z:0Mtransformer_decoder_1/self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Rtransformer_decoder_1/self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
@transformer_decoder_1/self_attention_layer_norm/moments/varianceMeanMtransformer_decoder_1/self_attention_layer_norm/moments/SquaredDifference:z:0[transformer_decoder_1/self_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
?transformer_decoder_1/self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
=transformer_decoder_1/self_attention_layer_norm/batchnorm/addAddV2Itransformer_decoder_1/self_attention_layer_norm/moments/variance:output:0Htransformer_decoder_1/self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
?transformer_decoder_1/self_attention_layer_norm/batchnorm/RsqrtRsqrtAtransformer_decoder_1/self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Ltransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpUtransformer_decoder_1_self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_decoder_1/self_attention_layer_norm/batchnorm/mulMulCtransformer_decoder_1/self_attention_layer_norm/batchnorm/Rsqrt:y:0Ttransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
?transformer_decoder_1/self_attention_layer_norm/batchnorm/mul_1Multransformer_decoder_1/add_1:z:0Atransformer_decoder_1/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
?transformer_decoder_1/self_attention_layer_norm/batchnorm/mul_2MulEtransformer_decoder_1/self_attention_layer_norm/moments/mean:output:0Atransformer_decoder_1/self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Htransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOpQtransformer_decoder_1_self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
=transformer_decoder_1/self_attention_layer_norm/batchnorm/subSubPtransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOp:value:0Ctransformer_decoder_1/self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
?transformer_decoder_1/self_attention_layer_norm/batchnorm/add_1AddV2Ctransformer_decoder_1/self_attention_layer_norm/batchnorm/mul_1:z:0Atransformer_decoder_1/self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOpVtransformer_decoder_1_feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Ctransformer_decoder_1/feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
Ctransformer_decoder_1/feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Dtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ShapeShapeCtransformer_decoder_1/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:�
Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Gtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2GatherV2Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Shape:output:0Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/free:output:0Utransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Ntransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Itransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV2Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Shape:output:0Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/axes:output:0Wtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Dtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Ctransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ProdProdPtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
Ftransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Etransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Prod_1ProdRtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2_1:output:0Otransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Jtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Etransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concatConcatV2Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/free:output:0Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/axes:output:0Stransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/stackPackLtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Prod:output:0Ntransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Htransformer_decoder_1/feedforward_intermediate_dense/Tensordot/transpose	TransposeCtransformer_decoder_1/self_attention_layer_norm/batchnorm/add_1:z:0Ntransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
Ftransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReshapeReshapeLtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/transpose:y:0Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Etransformer_decoder_1/feedforward_intermediate_dense/Tensordot/MatMulMatMulOtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Reshape:output:0Utransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Ftransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Ltransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Gtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1ConcatV2Ptransformer_decoder_1/feedforward_intermediate_dense/Tensordot/GatherV2:output:0Otransformer_decoder_1/feedforward_intermediate_dense/Tensordot/Const_2:output:0Utransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
>transformer_decoder_1/feedforward_intermediate_dense/TensordotReshapeOtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/MatMul:product:0Ptransformer_decoder_1/feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Ktransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOpTtransformer_decoder_1_feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<transformer_decoder_1/feedforward_intermediate_dense/BiasAddBiasAddGtransformer_decoder_1/feedforward_intermediate_dense/Tensordot:output:0Stransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
9transformer_decoder_1/feedforward_intermediate_dense/ReluReluEtransformer_decoder_1/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
Gtransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOpPtransformer_decoder_1_feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
=transformer_decoder_1/feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
=transformer_decoder_1/feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
>transformer_decoder_1/feedforward_output_dense/Tensordot/ShapeShapeGtransformer_decoder_1/feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:�
Ftransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Atransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2GatherV2Gtransformer_decoder_1/feedforward_output_dense/Tensordot/Shape:output:0Ftransformer_decoder_1/feedforward_output_dense/Tensordot/free:output:0Otransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Htransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Ctransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1GatherV2Gtransformer_decoder_1/feedforward_output_dense/Tensordot/Shape:output:0Ftransformer_decoder_1/feedforward_output_dense/Tensordot/axes:output:0Qtransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
>transformer_decoder_1/feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
=transformer_decoder_1/feedforward_output_dense/Tensordot/ProdProdJtransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2:output:0Gtransformer_decoder_1/feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: �
@transformer_decoder_1/feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
?transformer_decoder_1/feedforward_output_dense/Tensordot/Prod_1ProdLtransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2_1:output:0Itransformer_decoder_1/feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Dtransformer_decoder_1/feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
?transformer_decoder_1/feedforward_output_dense/Tensordot/concatConcatV2Ftransformer_decoder_1/feedforward_output_dense/Tensordot/free:output:0Ftransformer_decoder_1/feedforward_output_dense/Tensordot/axes:output:0Mtransformer_decoder_1/feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
>transformer_decoder_1/feedforward_output_dense/Tensordot/stackPackFtransformer_decoder_1/feedforward_output_dense/Tensordot/Prod:output:0Htransformer_decoder_1/feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Btransformer_decoder_1/feedforward_output_dense/Tensordot/transpose	TransposeGtransformer_decoder_1/feedforward_intermediate_dense/Relu:activations:0Htransformer_decoder_1/feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
@transformer_decoder_1/feedforward_output_dense/Tensordot/ReshapeReshapeFtransformer_decoder_1/feedforward_output_dense/Tensordot/transpose:y:0Gtransformer_decoder_1/feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
?transformer_decoder_1/feedforward_output_dense/Tensordot/MatMulMatMulItransformer_decoder_1/feedforward_output_dense/Tensordot/Reshape:output:0Otransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
@transformer_decoder_1/feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
Ftransformer_decoder_1/feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Atransformer_decoder_1/feedforward_output_dense/Tensordot/concat_1ConcatV2Jtransformer_decoder_1/feedforward_output_dense/Tensordot/GatherV2:output:0Itransformer_decoder_1/feedforward_output_dense/Tensordot/Const_2:output:0Otransformer_decoder_1/feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
8transformer_decoder_1/feedforward_output_dense/TensordotReshapeItransformer_decoder_1/feedforward_output_dense/Tensordot/MatMul:product:0Jtransformer_decoder_1/feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
Etransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOpNtransformer_decoder_1_feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6transformer_decoder_1/feedforward_output_dense/BiasAddBiasAddAtransformer_decoder_1/feedforward_output_dense/Tensordot:output:0Mtransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
transformer_decoder_1/add_2AddV2?transformer_decoder_1/feedforward_output_dense/BiasAdd:output:0Ctransformer_decoder_1/self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Ktransformer_decoder_1/feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
9transformer_decoder_1/feedforward_layer_norm/moments/meanMeantransformer_decoder_1/add_2:z:0Ttransformer_decoder_1/feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
Atransformer_decoder_1/feedforward_layer_norm/moments/StopGradientStopGradientBtransformer_decoder_1/feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Ftransformer_decoder_1/feedforward_layer_norm/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_1/add_2:z:0Jtransformer_decoder_1/feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
Otransformer_decoder_1/feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
=transformer_decoder_1/feedforward_layer_norm/moments/varianceMeanJtransformer_decoder_1/feedforward_layer_norm/moments/SquaredDifference:z:0Xtransformer_decoder_1/feedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
<transformer_decoder_1/feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
:transformer_decoder_1/feedforward_layer_norm/batchnorm/addAddV2Ftransformer_decoder_1/feedforward_layer_norm/moments/variance:output:0Etransformer_decoder_1/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
<transformer_decoder_1/feedforward_layer_norm/batchnorm/RsqrtRsqrt>transformer_decoder_1/feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
Itransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_decoder_1_feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:transformer_decoder_1/feedforward_layer_norm/batchnorm/mulMul@transformer_decoder_1/feedforward_layer_norm/batchnorm/Rsqrt:y:0Qtransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
<transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_1Multransformer_decoder_1/add_2:z:0>transformer_decoder_1/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
<transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_2MulBtransformer_decoder_1/feedforward_layer_norm/moments/mean:output:0>transformer_decoder_1/feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Etransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOpNtransformer_decoder_1_feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:transformer_decoder_1/feedforward_layer_norm/batchnorm/subSubMtransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOp:value:0@transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
<transformer_decoder_1/feedforward_layer_norm/batchnorm/add_1AddV2@transformer_decoder_1/feedforward_layer_norm/batchnorm/mul_1:z:0>transformer_decoder_1/feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
��'*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/Tensordot/ShapeShape@transformer_decoder_1/feedforward_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense/Tensordot/transpose	Transpose@transformer_decoder_1/feedforward_layer_norm/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������'b
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�'_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:�������������������'
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�'*
dtype0�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:�������������������'s
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOpE^token_and_position_embedding/position_embedding/Slice/ReadVariableOp>^token_and_position_embedding/token_embedding/embedding_lookupJ^transformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOpL^transformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOpD^transformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOpH^transformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOpD^transformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOpF^transformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOpG^transformer_decoder/self_attention/attention_output/add/ReadVariableOpQ^transformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp:^transformer_decoder/self_attention/key/add/ReadVariableOpD^transformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp<^transformer_decoder/self_attention/query/add/ReadVariableOpF^transformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp<^transformer_decoder/self_attention/value/add/ReadVariableOpF^transformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOpG^transformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOpK^transformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOpL^transformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOpN^transformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOpF^transformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOpJ^transformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOpF^transformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOpH^transformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOpI^transformer_decoder_1/self_attention/attention_output/add/ReadVariableOpS^transformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp<^transformer_decoder_1/self_attention/key/add/ReadVariableOpF^transformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp>^transformer_decoder_1/self_attention/query/add/ReadVariableOpH^transformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp>^transformer_decoder_1/self_attention/value/add/ReadVariableOpH^transformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOpI^transformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOpM^transformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2�
Dtoken_and_position_embedding/position_embedding/Slice/ReadVariableOpDtoken_and_position_embedding/position_embedding/Slice/ReadVariableOp2~
=token_and_position_embedding/token_embedding/embedding_lookup=token_and_position_embedding/token_embedding/embedding_lookup2�
Itransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOpItransformer_decoder/feedforward_intermediate_dense/BiasAdd/ReadVariableOp2�
Ktransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOpKtransformer_decoder/feedforward_intermediate_dense/Tensordot/ReadVariableOp2�
Ctransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOpCtransformer_decoder/feedforward_layer_norm/batchnorm/ReadVariableOp2�
Gtransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOpGtransformer_decoder/feedforward_layer_norm/batchnorm/mul/ReadVariableOp2�
Ctransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOpCtransformer_decoder/feedforward_output_dense/BiasAdd/ReadVariableOp2�
Etransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOpEtransformer_decoder/feedforward_output_dense/Tensordot/ReadVariableOp2�
Ftransformer_decoder/self_attention/attention_output/add/ReadVariableOpFtransformer_decoder/self_attention/attention_output/add/ReadVariableOp2�
Ptransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOpPtransformer_decoder/self_attention/attention_output/einsum/Einsum/ReadVariableOp2v
9transformer_decoder/self_attention/key/add/ReadVariableOp9transformer_decoder/self_attention/key/add/ReadVariableOp2�
Ctransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOpCtransformer_decoder/self_attention/key/einsum/Einsum/ReadVariableOp2z
;transformer_decoder/self_attention/query/add/ReadVariableOp;transformer_decoder/self_attention/query/add/ReadVariableOp2�
Etransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOpEtransformer_decoder/self_attention/query/einsum/Einsum/ReadVariableOp2z
;transformer_decoder/self_attention/value/add/ReadVariableOp;transformer_decoder/self_attention/value/add/ReadVariableOp2�
Etransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOpEtransformer_decoder/self_attention/value/einsum/Einsum/ReadVariableOp2�
Ftransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOpFtransformer_decoder/self_attention_layer_norm/batchnorm/ReadVariableOp2�
Jtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOpJtransformer_decoder/self_attention_layer_norm/batchnorm/mul/ReadVariableOp2�
Ktransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOpKtransformer_decoder_1/feedforward_intermediate_dense/BiasAdd/ReadVariableOp2�
Mtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOpMtransformer_decoder_1/feedforward_intermediate_dense/Tensordot/ReadVariableOp2�
Etransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOpEtransformer_decoder_1/feedforward_layer_norm/batchnorm/ReadVariableOp2�
Itransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOpItransformer_decoder_1/feedforward_layer_norm/batchnorm/mul/ReadVariableOp2�
Etransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOpEtransformer_decoder_1/feedforward_output_dense/BiasAdd/ReadVariableOp2�
Gtransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOpGtransformer_decoder_1/feedforward_output_dense/Tensordot/ReadVariableOp2�
Htransformer_decoder_1/self_attention/attention_output/add/ReadVariableOpHtransformer_decoder_1/self_attention/attention_output/add/ReadVariableOp2�
Rtransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOpRtransformer_decoder_1/self_attention/attention_output/einsum/Einsum/ReadVariableOp2z
;transformer_decoder_1/self_attention/key/add/ReadVariableOp;transformer_decoder_1/self_attention/key/add/ReadVariableOp2�
Etransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOpEtransformer_decoder_1/self_attention/key/einsum/Einsum/ReadVariableOp2~
=transformer_decoder_1/self_attention/query/add/ReadVariableOp=transformer_decoder_1/self_attention/query/add/ReadVariableOp2�
Gtransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOpGtransformer_decoder_1/self_attention/query/einsum/Einsum/ReadVariableOp2~
=transformer_decoder_1/self_attention/value/add/ReadVariableOp=transformer_decoder_1/self_attention/value/add/ReadVariableOp2�
Gtransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOpGtransformer_decoder_1/self_attention/value/einsum/Einsum/ReadVariableOp2�
Htransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOpHtransformer_decoder_1/self_attention_layer_norm/batchnorm/ReadVariableOp2�
Ltransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOpLtransformer_decoder_1/self_attention_layer_norm/batchnorm/mul/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
3__inference_transformer_decoder_layer_call_fn_46041
decoder_sequence
unknown:�U
	unknown_0:U 
	unknown_1:�U
	unknown_2:U 
	unknown_3:�U
	unknown_4:U 
	unknown_5:U�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldecoder_sequenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_44346}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
�.
�
@__inference_model_layer_call_and_return_conditional_losses_44786
input_16
"token_and_position_embedding_44707:
�'�6
"token_and_position_embedding_44709:
��0
transformer_decoder_44714:�U+
transformer_decoder_44716:U0
transformer_decoder_44718:�U+
transformer_decoder_44720:U0
transformer_decoder_44722:�U+
transformer_decoder_44724:U0
transformer_decoder_44726:U�(
transformer_decoder_44728:	�(
transformer_decoder_44730:	�(
transformer_decoder_44732:	�-
transformer_decoder_44734:
��(
transformer_decoder_44736:	�-
transformer_decoder_44738:
��(
transformer_decoder_44740:	�(
transformer_decoder_44742:	�(
transformer_decoder_44744:	�2
transformer_decoder_1_44747:�U-
transformer_decoder_1_44749:U2
transformer_decoder_1_44751:�U-
transformer_decoder_1_44753:U2
transformer_decoder_1_44755:�U-
transformer_decoder_1_44757:U2
transformer_decoder_1_44759:U�*
transformer_decoder_1_44761:	�*
transformer_decoder_1_44763:	�*
transformer_decoder_1_44765:	�/
transformer_decoder_1_44767:
��*
transformer_decoder_1_44769:	�/
transformer_decoder_1_44771:
��*
transformer_decoder_1_44773:	�*
transformer_decoder_1_44775:	�*
transformer_decoder_1_44777:	�
dense_44780:
��'
dense_44782:	�'
identity��dense/StatefulPartitionedCall�4token_and_position_embedding/StatefulPartitionedCall�+transformer_decoder/StatefulPartitionedCall�-transformer_decoder_1/StatefulPartitionedCall�
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1"token_and_position_embedding_44707"token_and_position_embedding_44709*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_43335i
'token_and_position_embedding/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : �
%token_and_position_embedding/NotEqualNotEqualinput_10token_and_position_embedding/NotEqual/y:output:0*
T0*0
_output_shapes
:�������������������
+transformer_decoder/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0transformer_decoder_44714transformer_decoder_44716transformer_decoder_44718transformer_decoder_44720transformer_decoder_44722transformer_decoder_44724transformer_decoder_44726transformer_decoder_44728transformer_decoder_44730transformer_decoder_44732transformer_decoder_44734transformer_decoder_44736transformer_decoder_44738transformer_decoder_44740transformer_decoder_44742transformer_decoder_44744*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_43519�
-transformer_decoder_1/StatefulPartitionedCallStatefulPartitionedCall4transformer_decoder/StatefulPartitionedCall:output:0transformer_decoder_1_44747transformer_decoder_1_44749transformer_decoder_1_44751transformer_decoder_1_44753transformer_decoder_1_44755transformer_decoder_1_44757transformer_decoder_1_44759transformer_decoder_1_44761transformer_decoder_1_44763transformer_decoder_1_44765transformer_decoder_1_44767transformer_decoder_1_44769transformer_decoder_1_44771transformer_decoder_1_44773transformer_decoder_1_44775transformer_decoder_1_44777*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_43729�
dense/StatefulPartitionedCallStatefulPartitionedCall6transformer_decoder_1/StatefulPartitionedCall:output:0dense_44780dense_44782*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������'*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_43793�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������'�
NoOpNoOp^dense/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_decoder/StatefulPartitionedCall.^transformer_decoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_decoder/StatefulPartitionedCall+transformer_decoder/StatefulPartitionedCall2^
-transformer_decoder_1/StatefulPartitionedCall-transformer_decoder_1/StatefulPartitionedCall:Y U
0
_output_shapes
:������������������
!
_user_specified_name	input_1
��
�
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_43519
decoder_sequenceQ
:self_attention_query_einsum_einsum_readvariableop_resource:�UB
0self_attention_query_add_readvariableop_resource:UO
8self_attention_key_einsum_einsum_readvariableop_resource:�U@
.self_attention_key_add_readvariableop_resource:UQ
:self_attention_value_einsum_einsum_readvariableop_resource:�UB
0self_attention_value_add_readvariableop_resource:U\
Eself_attention_attention_output_einsum_einsum_readvariableop_resource:U�J
;self_attention_attention_output_add_readvariableop_resource:	�N
?self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�J
;self_attention_layer_norm_batchnorm_readvariableop_resource:	�T
@feedforward_intermediate_dense_tensordot_readvariableop_resource:
��M
>feedforward_intermediate_dense_biasadd_readvariableop_resource:	�N
:feedforward_output_dense_tensordot_readvariableop_resource:
��G
8feedforward_output_dense_biasadd_readvariableop_resource:	�K
<feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�G
8feedforward_layer_norm_batchnorm_readvariableop_resource:	�
identity��5feedforward_intermediate_dense/BiasAdd/ReadVariableOp�7feedforward_intermediate_dense/Tensordot/ReadVariableOp�/feedforward_layer_norm/batchnorm/ReadVariableOp�3feedforward_layer_norm/batchnorm/mul/ReadVariableOp�/feedforward_output_dense/BiasAdd/ReadVariableOp�1feedforward_output_dense/Tensordot/ReadVariableOp�2self_attention/attention_output/add/ReadVariableOp�<self_attention/attention_output/einsum/Einsum/ReadVariableOp�%self_attention/key/add/ReadVariableOp�/self_attention/key/einsum/Einsum/ReadVariableOp�'self_attention/query/add/ReadVariableOp�1self_attention/query/einsum/Einsum/ReadVariableOp�'self_attention/value/add/ReadVariableOp�1self_attention/value/einsum/Einsum/ReadVariableOp�2self_attention_layer_norm/batchnorm/ReadVariableOp�6self_attention_layer_norm/batchnorm/mul/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
Shape_1Shapedecoder_sequence*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\

range/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: {
rangeRangerange/start:output:0range/Cast:y:0range/delta:output:0*

Tidx0*#
_output_shapes
:���������H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B : M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
addAddV2range:output:0Cast:y:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :l

ExpandDims
ExpandDimsadd:z:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������R
range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    R
range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
range_1/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
range_1Rangerange_1/start:output:0range_1/Cast:y:0range_1/delta:output:0*

Tidx0*#
_output_shapes
:���������~
GreaterEqualGreaterEqualExpandDims:output:0range_1:output:0*
T0*0
_output_shapes
:������������������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsGreaterEqual:z:0ExpandDims_1/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
packedPackstrided_slice:output:0strided_slice_3:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :�
BroadcastToBroadcastToExpandDims_1:output:0packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
1self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/query/einsum/EinsumEinsumdecoder_sequence9self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/query/add/ReadVariableOpReadVariableOp0self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/query/addAddV2+self_attention/query/einsum/Einsum:output:0/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp8self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
 self_attention/key/einsum/EinsumEinsumdecoder_sequence7self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
%self_attention/key/add/ReadVariableOpReadVariableOp.self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/key/addAddV2)self_attention/key/einsum/Einsum:output:0-self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
1self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/value/einsum/EinsumEinsumdecoder_sequence9self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/value/add/ReadVariableOpReadVariableOp0self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/value/addAddV2+self_attention/value/einsum/Einsum:output:0/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������UW
self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Uk
self_attention/CastCastself_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: U
self_attention/SqrtSqrtself_attention/Cast:y:0*
T0*
_output_shapes
: ]
self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
self_attention/truedivRealDiv!self_attention/truediv/x:output:0self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
self_attention/MulMulself_attention/query/add:z:0self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
self_attention/einsum/EinsumEinsumself_attention/key/add:z:0self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbeh
self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
self_attention/ExpandDims
ExpandDimsBroadcastTo:output:0&self_attention/ExpandDims/dim:output:0*
T0
*A
_output_shapes/
-:+����������������������������
self_attention/softmax/CastCast"self_attention/ExpandDims:output:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+���������������������������a
self_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
self_attention/softmax/subSub%self_attention/softmax/sub/x:output:0self_attention/softmax/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������a
self_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
self_attention/softmax/mulMulself_attention/softmax/sub:z:0%self_attention/softmax/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax/addAddV2%self_attention/einsum/Einsum:output:0self_attention/softmax/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax/SoftmaxSoftmaxself_attention/softmax/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/dropout/IdentityIdentity(self_attention/softmax/Softmax:softmax:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/einsum_1/EinsumEinsum(self_attention/dropout/Identity:output:0self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
<self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpEself_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
-self_attention/attention_output/einsum/EinsumEinsum'self_attention/einsum_1/Einsum:output:0Dself_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
2self_attention/attention_output/add/ReadVariableOpReadVariableOp;self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#self_attention/attention_output/addAddV26self_attention/attention_output/einsum/Einsum:output:0:self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
self_attention_dropout/IdentityIdentity'self_attention/attention_output/add:z:0*
T0*5
_output_shapes#
!:��������������������
add_1AddV2(self_attention_dropout/Identity:output:0decoder_sequence*
T0*5
_output_shapes#
!:��������������������
8self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&self_attention_layer_norm/moments/meanMean	add_1:z:0Aself_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
.self_attention_layer_norm/moments/StopGradientStopGradient/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
3self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference	add_1:z:07self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
<self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
*self_attention_layer_norm/moments/varianceMean7self_attention_layer_norm/moments/SquaredDifference:z:0Eself_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(n
)self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
'self_attention_layer_norm/batchnorm/addAddV23self_attention_layer_norm/moments/variance:output:02self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
)self_attention_layer_norm/batchnorm/RsqrtRsqrt+self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
6self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp?self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/mulMul-self_attention_layer_norm/batchnorm/Rsqrt:y:0>self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_1Mul	add_1:z:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_2Mul/self_attention_layer_norm/moments/mean:output:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
2self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOp;self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/subSub:self_attention_layer_norm/batchnorm/ReadVariableOp:value:0-self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/add_1AddV2-self_attention_layer_norm/batchnorm/mul_1:z:0+self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
7feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOp@feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
-feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
.feedforward_intermediate_dense/Tensordot/ShapeShape-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:x
6feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/GatherV2GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/free:output:0?feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/axes:output:0Afeedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
-feedforward_intermediate_dense/Tensordot/ProdProd:feedforward_intermediate_dense/Tensordot/GatherV2:output:07feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
/feedforward_intermediate_dense/Tensordot/Prod_1Prod<feedforward_intermediate_dense/Tensordot/GatherV2_1:output:09feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/feedforward_intermediate_dense/Tensordot/concatConcatV26feedforward_intermediate_dense/Tensordot/free:output:06feedforward_intermediate_dense/Tensordot/axes:output:0=feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
.feedforward_intermediate_dense/Tensordot/stackPack6feedforward_intermediate_dense/Tensordot/Prod:output:08feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
2feedforward_intermediate_dense/Tensordot/transpose	Transpose-self_attention_layer_norm/batchnorm/add_1:z:08feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
0feedforward_intermediate_dense/Tensordot/ReshapeReshape6feedforward_intermediate_dense/Tensordot/transpose:y:07feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
/feedforward_intermediate_dense/Tensordot/MatMulMatMul9feedforward_intermediate_dense/Tensordot/Reshape:output:0?feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
0feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�x
6feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/concat_1ConcatV2:feedforward_intermediate_dense/Tensordot/GatherV2:output:09feedforward_intermediate_dense/Tensordot/Const_2:output:0?feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_intermediate_dense/TensordotReshape9feedforward_intermediate_dense/Tensordot/MatMul:product:0:feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
5feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOp>feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&feedforward_intermediate_dense/BiasAddBiasAdd1feedforward_intermediate_dense/Tensordot:output:0=feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
#feedforward_intermediate_dense/ReluRelu/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
1feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOp:feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0q
'feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(feedforward_output_dense/Tensordot/ShapeShape1feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:r
0feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/GatherV2GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/free:output:09feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-feedforward_output_dense/Tensordot/GatherV2_1GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/axes:output:0;feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'feedforward_output_dense/Tensordot/ProdProd4feedforward_output_dense/Tensordot/GatherV2:output:01feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)feedforward_output_dense/Tensordot/Prod_1Prod6feedforward_output_dense/Tensordot/GatherV2_1:output:03feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)feedforward_output_dense/Tensordot/concatConcatV20feedforward_output_dense/Tensordot/free:output:00feedforward_output_dense/Tensordot/axes:output:07feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_output_dense/Tensordot/stackPack0feedforward_output_dense/Tensordot/Prod:output:02feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,feedforward_output_dense/Tensordot/transpose	Transpose1feedforward_intermediate_dense/Relu:activations:02feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
*feedforward_output_dense/Tensordot/ReshapeReshape0feedforward_output_dense/Tensordot/transpose:y:01feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)feedforward_output_dense/Tensordot/MatMulMatMul3feedforward_output_dense/Tensordot/Reshape:output:09feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
*feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�r
0feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/concat_1ConcatV24feedforward_output_dense/Tensordot/GatherV2:output:03feedforward_output_dense/Tensordot/Const_2:output:09feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"feedforward_output_dense/TensordotReshape3feedforward_output_dense/Tensordot/MatMul:product:04feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOp8feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 feedforward_output_dense/BiasAddBiasAdd+feedforward_output_dense/Tensordot:output:07feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
feedforward_dropout/IdentityIdentity)feedforward_output_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
add_2AddV2%feedforward_dropout/Identity:output:0-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:�������������������
5feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#feedforward_layer_norm/moments/meanMean	add_2:z:0>feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
+feedforward_layer_norm/moments/StopGradientStopGradient,feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
0feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference	add_2:z:04feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
9feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'feedforward_layer_norm/moments/varianceMean4feedforward_layer_norm/moments/SquaredDifference:z:0Bfeedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(k
&feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$feedforward_layer_norm/batchnorm/addAddV20feedforward_layer_norm/moments/variance:output:0/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
&feedforward_layer_norm/batchnorm/RsqrtRsqrt(feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
3feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp<feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/mulMul*feedforward_layer_norm/batchnorm/Rsqrt:y:0;feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_1Mul	add_2:z:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_2Mul,feedforward_layer_norm/moments/mean:output:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOp8feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/subSub7feedforward_layer_norm/batchnorm/ReadVariableOp:value:0*feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/add_1AddV2*feedforward_layer_norm/batchnorm/mul_1:z:0(feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentity*feedforward_layer_norm/batchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp6^feedforward_intermediate_dense/BiasAdd/ReadVariableOp8^feedforward_intermediate_dense/Tensordot/ReadVariableOp0^feedforward_layer_norm/batchnorm/ReadVariableOp4^feedforward_layer_norm/batchnorm/mul/ReadVariableOp0^feedforward_output_dense/BiasAdd/ReadVariableOp2^feedforward_output_dense/Tensordot/ReadVariableOp3^self_attention/attention_output/add/ReadVariableOp=^self_attention/attention_output/einsum/Einsum/ReadVariableOp&^self_attention/key/add/ReadVariableOp0^self_attention/key/einsum/Einsum/ReadVariableOp(^self_attention/query/add/ReadVariableOp2^self_attention/query/einsum/Einsum/ReadVariableOp(^self_attention/value/add/ReadVariableOp2^self_attention/value/einsum/Einsum/ReadVariableOp3^self_attention_layer_norm/batchnorm/ReadVariableOp7^self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 2n
5feedforward_intermediate_dense/BiasAdd/ReadVariableOp5feedforward_intermediate_dense/BiasAdd/ReadVariableOp2r
7feedforward_intermediate_dense/Tensordot/ReadVariableOp7feedforward_intermediate_dense/Tensordot/ReadVariableOp2b
/feedforward_layer_norm/batchnorm/ReadVariableOp/feedforward_layer_norm/batchnorm/ReadVariableOp2j
3feedforward_layer_norm/batchnorm/mul/ReadVariableOp3feedforward_layer_norm/batchnorm/mul/ReadVariableOp2b
/feedforward_output_dense/BiasAdd/ReadVariableOp/feedforward_output_dense/BiasAdd/ReadVariableOp2f
1feedforward_output_dense/Tensordot/ReadVariableOp1feedforward_output_dense/Tensordot/ReadVariableOp2h
2self_attention/attention_output/add/ReadVariableOp2self_attention/attention_output/add/ReadVariableOp2|
<self_attention/attention_output/einsum/Einsum/ReadVariableOp<self_attention/attention_output/einsum/Einsum/ReadVariableOp2N
%self_attention/key/add/ReadVariableOp%self_attention/key/add/ReadVariableOp2b
/self_attention/key/einsum/Einsum/ReadVariableOp/self_attention/key/einsum/Einsum/ReadVariableOp2R
'self_attention/query/add/ReadVariableOp'self_attention/query/add/ReadVariableOp2f
1self_attention/query/einsum/Einsum/ReadVariableOp1self_attention/query/einsum/Einsum/ReadVariableOp2R
'self_attention/value/add/ReadVariableOp'self_attention/value/add/ReadVariableOp2f
1self_attention/value/einsum/Einsum/ReadVariableOp1self_attention/value/einsum/Einsum/ReadVariableOp2h
2self_attention_layer_norm/batchnorm/ReadVariableOp2self_attention_layer_norm/batchnorm/ReadVariableOp2p
6self_attention_layer_norm/batchnorm/mul/ReadVariableOp6self_attention_layer_norm/batchnorm/mul/ReadVariableOp:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
��
�
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_44346
decoder_sequenceQ
:self_attention_query_einsum_einsum_readvariableop_resource:�UB
0self_attention_query_add_readvariableop_resource:UO
8self_attention_key_einsum_einsum_readvariableop_resource:�U@
.self_attention_key_add_readvariableop_resource:UQ
:self_attention_value_einsum_einsum_readvariableop_resource:�UB
0self_attention_value_add_readvariableop_resource:U\
Eself_attention_attention_output_einsum_einsum_readvariableop_resource:U�J
;self_attention_attention_output_add_readvariableop_resource:	�N
?self_attention_layer_norm_batchnorm_mul_readvariableop_resource:	�J
;self_attention_layer_norm_batchnorm_readvariableop_resource:	�T
@feedforward_intermediate_dense_tensordot_readvariableop_resource:
��M
>feedforward_intermediate_dense_biasadd_readvariableop_resource:	�N
:feedforward_output_dense_tensordot_readvariableop_resource:
��G
8feedforward_output_dense_biasadd_readvariableop_resource:	�K
<feedforward_layer_norm_batchnorm_mul_readvariableop_resource:	�G
8feedforward_layer_norm_batchnorm_readvariableop_resource:	�
identity��5feedforward_intermediate_dense/BiasAdd/ReadVariableOp�7feedforward_intermediate_dense/Tensordot/ReadVariableOp�/feedforward_layer_norm/batchnorm/ReadVariableOp�3feedforward_layer_norm/batchnorm/mul/ReadVariableOp�/feedforward_output_dense/BiasAdd/ReadVariableOp�1feedforward_output_dense/Tensordot/ReadVariableOp�2self_attention/attention_output/add/ReadVariableOp�<self_attention/attention_output/einsum/Einsum/ReadVariableOp�%self_attention/key/add/ReadVariableOp�/self_attention/key/einsum/Einsum/ReadVariableOp�'self_attention/query/add/ReadVariableOp�1self_attention/query/einsum/Einsum/ReadVariableOp�'self_attention/value/add/ReadVariableOp�1self_attention/value/einsum/Einsum/ReadVariableOp�2self_attention_layer_norm/batchnorm/ReadVariableOp�6self_attention_layer_norm/batchnorm/mul/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
Shape_1Shapedecoder_sequence*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\

range/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: {
rangeRangerange/start:output:0range/Cast:y:0range/delta:output:0*

Tidx0*#
_output_shapes
:���������H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B : M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: T
addAddV2range:output:0Cast:y:0*
T0*#
_output_shapes
:���������P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :l

ExpandDims
ExpandDimsadd:z:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:���������R
range_1/startConst*
_output_shapes
: *
dtype0*
valueB
 *    R
range_1/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
range_1/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*
_output_shapes
: �
range_1Rangerange_1/start:output:0range_1/Cast:y:0range_1/delta:output:0*

Tidx0*#
_output_shapes
:���������~
GreaterEqualGreaterEqualExpandDims:output:0range_1:output:0*
T0*0
_output_shapes
:������������������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
ExpandDims_1
ExpandDimsGreaterEqual:z:0ExpandDims_1/dim:output:0*
T0
*4
_output_shapes"
 :�������������������
packedPackstrided_slice:output:0strided_slice_3:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :�
BroadcastToBroadcastToExpandDims_1:output:0packed:output:0*
T0
*=
_output_shapes+
):'����������������������������
1self_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/query/einsum/EinsumEinsumdecoder_sequence9self_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/query/add/ReadVariableOpReadVariableOp0self_attention_query_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/query/addAddV2+self_attention/query/einsum/Einsum:output:0/self_attention/query/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
/self_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp8self_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
 self_attention/key/einsum/EinsumEinsumdecoder_sequence7self_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
%self_attention/key/add/ReadVariableOpReadVariableOp.self_attention_key_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/key/addAddV2)self_attention/key/einsum/Einsum:output:0-self_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������U�
1self_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp:self_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�U*
dtype0�
"self_attention/value/einsum/EinsumEinsumdecoder_sequence9self_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationabc,cde->abde�
'self_attention/value/add/ReadVariableOpReadVariableOp0self_attention_value_add_readvariableop_resource*
_output_shapes

:U*
dtype0�
self_attention/value/addAddV2+self_attention/value/einsum/Einsum:output:0/self_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������UW
self_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Uk
self_attention/CastCastself_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: U
self_attention/SqrtSqrtself_attention/Cast:y:0*
T0*
_output_shapes
: ]
self_attention/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
self_attention/truedivRealDiv!self_attention/truediv/x:output:0self_attention/Sqrt:y:0*
T0*
_output_shapes
: �
self_attention/MulMulself_attention/query/add:z:0self_attention/truediv:z:0*
T0*8
_output_shapes&
$:"������������������U�
self_attention/einsum/EinsumEinsumself_attention/key/add:z:0self_attention/Mul:z:0*
N*
T0*A
_output_shapes/
-:+���������������������������*
equationaecd,abcd->acbeh
self_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
self_attention/ExpandDims
ExpandDimsBroadcastTo:output:0&self_attention/ExpandDims/dim:output:0*
T0
*A
_output_shapes/
-:+����������������������������
self_attention/softmax/CastCast"self_attention/ExpandDims:output:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+���������������������������a
self_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
self_attention/softmax/subSub%self_attention/softmax/sub/x:output:0self_attention/softmax/Cast:y:0*
T0*A
_output_shapes/
-:+���������������������������a
self_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
self_attention/softmax/mulMulself_attention/softmax/sub:z:0%self_attention/softmax/mul/y:output:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax/addAddV2%self_attention/einsum/Einsum:output:0self_attention/softmax/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/softmax/SoftmaxSoftmaxself_attention/softmax/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
self_attention/einsum_1/EinsumEinsum(self_attention/softmax/Softmax:softmax:0self_attention/value/add:z:0*
N*
T0*8
_output_shapes&
$:"������������������U*
equationacbe,aecd->abcd�
<self_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpEself_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:U�*
dtype0�
-self_attention/attention_output/einsum/EinsumEinsum'self_attention/einsum_1/Einsum:output:0Dself_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
2self_attention/attention_output/add/ReadVariableOpReadVariableOp;self_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#self_attention/attention_output/addAddV26self_attention/attention_output/einsum/Einsum:output:0:self_attention/attention_output/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
add_1AddV2'self_attention/attention_output/add:z:0decoder_sequence*
T0*5
_output_shapes#
!:��������������������
8self_attention_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
&self_attention_layer_norm/moments/meanMean	add_1:z:0Aself_attention_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
.self_attention_layer_norm/moments/StopGradientStopGradient/self_attention_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
3self_attention_layer_norm/moments/SquaredDifferenceSquaredDifference	add_1:z:07self_attention_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
<self_attention_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
*self_attention_layer_norm/moments/varianceMean7self_attention_layer_norm/moments/SquaredDifference:z:0Eself_attention_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(n
)self_attention_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
'self_attention_layer_norm/batchnorm/addAddV23self_attention_layer_norm/moments/variance:output:02self_attention_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
)self_attention_layer_norm/batchnorm/RsqrtRsqrt+self_attention_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
6self_attention_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp?self_attention_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/mulMul-self_attention_layer_norm/batchnorm/Rsqrt:y:0>self_attention_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_1Mul	add_1:z:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/mul_2Mul/self_attention_layer_norm/moments/mean:output:0+self_attention_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
2self_attention_layer_norm/batchnorm/ReadVariableOpReadVariableOp;self_attention_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'self_attention_layer_norm/batchnorm/subSub:self_attention_layer_norm/batchnorm/ReadVariableOp:value:0-self_attention_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
)self_attention_layer_norm/batchnorm/add_1AddV2-self_attention_layer_norm/batchnorm/mul_1:z:0+self_attention_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
7feedforward_intermediate_dense/Tensordot/ReadVariableOpReadVariableOp@feedforward_intermediate_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
-feedforward_intermediate_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-feedforward_intermediate_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
.feedforward_intermediate_dense/Tensordot/ShapeShape-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*
_output_shapes
:x
6feedforward_intermediate_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/GatherV2GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/free:output:0?feedforward_intermediate_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8feedforward_intermediate_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
3feedforward_intermediate_dense/Tensordot/GatherV2_1GatherV27feedforward_intermediate_dense/Tensordot/Shape:output:06feedforward_intermediate_dense/Tensordot/axes:output:0Afeedforward_intermediate_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.feedforward_intermediate_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
-feedforward_intermediate_dense/Tensordot/ProdProd:feedforward_intermediate_dense/Tensordot/GatherV2:output:07feedforward_intermediate_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0feedforward_intermediate_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
/feedforward_intermediate_dense/Tensordot/Prod_1Prod<feedforward_intermediate_dense/Tensordot/GatherV2_1:output:09feedforward_intermediate_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4feedforward_intermediate_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/feedforward_intermediate_dense/Tensordot/concatConcatV26feedforward_intermediate_dense/Tensordot/free:output:06feedforward_intermediate_dense/Tensordot/axes:output:0=feedforward_intermediate_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
.feedforward_intermediate_dense/Tensordot/stackPack6feedforward_intermediate_dense/Tensordot/Prod:output:08feedforward_intermediate_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
2feedforward_intermediate_dense/Tensordot/transpose	Transpose-self_attention_layer_norm/batchnorm/add_1:z:08feedforward_intermediate_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
0feedforward_intermediate_dense/Tensordot/ReshapeReshape6feedforward_intermediate_dense/Tensordot/transpose:y:07feedforward_intermediate_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
/feedforward_intermediate_dense/Tensordot/MatMulMatMul9feedforward_intermediate_dense/Tensordot/Reshape:output:0?feedforward_intermediate_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
0feedforward_intermediate_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�x
6feedforward_intermediate_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1feedforward_intermediate_dense/Tensordot/concat_1ConcatV2:feedforward_intermediate_dense/Tensordot/GatherV2:output:09feedforward_intermediate_dense/Tensordot/Const_2:output:0?feedforward_intermediate_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_intermediate_dense/TensordotReshape9feedforward_intermediate_dense/Tensordot/MatMul:product:0:feedforward_intermediate_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
5feedforward_intermediate_dense/BiasAdd/ReadVariableOpReadVariableOp>feedforward_intermediate_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&feedforward_intermediate_dense/BiasAddBiasAdd1feedforward_intermediate_dense/Tensordot:output:0=feedforward_intermediate_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
#feedforward_intermediate_dense/ReluRelu/feedforward_intermediate_dense/BiasAdd:output:0*
T0*5
_output_shapes#
!:��������������������
1feedforward_output_dense/Tensordot/ReadVariableOpReadVariableOp:feedforward_output_dense_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0q
'feedforward_output_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'feedforward_output_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
(feedforward_output_dense/Tensordot/ShapeShape1feedforward_intermediate_dense/Relu:activations:0*
T0*
_output_shapes
:r
0feedforward_output_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/GatherV2GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/free:output:09feedforward_output_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2feedforward_output_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-feedforward_output_dense/Tensordot/GatherV2_1GatherV21feedforward_output_dense/Tensordot/Shape:output:00feedforward_output_dense/Tensordot/axes:output:0;feedforward_output_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(feedforward_output_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
'feedforward_output_dense/Tensordot/ProdProd4feedforward_output_dense/Tensordot/GatherV2:output:01feedforward_output_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*feedforward_output_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
)feedforward_output_dense/Tensordot/Prod_1Prod6feedforward_output_dense/Tensordot/GatherV2_1:output:03feedforward_output_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.feedforward_output_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)feedforward_output_dense/Tensordot/concatConcatV20feedforward_output_dense/Tensordot/free:output:00feedforward_output_dense/Tensordot/axes:output:07feedforward_output_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
(feedforward_output_dense/Tensordot/stackPack0feedforward_output_dense/Tensordot/Prod:output:02feedforward_output_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
,feedforward_output_dense/Tensordot/transpose	Transpose1feedforward_intermediate_dense/Relu:activations:02feedforward_output_dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:��������������������
*feedforward_output_dense/Tensordot/ReshapeReshape0feedforward_output_dense/Tensordot/transpose:y:01feedforward_output_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
)feedforward_output_dense/Tensordot/MatMulMatMul3feedforward_output_dense/Tensordot/Reshape:output:09feedforward_output_dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
*feedforward_output_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�r
0feedforward_output_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+feedforward_output_dense/Tensordot/concat_1ConcatV24feedforward_output_dense/Tensordot/GatherV2:output:03feedforward_output_dense/Tensordot/Const_2:output:09feedforward_output_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
"feedforward_output_dense/TensordotReshape3feedforward_output_dense/Tensordot/MatMul:product:04feedforward_output_dense/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_output_dense/BiasAdd/ReadVariableOpReadVariableOp8feedforward_output_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 feedforward_output_dense/BiasAddBiasAdd+feedforward_output_dense/Tensordot:output:07feedforward_output_dense/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
add_2AddV2)feedforward_output_dense/BiasAdd:output:0-self_attention_layer_norm/batchnorm/add_1:z:0*
T0*5
_output_shapes#
!:�������������������
5feedforward_layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
#feedforward_layer_norm/moments/meanMean	add_2:z:0>feedforward_layer_norm/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
+feedforward_layer_norm/moments/StopGradientStopGradient,feedforward_layer_norm/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
0feedforward_layer_norm/moments/SquaredDifferenceSquaredDifference	add_2:z:04feedforward_layer_norm/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
9feedforward_layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
'feedforward_layer_norm/moments/varianceMean4feedforward_layer_norm/moments/SquaredDifference:z:0Bfeedforward_layer_norm/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(k
&feedforward_layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$feedforward_layer_norm/batchnorm/addAddV20feedforward_layer_norm/moments/variance:output:0/feedforward_layer_norm/batchnorm/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
&feedforward_layer_norm/batchnorm/RsqrtRsqrt(feedforward_layer_norm/batchnorm/add:z:0*
T0*4
_output_shapes"
 :�������������������
3feedforward_layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp<feedforward_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/mulMul*feedforward_layer_norm/batchnorm/Rsqrt:y:0;feedforward_layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_1Mul	add_2:z:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/mul_2Mul,feedforward_layer_norm/moments/mean:output:0(feedforward_layer_norm/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:��������������������
/feedforward_layer_norm/batchnorm/ReadVariableOpReadVariableOp8feedforward_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$feedforward_layer_norm/batchnorm/subSub7feedforward_layer_norm/batchnorm/ReadVariableOp:value:0*feedforward_layer_norm/batchnorm/mul_2:z:0*
T0*5
_output_shapes#
!:��������������������
&feedforward_layer_norm/batchnorm/add_1AddV2*feedforward_layer_norm/batchnorm/mul_1:z:0(feedforward_layer_norm/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:��������������������
IdentityIdentity*feedforward_layer_norm/batchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:��������������������
NoOpNoOp6^feedforward_intermediate_dense/BiasAdd/ReadVariableOp8^feedforward_intermediate_dense/Tensordot/ReadVariableOp0^feedforward_layer_norm/batchnorm/ReadVariableOp4^feedforward_layer_norm/batchnorm/mul/ReadVariableOp0^feedforward_output_dense/BiasAdd/ReadVariableOp2^feedforward_output_dense/Tensordot/ReadVariableOp3^self_attention/attention_output/add/ReadVariableOp=^self_attention/attention_output/einsum/Einsum/ReadVariableOp&^self_attention/key/add/ReadVariableOp0^self_attention/key/einsum/Einsum/ReadVariableOp(^self_attention/query/add/ReadVariableOp2^self_attention/query/einsum/Einsum/ReadVariableOp(^self_attention/value/add/ReadVariableOp2^self_attention/value/einsum/Einsum/ReadVariableOp3^self_attention_layer_norm/batchnorm/ReadVariableOp7^self_attention_layer_norm/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 2n
5feedforward_intermediate_dense/BiasAdd/ReadVariableOp5feedforward_intermediate_dense/BiasAdd/ReadVariableOp2r
7feedforward_intermediate_dense/Tensordot/ReadVariableOp7feedforward_intermediate_dense/Tensordot/ReadVariableOp2b
/feedforward_layer_norm/batchnorm/ReadVariableOp/feedforward_layer_norm/batchnorm/ReadVariableOp2j
3feedforward_layer_norm/batchnorm/mul/ReadVariableOp3feedforward_layer_norm/batchnorm/mul/ReadVariableOp2b
/feedforward_output_dense/BiasAdd/ReadVariableOp/feedforward_output_dense/BiasAdd/ReadVariableOp2f
1feedforward_output_dense/Tensordot/ReadVariableOp1feedforward_output_dense/Tensordot/ReadVariableOp2h
2self_attention/attention_output/add/ReadVariableOp2self_attention/attention_output/add/ReadVariableOp2|
<self_attention/attention_output/einsum/Einsum/ReadVariableOp<self_attention/attention_output/einsum/Einsum/ReadVariableOp2N
%self_attention/key/add/ReadVariableOp%self_attention/key/add/ReadVariableOp2b
/self_attention/key/einsum/Einsum/ReadVariableOp/self_attention/key/einsum/Einsum/ReadVariableOp2R
'self_attention/query/add/ReadVariableOp'self_attention/query/add/ReadVariableOp2f
1self_attention/query/einsum/Einsum/ReadVariableOp1self_attention/query/einsum/Einsum/ReadVariableOp2R
'self_attention/value/add/ReadVariableOp'self_attention/value/add/ReadVariableOp2f
1self_attention/value/einsum/Einsum/ReadVariableOp1self_attention/value/einsum/Einsum/ReadVariableOp2h
2self_attention_layer_norm/batchnorm/ReadVariableOp2self_attention_layer_norm/batchnorm/ReadVariableOp2p
6self_attention_layer_norm/batchnorm/mul/ReadVariableOp6self_attention_layer_norm/batchnorm/mul/ReadVariableOp:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence
�
�
5__inference_transformer_decoder_1_layer_call_fn_46427
decoder_sequence
unknown:�U
	unknown_0:U 
	unknown_1:�U
	unknown_2:U 
	unknown_3:�U
	unknown_4:U 
	unknown_5:U�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldecoder_sequenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_43729}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:�������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:�������������������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
5
_output_shapes#
!:�������������������
*
_user_specified_namedecoder_sequence"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
D
input_19
serving_default_input_1:0������������������G
dense>
StatefulPartitionedCall:0�������������������'tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
position_embedding"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_self_attention_layer
_self_attention_layer_norm
_self_attention_dropout
# _feedforward_intermediate_dense
!_feedforward_output_dense
"_feedforward_layer_norm
#_feedforward_dropout"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*_self_attention_layer
+_self_attention_layer_norm
,_self_attention_dropout
#-_feedforward_intermediate_dense
._feedforward_output_dense
/_feedforward_layer_norm
0_feedforward_dropout"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
�
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
Y32
Z33
734
835"
trackable_list_wrapper
�
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
R25
S26
T27
U28
V29
W30
X31
Y32
Z33
734
835"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
`trace_0
atrace_1
btrace_2
ctrace_32�
%__inference_model_layer_call_fn_43875
%__inference_model_layer_call_fn_45030
%__inference_model_layer_call_fn_45107
%__inference_model_layer_call_fn_44704�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0zatrace_1zbtrace_2zctrace_3
�
dtrace_0
etrace_1
ftrace_2
gtrace_32�
@__inference_model_layer_call_and_return_conditional_losses_45520
@__inference_model_layer_call_and_return_conditional_losses_45927
@__inference_model_layer_call_and_return_conditional_losses_44786
@__inference_model_layer_call_and_return_conditional_losses_44868�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0zetrace_1zftrace_2zgtrace_3
�B�
 __inference__wrapped_model_43297input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
hiter

ibeta_1

jbeta_2
	kdecay
llearning_rate7m�8m�9m�:m�;m�<m�=m�>m�?m�@m�Am�Bm�Cm�Dm�Em�Fm�Gm�Hm�Im�Jm�Km�Lm�Mm�Nm�Om�Pm�Qm�Rm�Sm�Tm�Um�Vm�Wm�Xm�Ym�Zm�7v�8v�9v�:v�;v�<v�=v�>v�?v�@v�Av�Bv�Cv�Dv�Ev�Fv�Gv�Hv�Iv�Jv�Kv�Lv�Mv�Nv�Ov�Pv�Qv�Rv�Sv�Tv�Uv�Vv�Wv�Xv�Yv�Zv�"
	optimizer
,
mserving_default"
signature_map
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
strace_02�
<__inference_token_and_position_embedding_layer_call_fn_45936�
���
FullArgSpec,
args$�!
jself
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zstrace_0
�
ttrace_02�
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_45967�
���
FullArgSpec,
args$�!
jself
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zttrace_0
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
9
embeddings"
_tf_keras_layer
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
:
embeddings
:position_embeddings"
_tf_keras_layer
�
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15"
trackable_list_wrapper
�
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_transformer_decoder_layer_call_fn_46004
3__inference_transformer_decoder_layer_call_fn_46041�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_46217
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_46390�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Cgamma
Dbeta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ekernel
Fbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Gkernel
Hbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Igamma
Jbeta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12
X13
Y14
Z15"
trackable_list_wrapper
�
K0
L1
M2
N3
O4
P5
Q6
R7
S8
T9
U10
V11
W12
X13
Y14
Z15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_transformer_decoder_1_layer_call_fn_46427
5__inference_transformer_decoder_1_layer_call_fn_46464�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_46640
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_46813�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Sgamma
Tbeta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ukernel
Vbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Wkernel
Xbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	Ygamma
Zbeta"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_dense_layer_call_fn_46822�
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
 z�trace_0
�
�trace_02�
@__inference_dense_layer_call_and_return_conditional_losses_46852�
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
 z�trace_0
 :
��'2dense/kernel
:�'2
dense/bias
;:9
�'�2'token_and_position_embedding/embeddings
;:9
��2'token_and_position_embedding/embeddings
F:D�U2/transformer_decoder/self_attention/query/kernel
?:=U2-transformer_decoder/self_attention/query/bias
D:B�U2-transformer_decoder/self_attention/key/kernel
=:;U2+transformer_decoder/self_attention/key/bias
F:D�U2/transformer_decoder/self_attention/value/kernel
?:=U2-transformer_decoder/self_attention/value/bias
Q:OU�2:transformer_decoder/self_attention/attention_output/kernel
G:E�28transformer_decoder/self_attention/attention_output/bias
:�2gamma
:�2beta
:
��2kernel
:�2bias
:
��2kernel
:�2bias
:�2gamma
:�2beta
H:F�U21transformer_decoder_1/self_attention/query/kernel
A:?U2/transformer_decoder_1/self_attention/query/bias
F:D�U2/transformer_decoder_1/self_attention/key/kernel
?:=U2-transformer_decoder_1/self_attention/key/bias
H:F�U21transformer_decoder_1/self_attention/value/kernel
A:?U2/transformer_decoder_1/self_attention/value/bias
S:QU�2<transformer_decoder_1/self_attention/attention_output/kernel
I:G�2:transformer_decoder_1/self_attention/attention_output/bias
:�2gamma
:�2beta
:
��2kernel
:�2bias
:
��2kernel
:�2bias
:�2gamma
:�2beta
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_43875input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_45030inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_45107inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_44704input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_45520inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_45927inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_44786input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_44868input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
#__inference_signature_wrapper_44953input_1"�
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
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
<__inference_token_and_position_embedding_layer_call_fn_45936inputs"�
���
FullArgSpec,
args$�!
jself
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_45967inputs"�
���
FullArgSpec,
args$�!
jself
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
90"
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec(
args �
jself
jinputs
	jreverse
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec(
args �
jself
jinputs
	jreverse
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
:0"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec,
args$�!
jself
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec,
args$�!
jself
jinputs
jstart_index
varargs
 
varkw
 
defaults�
` 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
Q
0
1
2
 3
!4
"5
#6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_transformer_decoder_layer_call_fn_46004decoder_sequence"�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
3__inference_transformer_decoder_layer_call_fn_46041decoder_sequence"�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_46217decoder_sequence"�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_46390decoder_sequence"�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
X
;0
<1
=2
>3
?4
@5
A6
B7"
trackable_list_wrapper
X
;0
<1
=2
>3
?4
@5
A6
B7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec]
argsU�R
jself
jquery
jvalue
jkey
jattention_mask
jcache
jcache_update_index
varargs
 
varkw
 
defaults�

 

 

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec]
argsU�R
jself
jquery
jvalue
jkey
jattention_mask
jcache
jcache_update_index
varargs
 
varkw
 
defaults�

 

 

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

;kernel
<bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

=kernel
>bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

?kernel
@bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Akernel
Bbias"
_tf_keras_layer
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
Q
*0
+1
,2
-3
.4
/5
06"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_transformer_decoder_1_layer_call_fn_46427decoder_sequence"�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
5__inference_transformer_decoder_1_layer_call_fn_46464decoder_sequence"�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_46640decoder_sequence"�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_46813decoder_sequence"�
���
FullArgSpec�
args���
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
jself_attention_cache
#j!self_attention_cache_update_index
jcross_attention_cache
$j"cross_attention_cache_update_index
juse_causal_mask
varargs
 
varkw
 7
defaults+�(

 

 

 

 

 

 

 

 

 
p

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
X
K0
L1
M2
N3
O4
P5
Q6
R7"
trackable_list_wrapper
X
K0
L1
M2
N3
O4
P5
Q6
R7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec]
argsU�R
jself
jquery
jvalue
jkey
jattention_mask
jcache
jcache_update_index
varargs
 
varkw
 
defaults�

 

 

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec]
argsU�R
jself
jquery
jvalue
jkey
jattention_mask
jcache
jcache_update_index
varargs
 
varkw
 
defaults�

 

 

 

 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Kkernel
Lbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Mkernel
Nbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Okernel
Pbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

Qkernel
Rbias"
_tf_keras_layer
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
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
�B�
%__inference_dense_layer_call_fn_46822inputs"�
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
�B�
@__inference_dense_layer_call_and_return_conditional_losses_46852inputs"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
�
�	variables
�	keras_api
�aggregate_crossentropy
�_aggregate_crossentropy
�number_of_samples
�_number_of_samples"
_tf_keras_metric
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
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
":   (2aggregate_crossentropy
:  (2number_of_samples
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
%:#
��'2Adam/dense/kernel/m
:�'2Adam/dense/bias/m
@:>
�'�2.Adam/token_and_position_embedding/embeddings/m
@:>
��2.Adam/token_and_position_embedding/embeddings/m
K:I�U26Adam/transformer_decoder/self_attention/query/kernel/m
D:BU24Adam/transformer_decoder/self_attention/query/bias/m
I:G�U24Adam/transformer_decoder/self_attention/key/kernel/m
B:@U22Adam/transformer_decoder/self_attention/key/bias/m
K:I�U26Adam/transformer_decoder/self_attention/value/kernel/m
D:BU24Adam/transformer_decoder/self_attention/value/bias/m
V:TU�2AAdam/transformer_decoder/self_attention/attention_output/kernel/m
L:J�2?Adam/transformer_decoder/self_attention/attention_output/bias/m
:�2Adam/gamma/m
:�2Adam/beta/m
:
��2Adam/kernel/m
:�2Adam/bias/m
:
��2Adam/kernel/m
:�2Adam/bias/m
:�2Adam/gamma/m
:�2Adam/beta/m
M:K�U28Adam/transformer_decoder_1/self_attention/query/kernel/m
F:DU26Adam/transformer_decoder_1/self_attention/query/bias/m
K:I�U26Adam/transformer_decoder_1/self_attention/key/kernel/m
D:BU24Adam/transformer_decoder_1/self_attention/key/bias/m
M:K�U28Adam/transformer_decoder_1/self_attention/value/kernel/m
F:DU26Adam/transformer_decoder_1/self_attention/value/bias/m
X:VU�2CAdam/transformer_decoder_1/self_attention/attention_output/kernel/m
N:L�2AAdam/transformer_decoder_1/self_attention/attention_output/bias/m
:�2Adam/gamma/m
:�2Adam/beta/m
:
��2Adam/kernel/m
:�2Adam/bias/m
:
��2Adam/kernel/m
:�2Adam/bias/m
:�2Adam/gamma/m
:�2Adam/beta/m
%:#
��'2Adam/dense/kernel/v
:�'2Adam/dense/bias/v
@:>
�'�2.Adam/token_and_position_embedding/embeddings/v
@:>
��2.Adam/token_and_position_embedding/embeddings/v
K:I�U26Adam/transformer_decoder/self_attention/query/kernel/v
D:BU24Adam/transformer_decoder/self_attention/query/bias/v
I:G�U24Adam/transformer_decoder/self_attention/key/kernel/v
B:@U22Adam/transformer_decoder/self_attention/key/bias/v
K:I�U26Adam/transformer_decoder/self_attention/value/kernel/v
D:BU24Adam/transformer_decoder/self_attention/value/bias/v
V:TU�2AAdam/transformer_decoder/self_attention/attention_output/kernel/v
L:J�2?Adam/transformer_decoder/self_attention/attention_output/bias/v
:�2Adam/gamma/v
:�2Adam/beta/v
:
��2Adam/kernel/v
:�2Adam/bias/v
:
��2Adam/kernel/v
:�2Adam/bias/v
:�2Adam/gamma/v
:�2Adam/beta/v
M:K�U28Adam/transformer_decoder_1/self_attention/query/kernel/v
F:DU26Adam/transformer_decoder_1/self_attention/query/bias/v
K:I�U26Adam/transformer_decoder_1/self_attention/key/kernel/v
D:BU24Adam/transformer_decoder_1/self_attention/key/bias/v
M:K�U28Adam/transformer_decoder_1/self_attention/value/kernel/v
F:DU26Adam/transformer_decoder_1/self_attention/value/bias/v
X:VU�2CAdam/transformer_decoder_1/self_attention/attention_output/kernel/v
N:L�2AAdam/transformer_decoder_1/self_attention/attention_output/bias/v
:�2Adam/gamma/v
:�2Adam/beta/v
:
��2Adam/kernel/v
:�2Adam/bias/v
:
��2Adam/kernel/v
:�2Adam/bias/v
:�2Adam/gamma/v
:�2Adam/beta/v�
 __inference__wrapped_model_43297�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ789�6
/�,
*�'
input_1������������������
� ";�8
6
dense-�*
dense�������������������'�
@__inference_dense_layer_call_and_return_conditional_losses_46852x78=�:
3�0
.�+
inputs�������������������
� "3�0
)�&
0�������������������'
� �
%__inference_dense_layer_call_fn_46822k78=�:
3�0
.�+
inputs�������������������
� "&�#�������������������'�
@__inference_model_layer_call_and_return_conditional_losses_44786�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ78A�>
7�4
*�'
input_1������������������
p 

 
� "3�0
)�&
0�������������������'
� �
@__inference_model_layer_call_and_return_conditional_losses_44868�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ78A�>
7�4
*�'
input_1������������������
p

 
� "3�0
)�&
0�������������������'
� �
@__inference_model_layer_call_and_return_conditional_losses_45520�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ78@�=
6�3
)�&
inputs������������������
p 

 
� "3�0
)�&
0�������������������'
� �
@__inference_model_layer_call_and_return_conditional_losses_45927�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ78@�=
6�3
)�&
inputs������������������
p

 
� "3�0
)�&
0�������������������'
� �
%__inference_model_layer_call_fn_43875�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ78A�>
7�4
*�'
input_1������������������
p 

 
� "&�#�������������������'�
%__inference_model_layer_call_fn_44704�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ78A�>
7�4
*�'
input_1������������������
p

 
� "&�#�������������������'�
%__inference_model_layer_call_fn_45030�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ78@�=
6�3
)�&
inputs������������������
p 

 
� "&�#�������������������'�
%__inference_model_layer_call_fn_45107�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ78@�=
6�3
)�&
inputs������������������
p

 
� "&�#�������������������'�
#__inference_signature_wrapper_44953�$9:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ78D�A
� 
:�7
5
input_1*�'
input_1������������������";�8
6
dense-�*
dense�������������������'�
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_45967w9:<�9
2�/
)�&
inputs������������������
` 
� "3�0
)�&
0�������������������
� �
<__inference_token_and_position_embedding_layer_call_fn_45936j9:<�9
2�/
)�&
inputs������������������
` 
� "&�#��������������������
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_46640�KLMNOPQRSTUVWXYZ�|
e�b
8�5
decoder_sequence�������������������

 

 

 

 

 

 

 

 

 
p
�

trainingp "3�0
)�&
0�������������������
� �
P__inference_transformer_decoder_1_layer_call_and_return_conditional_losses_46813�KLMNOPQRSTUVWXYZ�|
e�b
8�5
decoder_sequence�������������������

 

 

 

 

 

 

 

 

 
p
�

trainingp"3�0
)�&
0�������������������
� �
5__inference_transformer_decoder_1_layer_call_fn_46427�KLMNOPQRSTUVWXYZ�|
e�b
8�5
decoder_sequence�������������������

 

 

 

 

 

 

 

 

 
p
�

trainingp "&�#��������������������
5__inference_transformer_decoder_1_layer_call_fn_46464�KLMNOPQRSTUVWXYZ�|
e�b
8�5
decoder_sequence�������������������

 

 

 

 

 

 

 

 

 
p
�

trainingp"&�#��������������������
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_46217�;<=>?@ABCDEFGHIJ�|
e�b
8�5
decoder_sequence�������������������

 

 

 

 

 

 

 

 

 
p
�

trainingp "3�0
)�&
0�������������������
� �
N__inference_transformer_decoder_layer_call_and_return_conditional_losses_46390�;<=>?@ABCDEFGHIJ�|
e�b
8�5
decoder_sequence�������������������

 

 

 

 

 

 

 

 

 
p
�

trainingp"3�0
)�&
0�������������������
� �
3__inference_transformer_decoder_layer_call_fn_46004�;<=>?@ABCDEFGHIJ�|
e�b
8�5
decoder_sequence�������������������

 

 

 

 

 

 

 

 

 
p
�

trainingp "&�#��������������������
3__inference_transformer_decoder_layer_call_fn_46041�;<=>?@ABCDEFGHIJ�|
e�b
8�5
decoder_sequence�������������������

 

 

 

 

 

 

 

 

 
p
�

trainingp"&�#�������������������