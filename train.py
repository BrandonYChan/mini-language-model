from position_embedding import PositionEncoder
from token_embedding import TokenEncoder
from query_key_value import QKV
from formatted_printing import PrintFormatter

# Initialize hyperparameters
num_heads = 6
embedding_dim = 512

sequence = ["The", "big", "dog"]
sequence_length = len(sequence)

token_encoder = TokenEncoder(embedding_dim)
position_encoder= PositionEncoder(embedding_dim, sequence_length)

PrintFormatter.print_header("Commencing Training")

for token in sequence:
    token_encoder.add_token(token)

token_matrix = token_encoder.token_sequence_matrix
position_matrix = position_encoder.position_matrix

print("Training data: ", sequence)

PrintFormatter.print_header("Preview Embedding Matrices")

print(f"""
First 3x3 of token embedding matrix
{token_matrix[:3,:3]}

First 3x3 of positional embedding matrix
{position_matrix[:3,:3]}

Token matrix shape: {token_matrix.shape}
Position matrix shape: {position_matrix.shape}
""")


PrintFormatter.print_header("Initializing QKV")

qkv = QKV(embedding_dim, sequence_length, num_heads)

print("Q weight matrix: \n", qkv.q_weights[:3, :3])
print("K weight matrix: \n", qkv.k_weights[:3, :3])
print("V weight matrix: \n", qkv.v_weights[:3, :3])

PrintFormatter.print_header("Passing to Model")
