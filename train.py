from position_embedding import PositionEncoder
from token_embedding import TokenEncoder

training_data = ["test1", "test2"]
sequence_length = len(training_data)

embedding_dim = 512
token_encoder = TokenEncoder(embedding_dim)
position_encoder= PositionEncoder(embedding_dim, sequence_length)


print("\n-----------------------------------------------------")
print("=================Commencing Training================")
print("-----------------------------------------------------\n")

for token in training_data:
    token_encoder.add_token(token)

token_matrix = token_encoder.token_sequence_matrix
position_matrix = position_encoder.position_matrix

print("Training data: ", training_data)

print("\n-----------------------------------------------------")
print("=================Matrix Shapes================")
print("-----------------------------------------------------\n")

print(f"Token matrix shape: {token_matrix.shape}")
print(f"Position matrix shape: {position_matrix.shape}")