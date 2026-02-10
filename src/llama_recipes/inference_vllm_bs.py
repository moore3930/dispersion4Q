from vllm import Tokenizer, Model, BeamSearch

# Initialize the model and tokenizer
model = Model.from_pretrained('model-name')
tokenizer = Tokenizer.from_pretrained('model-name')

# Set up beam search parameters
beam_width = 5  # Number of beams
max_tokens = 50  # Maximum number of tokens to generate

# Initialize beam search object
beam_search = BeamSearch(model=model, beam_width=beam_width, max_tokens=max_tokens)

# Example input prompt
prompt = "Once upon a time"

# Tokenize the input
input_tokens = tokenizer.encode(prompt)

# Run beam search
output = beam_search.generate(input_tokens)

# Decode the output tokens back to text
generated_text = tokenizer.decode(output)

print(generated_text)
