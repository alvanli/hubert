This follows the format of the Hugging Face candle library
- It implements the inference of Hubert `https://arxiv.org/abs/2106.07447`, using the weights from `facebook/hubert-large-ls960-ft`
- put this code under `candle\candle-examples\examples\hubert`
- run with this command `cargo run --example hubert --release`