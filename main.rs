#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
mod model;

use std::path::PathBuf;
use regex::Regex;
use anyhow::{Result};

use candle::{Tensor, IndexOp, D};
use serde_json::{Map, Value};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::{Config, HubertForCTC, DTYPE};

// https://huggingface.co/facebook/hubert-large-ls960-ft/blob/main/preprocessor_config.json
const SAMPLE_RATE: usize = 16000;

// https://huggingface.co/facebook/hubert-large-ls960-ft/blob/main/tokenizer_config.json
const UNK_TOKEN: &str = "<unk>";
const PAD_TOKEN: &str = "<pad>";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Run offline (you must have the files already cached)
    #[arg(long)]
    offline: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The input to be processed, in wav format, will default to `jfk.wav`. Alternatively
    /// this can be set to sample:jfk, sample:gb1, ... to fetch a sample from the following
    /// repo: https://huggingface.co/datasets/Narsil/candle_demo/
    #[arg(long)]
    input: Option<String>,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,
}

fn reverse_json(json: &Value) -> Value {
    match json {
        Value::Object(obj) => {
            let mut new_obj = Map::new();
            for (key, value) in obj.iter() {
                new_obj.insert(value.to_string(), Value::String(key.to_string()));
            }
            Value::Object(new_obj)
        }
        _ => json.clone(),
    }
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(HubertForCTC, PathBuf, Value)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "alvanlii/hubert-large-ls960-ft-rust".to_string();
        let path = std::path::PathBuf::from(default_model.clone());

        let default_revision = "main".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let (config_filename, weights_filename, tokenizer_filename, input) = if path.exists() {
            let mut config_filename = path.clone();
            config_filename.push("config.json");
            let mut tokenizer_filename = path.clone();
            tokenizer_filename.push("vocab.json");
            let mut model_filename = path;
            model_filename.push("model.safetensors");
            let input_path = self.input.clone().expect("Please specific file path");
            (
                config_filename,
                model_filename,
                tokenizer_filename,
                std::path::PathBuf::from(input_path),
            )
        } else {
            let api = Api::new()?;
            let dataset = api.dataset("Narsil/candle-examples".to_string());
            let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
            let sample = if let Some(input) = &self.input {
                if let Some(sample) = input.strip_prefix("sample:") {
                    dataset.get(&format!("samples_{sample}.wav"))?
                } else {
                    std::path::PathBuf::from(input)
                }
            } else {
                println!("No audio file submitted: Downloading https://huggingface.co/datasets/Narsil/candle_demo/blob/main/samples_jfk.wav");
                dataset.get("samples_jfk.wav")?
            };
            (
                repo.get("config.json")?,
                repo.get("model.safetensors")?,
                repo.get("vocab.json")?,
                sample,
            )
        };
        let config: Config = Config::_hubert_large_ft();
        println!("{:?}", config);

        let vocab_json:Value = serde_json::from_str(&std::fs::read_to_string(tokenizer_filename)?)?;
        let tokenizer = reverse_json(&vocab_json);

        let weights = unsafe { candle::safetensors::MmapedFile::new(weights_filename)? };
        let weights = weights.deserialize()?;
        let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, &device);
        let model = HubertForCTC::load(vb, config)?;
        Ok((model, input, tokenizer))
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let (model, input, tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.device;

    // read audio
    let mut input = std::fs::File::open(input)?;
    let (header, data) = wav::read(&mut input)?;
    println!("loaded wav data: {header:?}");
    if header.sampling_rate != SAMPLE_RATE as u32 {
        anyhow::bail!("wav file must have a {} sampling rate", SAMPLE_RATE)
    }
    let data = data.as_sixteen().expect("expected 16 bit wav file");
    let pcm_data: Vec<_> = data[..data.len() / header.channel_count as usize]
        .iter()
        .map(|v| *v as f32 / 32768.)
        .collect();
    println!("pcm data loaded {}", pcm_data.len());

    let input_vec = Tensor::from_vec(pcm_data.clone(), (1, pcm_data.len()), &device)?;
    let start = std::time::Instant::now();
    println!("Starting");
    let output_logits = model.forward(&input_vec)?;
    let output = output_logits.argmax(D::Minus1)?;
    println!("Elapsed {:?}", start.elapsed());
    let output = output.squeeze(0)?;
    let output_ids = output.to_vec1::<u32>()?;
    // println!("{:?}", output.dims());
    // println!("{:?}", output_ids);

    let mut result = String::new();
    for key in output_ids.iter() {
        if let Some(value) = tokenizer.get(key.to_string()) {
            let mut curr_str = value.to_string();
            curr_str = curr_str.replace("\"", "");
            if curr_str != PAD_TOKEN {
                result.push_str(&curr_str);
            }
        }
    }
    let re = Regex::new(r"\|+").unwrap();
    result = re.replace_all(&result, " ").to_string();
    println!("{}", result);

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
