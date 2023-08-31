use candle::{DType, Device, Result,IndexOp,Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};
use serde::Deserialize;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HiddenAct {
    Gelu,
    Relu,
}

struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            HiddenAct::Gelu => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    span: tracing::Span,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        Self { weight, bias, span }
    }

    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let _enter = self.span.enter();
        let w = match x.dims() {
            &[bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
            _ => self.weight.t()?,
        };
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

#[derive(Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
        Self {
            weight,
            bias,
            eps,
            span,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let (_bsize, _seq_len, hidden_size) = x.dims3()?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(2)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(2)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;
        Ok(x)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}
pub fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight_g = vb.get((1, 1, kernel_size), "weight_g")?;
    let weight_v = vb.get((out_c, config.padding, kernel_size), "weight_v")?;
    let norm_v = weight_v.sqr()?.sum_keepdim((0, 1))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    activation_dropout: f64,
    apply_spec_augment: bool,
    architectures: Vec<String>,
    attention_dropout: f64,
    bos_token_id: usize,
    conv_bias: bool,
    conv_dim: Vec<usize>,
    conv_kernel: Vec<usize>,
    conv_stride: Vec<usize>,
    ctc_loss_reduction: String,
    ctc_zero_infinity: bool,
    diversity_loss_weight: f64,
    do_stable_layer_norm: bool,
    eos_token_id: usize,
    feat_extract_activation: HiddenAct,
    feat_extract_dropout: f64,
    feat_extract_norm: String,
    feat_proj_dropout: f64,
    final_dropout: f64,
    gradient_checkpointing: bool,
    hidden_act: HiddenAct,
    hidden_dropout: f64,
    hidden_dropout_prob: f64,
    hidden_size: usize,
    initializer_range: f64,
    intermediate_size: usize,
    layer_norm_eps: f64,
    layerdrop: f64,
    mask_feature_length: usize,
    mask_feature_prob: f64,
    mask_time_length: usize,
    mask_time_prob: f64,
    model_type: Option<String>,
    num_attention_heads: usize,
    num_conv_pos_embedding_groups: usize,
    num_conv_pos_embeddings: usize,
    num_feat_extract_layers: usize,
    num_hidden_layers: usize,
    pad_token_id: u32,
    vocab_size: usize,
    feat_proj_layer_norm: Option<bool>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            activation_dropout: 0.1,
            apply_spec_augment: true,
            architectures: vec![String::from("HubertForCTC")],
            attention_dropout: 0.1,
            bos_token_id: 1,
            conv_bias: true,
            conv_dim: vec![512, 512, 512, 512, 512, 512, 512, 512],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            ctc_loss_reduction: String::from("sum"),
            ctc_zero_infinity: false,
            diversity_loss_weight: 0.1,
            do_stable_layer_norm: true,
            eos_token_id: 2,
            feat_extract_activation: HiddenAct::Gelu,
            feat_extract_dropout: 0.0,
            feat_extract_norm: String::from("layer"),
            feat_proj_dropout: 0.1,
            final_dropout: 0.1,
            gradient_checkpointing: false,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout: 0.1,
            hidden_dropout_prob: 0.1,
            hidden_size: 1024,
            initializer_range: 0.02,
            intermediate_size: 4096,
            layer_norm_eps: 1e-05,
            layerdrop: 0.1,
            mask_feature_length: 10,
            mask_feature_prob: 0.0,
            mask_time_length: 10,
            mask_time_prob: 0.05,
            model_type: Some(String::from("hubert")),
            num_attention_heads: 16,
            num_conv_pos_embedding_groups: 16,
            num_conv_pos_embeddings: 128,
            num_feat_extract_layers: 7,
            num_hidden_layers: 24,
            pad_token_id: 0,
            vocab_size: 32,
            feat_proj_layer_norm: Some(true),
        }
    }
}

impl Config {
    pub fn _hubert_large_ft() -> Self {
        // https://huggingface.co/facebook/hubert-large-ls960-ft/blob/main/config.json
        Self {
            activation_dropout: 0.1,
            apply_spec_augment: true,
            architectures: vec![String::from("HubertForCTC")],
            attention_dropout: 0.1,
            bos_token_id: 1,
            conv_bias: true,
            conv_dim: vec![512, 512, 512, 512, 512, 512, 512, 512],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            ctc_loss_reduction: String::from("sum"),
            ctc_zero_infinity: false,
            diversity_loss_weight: 0.1,
            do_stable_layer_norm: true,
            eos_token_id: 2,
            feat_extract_activation: HiddenAct::Gelu,
            feat_extract_dropout: 0.0,
            feat_extract_norm: String::from("layer"),
            feat_proj_dropout: 0.1,
            final_dropout: 0.1,
            gradient_checkpointing: false,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout: 0.1,
            hidden_dropout_prob: 0.1,
            hidden_size: 1024,
            initializer_range: 0.02,
            intermediate_size: 4096,
            layer_norm_eps: 1e-05,
            layerdrop: 0.1,
            mask_feature_length: 10,
            mask_feature_prob: 0.0,
            mask_time_length: 10,
            mask_time_prob: 0.05,
            model_type: Some(String::from("hubert")),
            num_attention_heads: 16,
            num_conv_pos_embedding_groups: 16,
            num_conv_pos_embeddings: 128,
            num_feat_extract_layers: 7,
            num_hidden_layers: 24,
            pad_token_id: 0,
            vocab_size: 32,
            feat_proj_layer_norm: Some(true),
        }
    }
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    let bias = vb.get(size2, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

fn layer_norm_fn(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (vb.get(size, "gamma"), vb.get(size, "beta")) {
                (weight, bias)
            } else {
                return Err(err);
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

struct HubertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl HubertSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("q_proj"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("v_proj"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("k_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            candle_nn::ops::softmax(&attention_scores, candle::D::Minus1)?
        };
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle::D::Minus2)?;
        Ok(context_layer)
    }
}

struct HubertSelfOutput {
    dense: Linear,
    dropout: Dropout,
    span: tracing::Span,
}

impl HubertSelfOutput {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("out_proj"))?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        hidden_states + input_tensor
    }
}

struct HubertAttention {
    self_attention: HubertSelfAttention,
    self_output: HubertSelfOutput,
    span: tracing::Span,
}

impl HubertAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let self_attention = HubertSelfAttention::load(vb.clone(), config)?;
        let self_output = HubertSelfOutput::load(vb.clone(), config)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_outputs = self.self_attention.forward(hidden_states)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

struct HubertEncoderLayerStableLayerNorm {
    attention: HubertAttention,
    dropout: Dropout,
    layer_norm: LayerNorm,
    feed_forward: HubertFeedForward,
    final_layer_norm: LayerNorm,
    span: tracing::Span,
}

impl HubertEncoderLayerStableLayerNorm {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = HubertAttention::load(vb.pp("attention"), config)?;
        let dropout = Dropout::new(config.hidden_dropout);
        let feed_forward = HubertFeedForward::load(vb.pp("feed_forward"), config)?;
        let layer_norm = layer_norm_fn(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm"),
        )?;
        let final_layer_norm = layer_norm_fn(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("final_layer_norm"),
        )?;

        Ok(Self {
            attention,
            dropout,
            layer_norm,
            feed_forward,
            final_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attn_residual = hidden_states.clone();
        let hidden_states = self.layer_norm.forward(&hidden_states)?;
        let attention_output = self.attention.forward(&hidden_states)?;
        let hidden_states = self.dropout.forward(&attention_output);
        let hidden_states = (attn_residual + hidden_states)?;
        let norm_states = self.final_layer_norm.forward(&hidden_states)?;
        hidden_states + self.feed_forward.forward(&norm_states)
    }
}

struct HubertEncoderStableLayerNorm {
    layers: Vec<HubertEncoderLayerStableLayerNorm>,
    pos_conv_embed: HubertPositionalConvEmbedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl HubertEncoderStableLayerNorm {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                HubertEncoderLayerStableLayerNorm::load(vb.pp(&format!("layers.{index}")), config)
            })
            .collect::<Result<Vec<_>>>()?;
        let pos_conv_embed = HubertPositionalConvEmbedding::load(vb.pp("pos_conv_embed"), config)?;
        let layer_norm = layer_norm_fn(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout);

        let span = tracing::span!(tracing::Level::TRACE, "encoder");

        Ok(HubertEncoderStableLayerNorm {
            layers,
            pos_conv_embed,
            layer_norm,
            dropout,
            span,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // TODO: attention mask?
        let _enter = self.span.enter();
        let position_embeddings = self.pos_conv_embed.forward(&hidden_states)?;
        let hidden_states = (hidden_states + position_embeddings)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        let mut hidden_states = hidden_states.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?
        }
        self.layer_norm.forward(&hidden_states)
    }
}

// TODO: _compute_mask_indices
pub struct HubertLayerNormConvLayer {
    conv: Conv1d,
    activation: HiddenActLayer,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl HubertLayerNormConvLayer {
    fn load(vb: VarBuilder, cfg: &Config, layer_id: usize) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "HubertLayerNormConvLayer");

        let in_conv_dim = if layer_id > 0 {
            cfg.conv_dim[layer_id - 1]
        } else {
            1
        };
        let out_conv_dim = cfg.conv_dim[layer_id];

        let cfg1 = Conv1dConfig {
            padding: 0,
            stride: cfg.conv_stride[layer_id],
            groups: 1,
        };
        let conv = conv1d(
            in_conv_dim,
            out_conv_dim,
            cfg.conv_kernel[layer_id],
            cfg1,
            vb.pp("conv"),
        )?;
        let activation = HiddenActLayer::new(cfg.feat_extract_activation);
        let layer_norm = layer_norm_fn(out_conv_dim, cfg.layer_norm_eps, vb.pp("layer_norm"))?;

        Ok(Self {
            conv,
            activation,
            layer_norm,
            span,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.conv.forward(&hidden_states)?;
        let hidden_states = hidden_states.transpose(1,2)?;
        let hidden_states = self.layer_norm.forward(&hidden_states)?;
        let hidden_states = hidden_states.transpose(1,2)?;
        self.activation.forward(&hidden_states)
    }
}

pub struct HubertPositionalConvEmbedding {
    conv: Conv1d,
    activation: HiddenActLayer,
    span: tracing::Span,
}

impl HubertPositionalConvEmbedding {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "HubertPositionalConvEmbedding");
        let cfg1 = Conv1dConfig {
            padding: cfg.num_conv_pos_embeddings / 2,
            stride: 1,
            groups: cfg.num_conv_pos_embedding_groups,
        };
        let conv = conv1d_weight_norm(
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.num_conv_pos_embeddings,
            cfg1,
            vb.pp("conv"),
        )?;
        let activation = HiddenActLayer::new(cfg.feat_extract_activation);

        Ok(Self {
            conv,
            activation,
            span,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = hidden_states.transpose(1, 2)?;
        let hidden_states = self.conv.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., .., ..hidden_states.dims()[2]-1))?;
        let hidden_states = self.activation.forward(&hidden_states)?;
        hidden_states.transpose(1, 2)
    }
}

struct HubertFeatureEncoder {
    layers: Vec<HubertLayerNormConvLayer>,
    span: tracing::Span,
}

impl HubertFeatureEncoder {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.num_feat_extract_layers)
            .map(|index| {
                HubertLayerNormConvLayer::load(
                    vb.pp(&format!("conv_layers.{index}")),
                    config,
                    index,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(HubertFeatureEncoder { layers, span })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?;
        }
        Ok(hidden_states)
    }
}

struct HubertFeedForward {
    intermediate_dense: Linear,
    output_dense: Linear,
    intermediate_act_fn: HiddenActLayer,
    intermediate_dropout: Dropout,
    output_dropout: Dropout,
    span: tracing::Span,
}

impl HubertFeedForward {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let intermediate_dense = linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("intermediate_dense"),
        )?;
        let output_dense = linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("output_dense"),
        )?;
        let intermediate_act_fn = HiddenActLayer::new(config.feat_extract_activation);
        let intermediate_dropout = Dropout::new(config.activation_dropout);
        let output_dropout = Dropout::new(config.hidden_dropout);

        Ok(Self {
            intermediate_dense,
            output_dense,
            intermediate_act_fn,
            intermediate_dropout,
            output_dropout,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.intermediate_dense.forward(hidden_states)?;
        let hidden_states = self.intermediate_act_fn.forward(&hidden_states)?;
        let hidden_states = self.intermediate_dropout.forward(&hidden_states)?;
        let hidden_states = self.output_dense.forward(&hidden_states)?;
        self.output_dropout.forward(&hidden_states)
    }
}

struct HubertFeatureProjection {
    dense: Linear,
    layer_norm: Option<LayerNorm>,
    dropout: Dropout,
    span: tracing::Span,
}

impl HubertFeatureProjection {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear(
            config.conv_dim[config.conv_dim.len()-1],
            config.hidden_size,
            vb.pp("projection"),
        )?;
        let layer_norm: Option<LayerNorm> = match config.feat_proj_layer_norm {
            Some(feat_proj_layer_norm) => {
                if feat_proj_layer_norm {
                    Some(layer_norm_fn(
                        config.conv_dim[config.conv_dim.len()-1],
                        config.layer_norm_eps,
                        vb.pp("layer_norm"),
                    )?)
                } else {
                    None
                }
            }
            None => None,
        };
        let dropout = Dropout::new(config.feat_proj_dropout);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        if let Some(layer_norm) = &self.layer_norm {
            hidden_states = layer_norm.forward(&hidden_states)?;
        }
        let hidden_states = self.dense.forward(&hidden_states)?;
        self.dropout.forward(&hidden_states)
    }
}

pub struct HubertModel {
    feature_extractor: HubertFeatureEncoder,
    feature_projection: HubertFeatureProjection,
    encoder: HubertEncoderStableLayerNorm,
    pub device: Device,
    span: tracing::Span,
}

impl HubertModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (feature_extractor, feature_projection, encoder) = match (
            HubertFeatureEncoder::load(vb.pp("feature_extractor"), config),
            HubertFeatureProjection::load(vb.pp("feature_projection"), config),
            HubertEncoderStableLayerNorm::load(vb.pp("encoder"), config),
        ) {
            (Ok(feature_extractor), Ok(feature_projection), Ok(encoder)) => {
                (feature_extractor, feature_projection, encoder)
            }
            (Err(err), _, _) | (_, Err(err), _) | (_, _, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(feature_extractor), Ok(feature_projection), Ok(encoder)) = (
                        HubertFeatureEncoder::load(
                            vb.pp(&format!("{model_type}.feature_encoder")),
                            config,
                        ),
                        HubertFeatureProjection::load(
                            vb.pp(&format!("{model_type}.feature_projection")),
                            config,
                        ),
                        HubertEncoderStableLayerNorm::load(
                            vb.pp(&format!("{model_type}.encoder")),
                            config,
                        ),
                    ) {
                        (feature_extractor, feature_projection, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            feature_extractor,
            feature_projection,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, input_values: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_values = input_values.unsqueeze(1)?;
        let extract_features = self.feature_extractor.forward(&input_values)?;
        let extract_features = extract_features.transpose(1, 2)?;
        let hidden_states = self.feature_projection.forward(&extract_features)?;
        let encoder_outputs = self.encoder.forward(&hidden_states)?;
        Ok(encoder_outputs)
    }
}

pub struct HubertForCTC {
    pub model: HubertModel,
    dropout: Dropout,
    pub lm_head: Linear,
    pub config: Config,
    pub device: Device,
    span: tracing::Span,
}

impl HubertForCTC {
    pub fn load(vb: VarBuilder, config: Config) -> Result<Self> {
        let model = HubertModel::load(vb.pp("hubert"), &config)?;
        let dropout = Dropout::new(config.feat_proj_dropout);
        let lm_head = linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            model,
            dropout,
            lm_head,
            config,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, input_values: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.model.forward(&input_values)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }
}
