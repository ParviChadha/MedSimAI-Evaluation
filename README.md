# Medical Dialogue Analysis Automation Pipeline

This project provides an automated pipeline for analyzing medical dialogues between medical students and patients to evaluate whether students appropriately explored relevant medical history risk factors.

## Overview

The pipeline uses Large Language Models (LLMs) to:
1. Extract symptoms from medical conversation annotations
2. Generate comprehensive criteria lists for evaluation
3. Assess whether medical history topics were properly explored
4. Calculate performance metrics (precision, recall, F1-score)

## Features

- **Multi-Model Support**: Works with OpenAI GPT-4, Claude (Anthropic), Gemini (Google), and Fireworks models
- **Parallel Processing**: Supports multi-threaded processing for faster analysis
- **Comprehensive Metrics**: Tracks both evaluation metrics and token usage
- **Automated Pipeline**: End-to-end processing from raw dialogues to final metrics

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd medical-dialogue-analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (see [API Key Setup](#api-key-setup) below)

## API Key Setup

You have two options for providing API keys:

### Option 1: Environment Variables (Recommended)
Set up environment variables for the models you want to use:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Claude (Anthropic)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Gemini (Google)
export GOOGLE_API_KEY="your-google-api-key"

# For Fireworks
export FIREWORKS_API_KEY="your-fireworks-api-key"
```

### Option 2: Command Line Arguments
Pass the API key directly when running the pipeline:
```bash
python analysis_automation.py --api-key "your-api-key"
```

**Note**: You only need the API key for the model you intend to use. The pipeline will prompt you to select a model during execution.

## Quick Start

### Basic Usage
Run the automation pipeline with default settings:
```bash
python analysis_automation.py
```

### Advanced Usage
```bash
python analysis_automation.py --model openai --start 101 --end 102 --parallel 4 --force
```

**Model Options**: You can specify `openai`, `claude`, `gemini`, or `fireworks` for the `--model` parameter.

**Force Flag**: The `--force` flag clears and reprocesses all existing files, useful when you want to regenerate results with different settings.

### Recommended Approach for Large Batches
For processing many conversations efficiently, run them in small batches using different models, then aggregate:

```bash
# First batch with OpenAI (with --force to ensure clean start)
python analysis_automation.py --model openai --start 101 --end 110 --parallel 4 --force

# Second batch with Claude (no --force needed, continues from where left off)
python analysis_automation.py --model claude --start 111 --end 120 --parallel 4

# Generate aggregated metrics for all conversations (101-120)
python analysis_automation.py --model claude --start 101 --end 120 --parallel 4
```

This approach helps with:
- **Load balancing** across different API providers
- **Cost management** by using different models strategically  
- **Reliability** by processing in smaller, manageable chunks

## Pipeline Components

When you run `analysis_automation.py`, the following process occurs for each conversation:

### 1. **Directory Setup**
Creates organized folder structure:
- `transcripts/` - Formatted conversation transcripts
- `dialogs/` - Individual dialog files
- `annotations/` - Individual annotation files  
- `criteria/` - Generated evaluation criteria
- `results/` - LLM evaluation results
- `metrics/` - Performance metrics
- `token_metrics/` - Token usage tracking

### 2. **File Extraction**
- Splits main `dialogs.json` and `annotations.json` into individual conversation files
- **Files used**: `dialogs.json`, `annotations.json`

### 3. **Transcript Creation**
- Converts dialog JSON to readable transcript format
- **Script**: `create_transcript.py`
- **Output**: `transcripts/transcript{ID}.txt`

### 4. **Criteria Generation** 
- Extracts real symptoms from conversation annotations
- Adds additional symptoms from comprehensive symptom list (`all_symptoms.json`)
- Uses LLM to filter out similar/redundant symptoms
- **Script**: `create_criteria.py`
- **Output**: `criteria/criteria{ID}.json`

### 5. **Medical History Assessment**
- Evaluates transcript against criteria using LLM
- Determines if each symptom/condition was properly assessed
- **Script**: `create_results.py`
- **Output**: `results/results{ID}.json`

### 6. **Metrics Calculation**
- Compares LLM results against ground truth from annotations
- Calculates precision, recall, F1-score
- **Script**: `metrics_evaluation.py`
- **Output**: `metrics/metrics{ID}.json`

### 7. **Aggregation**
- Combines individual metrics into summary statistics
- Generates token usage reports
- **Output**: `metrics/aggregated_metrics.json`, `token_metrics/aggregated_token_metrics.json`

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dialogs` | Main dialogs JSON file | `dialogs.json` |
| `--annotations` | Main annotations JSON file | `annotations.json` |
| `--start` | Starting conversation ID | `101` |
| `--end` | Ending conversation ID | `201` |
| `--parallel` | Number of parallel processes | `4` |
| `--model` | Pre-select model (openai/claude/gemini/fireworks) | Interactive selection |
| `--api-key` | API key for selected model | From environment |
| `--force` | Force reprocessing of existing files (clears all output directories) | `False` |
| `--skip-token-summary` | Skip token usage summary | `False` |

## Individual Script Usage

You can also run individual components:

### Create Transcript
```bash
python create_transcript.py dialogs102.json transcript102.txt
```

### Generate Criteria
```bash
python create_criteria.py annotations102.json criteria102.json 102 all_symptoms.json
```

### Evaluate Results
```bash
python create_results.py transcript102.txt criteria102.json results102.json
```

### Calculate Metrics
```bash
python metrics_evaluation.py 102 --results results102.json --annotations annotations102.json
```

## File Formats

### Input Files
- **dialogs.json**: Conversation utterances with speaker roles
- **annotations.json**: Structured annotations with symptom information
- **all_symptoms.json**: Comprehensive list of medical symptoms/conditions

### Output Files
- **Transcripts**: Human-readable conversation format
- **Criteria**: JSON list of symptoms to evaluate
- **Results**: LLM assessment of which symptoms were discussed
- **Metrics**: Performance statistics and error analysis

## Model Information

### Supported Models
- **OpenAI**: GPT-4o-mini (fast, cost-effective)
- **Claude**: Claude-3-haiku (reliable, good reasoning)
- **Gemini**: Gemini-2.5-flash (Google's latest)
- **Fireworks**: Gemma-3-27b (open-source option)

### Token Usage
The pipeline tracks token consumption for cost monitoring:
- Input tokens (prompts + transcripts)
- Output tokens (model responses)
- Per-conversation and aggregated statistics

## Troubleshooting

### Common Issues

1. **"No API key found"**
   - Ensure environment variables are set correctly
   - Or pass `--api-key` argument

2. **"File not found"**
   - Check that input files exist in the correct location
   - Verify file naming conventions match expected format

3. **Timeout errors**
   - Reduce `--parallel` value for fewer concurrent requests
   - Check network connectivity

4. **JSON parsing errors**
   - Models occasionally return malformed JSON
   - Pipeline includes fallback parsing mechanisms

### Performance Tips

- Use `--parallel 4` or higher for faster processing
- Use `--force` to reprocess existing files
- Monitor token usage to manage API costs
- Use faster models (like GPT-4o-mini) for large batches

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this pipeline in your research, please cite:

```bibtex
[Add citation information here]
```