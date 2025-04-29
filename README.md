# Odoo Addons Documentation Generator

This tool automatically extracts structured API documentation from Odoo module source files using OpenAI's GPT models. It processes Python, XML, JavaScript, and CSS files to create JSON documentation chunks that describe models, fields, methods, views, controllers, and other components.

## Features

- Traverses Odoo module directories and processes each file
- Handles large files by chunking content to fit within token limits
- Processes modules and files in parallel for improved performance
- Saves documentation as structured JSON chunks
- Merges results into comprehensive module documentation
- Provides error handling and logging
- Automatically detects and skips binary files
- Supports HTTP proxy for API requests
- Implements rate limiting for API calls

## Requirements

- Python 3.8+
- OpenAI API key
- Odoo modules to document

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/odoo-addons-doc-generator.git
   cd odoo-addons-doc-generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that the prompt template is available in the `prompts/prompt.md` file.

## Usage

### Basic Usage

```bash
python odoo_docs_generator.py --addons-path /path/to/odoo/addons --output-path ./output --api-key YOUR_API_KEY
```

### Options

- `--addons-path`: Path to the Odoo addons directory (required)
- `--output-path`: Path to save the generated documentation (required)
- `--api-key`: OpenAI API key (required)
- `--model`: OpenAI model to use (default: gpt-4o)
- `--max-workers`: Maximum number of parallel workers (default: 4)
- `--max-file-tokens`: Maximum number of tokens per file before warning (default: 3500)
- `--batch-size`: Number of modules to process in each batch (default: 10, set to 0 to disable batching)
- `--rate-limit`: API request rate limit in requests per second (default: 10.0, set to 0 to disable rate limiting)
- `--base-url`: Custom API base URL (e.g., https://api.x.ai/v1 for x.ai models)
- `--merge-only`: Only merge existing results without processing files

### Example

```bash
# Process all modules in the addons directory
python odoo_docs_generator.py --addons-path /path/to/odoo/addons --output-path ./output --api-key sk-xxxx

# Use a custom API base URL for alternative models (like x.ai)
python odoo_docs_generator.py --addons-path /path/to/odoo/addons --model grok-3-beta --api-key sk-xxxx --base-url https://api.x.ai/v1

# Process with rate limiting
python odoo_docs_generator.py --addons-path /path/to/odoo/addons --api-key sk-xxxx --rate-limit 5

# Only merge existing results (useful after partial processing)
python odoo_docs_generator.py --addons-path /path/to/odoo/addons --output-path ./output --api-key sk-xxxx --merge-only
```

### Using HTTP Proxy

The tool automatically uses HTTP proxy settings from environment variables:

**On Windows (CMD):**
```cmd
set HTTP_PROXY=127.0.0.1:7890
set HTTPS_PROXY=127.0.0.1:7890
```

**On Windows (PowerShell):**
```powershell
$env:HTTP_PROXY = "127.0.0.1:7890"
$env:HTTPS_PROXY = "127.0.0.1:7890"
```

**On Linux/macOS:**
```bash
export HTTP_PROXY=127.0.0.1:7890
export HTTPS_PROXY=127.0.0.1:7890
```

After setting the proxy environment variables, run the tool normally.

## Using with Batch Scripts

For convenience, you can use the provided batch scripts:

### On Windows:

```cmd
run.bat --addons-path C:\path\to\addons --http-proxy http://127.0.0.1:7890
```

### On Unix/Linux/macOS:

```bash
./run.sh --addons-path /path/to/addons --http-proxy http://127.0.0.1:7890
```

## Output Structure

The tool generates two types of output:

1. Individual JSON files for each processed source file:
   ```
   output/
     ├── module_name/
     │   ├── models/
     │   │   ├── file1.py.json
     │   │   └── file2.py.json
     │   ├── views/
     │   │   └── view1.xml.json
     │   └── static/
     │       ├── src/
     │       │   └── file.js.json
     │       └── css/
     │           └── style.css.json
   ```

2. A merged JSON file containing all documentation chunks for the module:
   ```
   output/
     ├── module_name/
     │   └── merged.json
   ```

Each JSON chunk follows this structure:

```json
{
  "codeTitle": "Short and clear title",
  "codeDescription": "Concise and detailed description",
  "codePath": "module_name/path/to/file.py#L1-L10",
  "moduleContext": "Broader module context",
  "codeType": "model|method|view|component|etc.",
  "codeContent": {
    "language": "python|javascript|xml|css|scss",
    "code": "Actual code here"
  }
}
```

## Logging

Logs are saved to `odoo_docs_generator.log` in the current directory.

## Error Handling

If the LLM response cannot be parsed as JSON, the raw response is saved to:
```
output/module_name/errors/filename_error.txt
```

## Performance Considerations

- Adjust `max_file_tokens` parameter based on your model's context limit (default: 3500)
- Use `rate_limit` to comply with API provider's rate limits (e.g., 10 requests per second)
- If behind a firewall, use the `http_proxy` parameter to route requests through your proxy
- Increase or decrease `max_workers` based on your system's capabilities and API rate limits
- For very large codebases, use the `batch_size` parameter to process modules in batches 